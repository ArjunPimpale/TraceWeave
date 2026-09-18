"""Legacy selection and atomic traceability persistence regressions."""

from __future__ import annotations

import sqlite3

import pytest

from pecs.models.correlation_result import CorrelationResult, CorrelationStatus
from pecs.models.extraction_result import EntityType, ExtractionResult
from pecs.models.traceability import AssessmentEnvelope, AssessmentReference
from pecs.store.traceability_repo import TraceabilityRepo
from pecs.traceability.reader import TraceabilityReader
from pecs.traceability.matrix import MatrixBuilder


def _requirement(name, cid):
    return ExtractionResult(entity_type=EntityType.REQUIREMENT, entity_id=name,
                            text="A requirement", source_document="spec.pdf", chunk_id=cid)


def _correlation(req, evidence, confidence, status=CorrelationStatus.PARTIALLY_IMPLEMENTED):
    return CorrelationResult(requirement_entity_id=req, evidence_entity_id=evidence,
        status=status, resolution_method="deterministic_rule", confidence=confidence)


def test_legacy_latest_tie_is_deterministic_and_duplicate_ids_are_not_fanned_out(
    tmp_db, evidence_repo, correlation_repo,
):
    evidence_repo.insert_batch([_requirement("R1", "first"), _requirement("R1", "second")])
    correlation_repo.insert_batch([_correlation("R1", "(none)", 0.7),
                                   _correlation("R1", "(none)", 0.9)])
    snapshot = TraceabilityReader(TraceabilityRepo(tmp_db)).load("legacy")
    assert len(snapshot["requirements"]) == 2
    assert all(row["status"] is None for row in snapshot["requirements"])
    assert len(snapshot["unlinked_assessments"]) == 2


def test_invalid_typed_reference_rolls_back_the_whole_run(tmp_db, evidence_repo):
    evidence_repo.insert(_requirement("R1", "req"))
    req = evidence_repo.get_all_requirements()[0]
    repo = TraceabilityRepo(tmp_db)
    result = _correlation("R1", "impl", 0.9)
    envelope = AssessmentEnvelope(result, req["id"], "requirement_assessment", 1,
                                  [AssessmentReference("explicit_implementation", "evidence",
                                                       evidence_row_id=999999)])
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_run(requirements=[req], envelopes=[envelope], candidates={}, snapshots={},
                      states={req["id"]: ("assessed", [])}, evidence_watermark=req["id"],
                      started_at="2026-01-01T00:00:00Z")
    assert repo.list_runs() == []
    assert tmp_db.connect().execute("SELECT count(*) FROM correlation").fetchone()[0] == 0


def test_migration_preserves_existing_rows_and_dataset_identity(tmp_db, evidence_repo):
    evidence_repo.insert(_requirement("R1", "req"))
    repo = TraceabilityRepo(tmp_db)
    dataset = repo.dataset_uuid()
    tmp_db.initialize_schema()
    assert TraceabilityRepo(tmp_db).dataset_uuid() == dataset
    assert len(evidence_repo.get_all_requirements()) == 1


def test_conflicting_same_run_assessments_remain_visible_and_matrix_uses_selected(tmp_db, evidence_repo):
    evidence_repo.insert(_requirement("R1", "req"))
    req = evidence_repo.get_all_requirements()[0]
    low = _correlation("R1", "(none)", 0.4, CorrelationStatus.PARTIALLY_IMPLEMENTED)
    high = _correlation("R1", "(none)", 0.9, CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED)
    repo = TraceabilityRepo(tmp_db)
    run_id = repo.save_run(requirements=[req], envelopes=[
        AssessmentEnvelope(low, req["id"], "requirement_assessment", 1),
        AssessmentEnvelope(high, req["id"], "requirement_assessment", 2),
    ], candidates={}, snapshots={}, states={req["id"]: ("assessed", [])},
       evidence_watermark=req["id"], started_at="2026-01-01T00:00:00Z")
    snapshot = TraceabilityReader(repo).load(run_id)
    row = snapshot["requirements"][0]
    assert row["conflicting"] is True
    assert set(row["assessment_ids"]) == {low.correlation_id, high.correlation_id}
    assert row["selected_correlation_id"] == high.correlation_id
    matrix = MatrixBuilder._matrix_from_snapshot(snapshot)
    assert matrix.rows[0].status == CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED
