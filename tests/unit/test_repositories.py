"""Characterization tests for SQLite repository write behavior."""

from __future__ import annotations

import json

from pecs.models.correlation_result import CorrelationResult, CorrelationStatus
from pecs.models.extraction_result import ExtractionResult, EntityType


def test_evidence_insert_and_batch_preserve_metadata_and_idempotency(
    evidence_repo, sample_extraction_result
) -> None:
    sample_extraction_result.metadata = {"confidence": 0.9}
    second = ExtractionResult(
        entity_type=EntityType.IMPLEMENTATION,
        entity_id="impl-auth",
        text="JWT authentication was implemented.",
        source_document="implementation.py",
        chunk_id="chunk-implementation",
    )

    inserted_id = evidence_repo.insert(sample_extraction_result)
    batch_ids = evidence_repo.insert_batch([sample_extraction_result, second])

    rows = evidence_repo.get_all()
    assert inserted_id is not None
    assert batch_ids == [inserted_id + 1]
    assert len(rows) == 2
    saved = next(row for row in rows if row["entity_id"] == "R1-auth")
    assert json.loads(saved["metadata"]) == {"confidence": 0.9}


def test_correlation_batch_serializes_supporting_chunk_ids(correlation_repo) -> None:
    result = CorrelationResult(
        correlation_id="correlation-1",
        requirement_entity_id="R1",
        evidence_entity_id="impl-R1",
        status=CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION,
        resolution_method="deterministic_rule",
        rule_name="exact_requirement_id_match",
        confidence=0.9,
        supporting_chunk_ids=["chunk-1", "chunk-2"],
        reasoning="Explicit evidence.",
    )

    assert correlation_repo.insert_batch([result]) == 1
    row = correlation_repo.get_all()[0]
    assert json.loads(row["supporting_chunk_ids"]) == ["chunk-1", "chunk-2"]
