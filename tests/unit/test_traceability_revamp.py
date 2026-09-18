"""End-to-end SQLite traceability contracts with synthetic evidence and chunks."""

from __future__ import annotations

from pecs.graph.projection import build_projection, evidence_key
from pecs.graph.views import select_view
from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.models.extraction_result import EntityType, ExtractionResult
from pecs.models.retrieved_evidence import RetrievedEvidence
from pecs.models.traceability import RetrievalOutcome
from pecs.store.traceability_repo import TraceabilityRepo
from pecs.traceability.reader import TraceabilityReader
from pecs.traceability.service import TraceabilityService
from pecs.traceability.matrix import MatrixBuilder


class Chunks:
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def get_by_ids(self, ids):
        found = [self.chunks[cid] for cid in ids if cid in self.chunks]
        return {"ids": [c.chunk_id for c in found],
                "documents": [c.normalized_text for c in found],
                "metadatas": [c.to_chroma_metadata() for c in found]}


def chunk(cid, filename, text, typ=SourceType.PYTHON):
    return EvidenceChunk(cid, filename, cid, typ, 0, "lines 1-10", text, len(text),
                         {"element_name": "train_model"})


def row(kind, name, text, doc, cid, link=None):
    return ExtractionResult(entity_type=kind, entity_id=name, text=text,
        linked_requirement=link, source_document=doc, chunk_id=cid)


def test_saved_run_keeps_all_explicit_participants_and_source_ownership(tmp_db, evidence_repo):
    evidence_repo.insert_batch([
        row(EntityType.REQUIREMENT, "R1", "Train a model", "spec.pdf", "req", None),
        row(EntityType.IMPLEMENTATION, "impl-a", "train_model is implemented", "a.py", "a", "R1"),
        row(EntityType.IMPLEMENTATION, "impl-b", "model training exists", "b.py", "b", "R1"),
        row(EntityType.EVALUATION, "eval-good", "good training", "review.pdf", "good", "R1"),
        row(EntityType.EVALUATION, "eval-bad", "training failed", "review.pdf", "bad", "R1"),
    ])
    req = evidence_repo.get_all_requirements()[0]
    saved = TraceabilityRepo(tmp_db)
    service = TraceabilityService(evidence_repo=evidence_repo, trace_repo=saved,
        chroma=Chunks([chunk("req", "spec.pdf", "Train a model", SourceType.PDF),
                       chunk("a", "a.py", "def train_model(): pass"),
                       chunk("b", "b.py", "def train_model(): return 1"),
                       chunk("good", "review.pdf", "good training", SourceType.PDF),
                       chunk("bad", "review.pdf", "training failed", SourceType.PDF)]))
    run_id = service.run({req["id"]: RetrievalOutcome([])}, use_stage2=False)
    snapshot = TraceabilityReader(saved).load(run_id)
    assert snapshot["requirements"][0]["status"] == "IMPLEMENTED_BUT_NEGATIVELY_EVALUATED"
    assert snapshot["requirements"][0]["implementation_entities"] == 2
    assert snapshot["requirements"][0]["evaluation_entities"] == 2
    assert snapshot["requirements"][0]["evidence_count"] == 4
    projection = build_projection(snapshot)
    requirement_key = evidence_key(snapshot["dataset_uuid"], req["id"])
    direct = [edge for edge in projection["edges"].values() if edge["target"] == requirement_key
              and edge["kind"] in ("REFERENCES_REQUIREMENT", "ASSESSES_REQUIREMENT")]
    assert len(direct) == 4
    for edge in direct:
        assert projection["nodes"][edge["source"]]["kind"] in ("Implementation", "Evaluation")
    view = select_view(projection, requirement_ids=[requirement_key])
    assert len(view["nodes"]) >= 5
    assert all(edge["source"] in {node["id"] for node in view["nodes"]} for edge in view["edges"])


def test_raw_retrieval_chunk_remains_a_chunk_and_survives_reload(tmp_db, evidence_repo):
    evidence_repo.insert(row(EntityType.REQUIREMENT, "R2", "Provide a training pipeline", "spec.pdf", "req"))
    req = evidence_repo.get_all_requirements()[0]
    raw = chunk("raw-code", "train.py", "def train_model():\n    return model")
    candidate = RetrievedEvidence(chunk=raw, combined_score=0.9, vector_score=0.9,
                                  retrieval_method="vector")
    saved = TraceabilityRepo(tmp_db)
    service = TraceabilityService(evidence_repo=evidence_repo, trace_repo=saved,
                                  chroma=Chunks([raw]))
    run_id = service.run({req["id"]: RetrievalOutcome([candidate])}, use_stage2=False)
    snapshot = TraceabilityReader(saved).load(run_id)
    projection = build_projection(snapshot)
    context_edges = [edge for edge in projection["edges"].values()
                     if edge["kind"] == "ASSESSMENT_CONTEXT"]
    assert len(context_edges) == 1
    assert projection["nodes"][context_edges[0]["source"]]["kind"] == "Chunk"
    assert snapshot["chunks"]["raw-code"]["normalized_text"] == raw.normalized_text
    assert snapshot["requirements"][0]["retrieved_candidate_chunks"] == 1


def test_missing_retrieval_is_not_classified_as_no_implementation(tmp_db, evidence_repo):
    evidence_repo.insert(row(EntityType.REQUIREMENT, "R3", "Provide an API", "spec.pdf", "req"))
    req = evidence_repo.get_all_requirements()[0]
    saved = TraceabilityRepo(tmp_db)
    service = TraceabilityService(evidence_repo=evidence_repo, trace_repo=saved, chroma=Chunks([]))
    run_id = service.run({req["id"]: RetrievalOutcome([], "search unavailable")}, use_stage2=False)
    snapshot = TraceabilityReader(saved).load(run_id)
    assert snapshot["requirements"][0]["status"] is None
    assert snapshot["requirements"][0]["workflow_state"] == "retrieval_failed"
    assert "search unavailable" in snapshot["requirements"][0]["diagnostics"][0]


def test_matrix_fresh_and_reloaded_use_same_saved_run(tmp_db, evidence_repo, correlation_repo):
    evidence_repo.insert_batch([
        row(EntityType.REQUIREMENT, "R4", "Provide API", "spec.pdf", "req"),
        row(EntityType.IMPLEMENTATION, "impl-api", "API routes exist", "routes.py", "impl", "R4"),
    ])
    req = evidence_repo.get_all_requirements()[0]
    builder = MatrixBuilder(evidence_repo=evidence_repo, correlation_repo=correlation_repo,
        chroma_store=Chunks([chunk("req", "spec.pdf", "Provide API", SourceType.PDF),
                             chunk("impl", "routes.py", "def api(): pass")]))
    fresh = builder.build({req["id"]: RetrievalOutcome([])}, use_stage2=False)
    loaded = builder.load_latest(fresh.run_id)
    assert fresh.snapshot_key == loaded.snapshot_key
    assert fresh.to_dict() == loaded.to_dict()
    assert fresh.rows[0].implementation_entities == 1
    assert fresh.rows[0].source_documents == ["routes.py"]
