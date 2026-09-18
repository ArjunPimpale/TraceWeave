"""Characterization tests for multi-source retrieval candidate merging."""

from __future__ import annotations

from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.retrieval.merger import CandidateMerger


def _chunk(chunk_id: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_document="source.md",
        source_hash="hash",
        source_type=SourceType.MARKDOWN,
        chunk_index=0,
        source_locator="section 1",
        normalized_text=f"text for {chunk_id}",
        char_count=len(f"text for {chunk_id}"),
    )


def test_merge_combines_methods_and_keeps_id_match_expansion() -> None:
    shared = _chunk("shared")
    vector_only = _chunk("vector")
    id_only = _chunk("id")

    results = CandidateMerger(vector_weight=0.7, bm25_weight=0.3).merge(
        vector_results=[(shared, 0.8), (vector_only, 0.7)],
        bm25_results=[(shared, 0.5)],
        id_match_results=[id_only],
        top_k=1,
    )

    assert [result.chunk_id for result in results] == ["id", "shared", "vector"]
    shared_result = next(result for result in results if result.chunk_id == "shared")
    assert shared_result.combined_score == 0.71
    assert shared_result.retrieval_method == "bm25,vector"
    assert results[0].id_match is True
    assert results[0].combined_score == 0.95
