"""
Candidate merger — combines and deduplicates results from all retrieval methods.

Implements the scoring formula from Section 10.6 of the implementation plan:
- vector only: combined_score = vector_score
- BM25 only: combined_score = normalized_bm25_score
- both: combined_score = 0.7 * vector_score + 0.3 * normalized_bm25_score
- ID match: combined_score = max(combined_score, 0.95)
"""

from __future__ import annotations

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk
from pecs.models.retrieved_evidence import RetrievedEvidence

logger = get_logger(__name__)


class CandidateMerger:
    """
    Merges results from vector search, BM25, and ID-matching into a
    deduplicated, scored list of RetrievedEvidence objects.
    """

    def __init__(
        self,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        id_match_boost: float = 0.95,
    ) -> None:
        self.vector_weight = vector_weight or settings.VECTOR_WEIGHT
        self.bm25_weight = bm25_weight or settings.BM25_WEIGHT
        self.id_match_boost = id_match_boost

    def merge(
        self,
        vector_results: list[tuple[EvidenceChunk, float]],
        bm25_results: list[tuple[EvidenceChunk, float]],
        id_match_results: list[EvidenceChunk],
        top_k: int | None = None,
    ) -> list[RetrievedEvidence]:
        """
        Merge results from all retrieval methods.

        Args:
            vector_results: [(chunk, cosine_similarity_score), ...].
            bm25_results: [(chunk, normalized_bm25_score_0_to_1), ...].
            id_match_results: Chunks found by requirement-ID matching.
            top_k: Maximum results to return. Defaults to RETRIEVAL_TOP_K.
                   Expanded to top_k + 5 if ID matches are present.

        Returns:
            Sorted list of RetrievedEvidence, highest combined_score first.
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K

        # Expand top_k if ID matches exist to avoid displacing them
        if id_match_results:
            effective_top_k = top_k + 5
        else:
            effective_top_k = top_k

        # Build lookup maps: chunk_id → (chunk, scores)
        candidates: dict[str, dict] = {}

        for chunk, score in vector_results:
            cid = chunk.chunk_id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "bm25_score": None,
                    "id_match": False,
                    "methods": [],
                }
            candidates[cid]["vector_score"] = score
            candidates[cid]["methods"].append("vector")

        for chunk, score in bm25_results:
            cid = chunk.chunk_id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "bm25_score": None,
                    "id_match": False,
                    "methods": [],
                }
            candidates[cid]["bm25_score"] = score
            candidates[cid]["methods"].append("bm25")

        for chunk in id_match_results:
            cid = chunk.chunk_id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "bm25_score": None,
                    "id_match": False,
                    "methods": [],
                }
            candidates[cid]["id_match"] = True
            if "id_match" not in candidates[cid]["methods"]:
                candidates[cid]["methods"].append("id_match")

        # Compute combined scores
        results: list[RetrievedEvidence] = []
        for cid, data in candidates.items():
            combined = self._compute_combined_score(
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                id_match=data["id_match"],
            )
            re = RetrievedEvidence(
                chunk=data["chunk"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                id_match=data["id_match"],
                combined_score=combined,
                retrieval_method=",".join(sorted(set(data["methods"]))),
            )
            results.append(re)

        # Sort by combined_score descending
        results.sort(key=lambda r: r.combined_score, reverse=True)

        logger.info(
            "Merged retrieval results",
            extra={"context": {
                "total_candidates": len(candidates),
                "vector_candidates": len(vector_results),
                "bm25_candidates": len(bm25_results),
                "id_match_candidates": len(id_match_results),
                "returned": min(effective_top_k, len(results)),
            }},
        )

        return results[:effective_top_k]

    def _compute_combined_score(
        self,
        vector_score: float,
        bm25_score: float | None,
        id_match: bool,
    ) -> float:
        """
        Compute the combined relevance score for a single candidate.

        Formula (Section 10.6):
        - vector only: combined = vector_score
        - BM25 only: combined = bm25_score
        - both: combined = 0.7 * vector_score + 0.3 * bm25_score
        - ID match: combined = max(combined, 0.95)
        """
        has_vector = vector_score > 0.0
        has_bm25 = bm25_score is not None

        if has_vector and has_bm25:
            combined = self.vector_weight * vector_score + self.bm25_weight * bm25_score
        elif has_vector:
            combined = vector_score
        elif has_bm25:
            combined = bm25_score
        else:
            combined = 0.0

        if id_match:
            combined = max(combined, self.id_match_boost)

        return round(min(combined, 1.0), 4)
