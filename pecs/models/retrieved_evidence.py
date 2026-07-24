"""
RetrievedEvidence — wraps an EvidenceChunk with retrieval-specific scores.

The retrieval pipeline uses multiple methods (vector, BM25, ID-matching).
RetrievedEvidence preserves each method's score so downstream stages
can make informed relevance decisions and the evaluation pipeline can
track which method surfaced each chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pecs.models.evidence_chunk import EvidenceChunk


@dataclass
class RetrievedEvidence:
    """
    An EvidenceChunk enriched with retrieval scores.

    Attributes:
        chunk: The underlying EvidenceChunk that was retrieved.
        vector_score: Cosine similarity score from ChromaDB (0.0–1.0).
                      ChromaDB returns distances; this is already converted
                      to similarity (1 - distance for cosine).
        bm25_score: BM25 score if keyword retrieval was used; None otherwise.
                    BM25 scores are normalized to 0–1 range by the merger.
        id_match: True if this chunk was surfaced by requirement-ID matching
                  (explicit mention of a requirement identifier in the text).
        combined_score: Weighted combination score used for final ranking.
                        Formula: see Section 10.6 of the implementation plan.
        retrieval_method: Comma-separated string of methods that surfaced this
                          chunk (e.g., "vector", "bm25", "id_match",
                          "vector,bm25", etc.).
    """

    chunk: EvidenceChunk
    vector_score: float = 0.0
    bm25_score: float | None = None
    id_match: bool = False
    combined_score: float = 0.0
    retrieval_method: str = "unknown"

    @property
    def chunk_id(self) -> str:
        """Convenience property to access the chunk's ID directly."""
        return self.chunk.chunk_id

    @property
    def source_document(self) -> str:
        """Convenience property to access the chunk's source document."""
        return self.chunk.source_document

    @property
    def normalized_text(self) -> str:
        """Convenience property to access the chunk's normalized text."""
        return self.chunk.normalized_text

    def __repr__(self) -> str:
        return (
            f"RetrievedEvidence(chunk_id={self.chunk_id[:8]}…, "
            f"combined={self.combined_score:.3f}, "
            f"method={self.retrieval_method!r})"
        )
