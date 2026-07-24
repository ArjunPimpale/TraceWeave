"""
BM25 keyword retrieval index.

Builds a BM25 index over all EvidenceChunk texts at ingestion time.
At query time, tokenizes the query and retrieves top-K matches.
Supplements vector search by catching exact keyword matches that
semantic similarity might miss.
"""

from __future__ import annotations

import re
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk

logger = get_logger(__name__)

# Simple word tokenizer (lowercase, alphanumeric + hyphens)
_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """
    In-memory BM25 index over EvidenceChunk texts.

    Built at ingestion time and held in memory. For the expected scale
    (thousands of chunks), this is fast enough without any persistence.

    The index is rebuilt whenever new chunks are added.
    """

    def __init__(self) -> None:
        self._chunks: list[EvidenceChunk] = []
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

    def build(self, chunks: list[EvidenceChunk]) -> None:
        """
        Build the BM25 index from a list of EvidenceChunks.

        Args:
            chunks: All EvidenceChunks currently in the system.
        """
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        self._corpus_tokens = [_tokenize(c.normalized_text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        logger.info(
            "BM25 index built",
            extra={"context": {"document_count": len(chunks)}},
        )

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        source_type_filter: list[str] | None = None,
    ) -> list[tuple[EvidenceChunk, float]]:
        """
        Query the BM25 index.

        Args:
            query_text: The query string.
            top_k: Number of results to return.
            source_type_filter: Optional list of SourceType values to restrict results.

        Returns:
            List of (EvidenceChunk, bm25_score) tuples, sorted by score descending.
            Returns empty list if the index has not been built.
        """
        if self._bm25 is None or not self._chunks:
            return []

        top_k = top_k or settings.BM25_TOP_K
        query_tokens = _tokenize(query_text)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with chunks and filter
        scored: list[tuple[EvidenceChunk, float]] = []
        for chunk, score in zip(self._chunks, scores):
            if source_type_filter and chunk.source_type.value not in source_type_filter:
                continue
            if score > 0:
                scored.append((chunk, float(score)))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "BM25 results",
            extra={"context": {
                "result_count": len(scored[:top_k]),
                "top_score": scored[0][1] if scored else 0,
            }},
        )

        return scored[:top_k]

    def normalize_scores(
        self, scored: list[tuple[EvidenceChunk, float]]
    ) -> list[tuple[EvidenceChunk, float]]:
        """
        Normalize BM25 scores to 0–1 range using min-max normalization.

        Needed to combine BM25 scores with vector similarity scores
        on the same scale.
        """
        if not scored:
            return scored

        scores = [s for _, s in scored]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return [(chunk, 1.0) for chunk, _ in scored]

        return [
            (chunk, (score - min_score) / score_range)
            for chunk, score in scored
        ]

    def is_built(self) -> bool:
        """Return True if the index has been built."""
        return self._bm25 is not None

    def document_count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._chunks)
