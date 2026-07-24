"""
Retrieval pipeline orchestrator.

Combines:
1. Metadata filtering (narrows search space)
2. Vector Top-K retrieval (ChromaDB)
3. BM25 keyword retrieval
4. Requirement-ID matching
5. Candidate merging with combined scoring

This is the multi-strategy retrieval system from Section 10 of the plan.
"""

from __future__ import annotations

import re
import time
from typing import Any

from pecs.config import settings
from pecs.embeddings.embedder import Embedder
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.models.retrieved_evidence import RetrievedEvidence
from pecs.retrieval.bm25 import BM25Index
from pecs.retrieval.merger import CandidateMerger
from pecs.retrieval.metadata_filter import MetadataFilter
from pecs.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)

# Patterns for requirement identifier variants
# Matches: R3, Requirement 3, req-3, requirement_3, REQ3, etc.
_REQ_ID_VARIANTS = [
    r"\b{req}\b",                      # Exact: R3
    r"\b[Rr]equirement\s*{num}\b",     # Requirement 3
    r"\b[Rr]eq[-_]?{num}\b",           # req-3, req_3, REQ3
]


class RetrievalPipeline:
    """
    Multi-strategy retrieval pipeline.

    Retrieves relevant EvidenceChunks for a given query using:
    - Vector similarity (ChromaDB)
    - BM25 keyword matching
    - Requirement-ID exact matching

    Results are merged and deduplicated with a combined relevance score.
    """

    def __init__(
        self,
        chroma_store: ChromaStore | None = None,
        embedder: Embedder | None = None,
        bm25_index: BM25Index | None = None,
        merger: CandidateMerger | None = None,
    ) -> None:
        self._chroma = chroma_store or ChromaStore()
        self._embedder = embedder or Embedder()
        self._bm25 = bm25_index or BM25Index()
        self._merger = merger or CandidateMerger()

    def retrieve(
        self,
        query_text: str,
        metadata_filter: MetadataFilter | None = None,
        requirement_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedEvidence]:
        """
        Run the full retrieval pipeline for a query.

        Args:
            query_text: The query or requirement text to retrieve evidence for.
            metadata_filter: Optional pre-filtering by source type, document, etc.
            requirement_ids: Optional list of requirement IDs to match explicitly.
            top_k: Number of results to return.

        Returns:
            List of RetrievedEvidence sorted by combined_score descending.
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K

        logger.info(
            "Retrieval started",
            extra={"context": {
                "query_preview": query_text[:80],
                "filters": str(metadata_filter.build() if metadata_filter else None)[:100],
            }},
        )

        t0 = time.monotonic()
        where_filter = metadata_filter.build() if metadata_filter else None

        # ── Step 2: Vector retrieval ──────────────────────────────────────
        vector_results = self._vector_retrieve(query_text, where_filter, top_k)

        # ── Step 3: BM25 keyword retrieval ───────────────────────────────
        source_types = metadata_filter.source_types if metadata_filter else None
        bm25_results_raw = self._bm25.query(query_text, top_k, source_types)
        bm25_results_normalized = self._bm25.normalize_scores(bm25_results_raw)

        # ── Step 4: Requirement-ID matching ──────────────────────────────
        id_match_results: list[EvidenceChunk] = []
        if requirement_ids:
            id_match_results = self._id_match_retrieve(requirement_ids, where_filter)

        # ── Step 5: Merge ─────────────────────────────────────────────────
        merged = self._merger.merge(
            vector_results=vector_results,
            bm25_results=bm25_results_normalized,
            id_match_results=id_match_results,
            top_k=top_k,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Retrieval completed",
            extra={"context": {
                "total_results": len(merged),
                "retrieval_latency_ms": elapsed_ms,
            }},
        )

        return merged

    def _vector_retrieve(
        self,
        query_text: str,
        where_filter: dict | None,
        top_k: int,
    ) -> list[tuple[EvidenceChunk, float]]:
        """Embed the query and retrieve top-K from ChromaDB."""
        if self._chroma.count() == 0:
            return []

        try:
            query_embedding = self._embedder.embed_query(query_text)
            results = self._chroma.query(
                query_embedding=query_embedding,
                n_results=top_k,
                where=where_filter,
            )
        except Exception as exc:
            logger.error(
                "Vector retrieval failed",
                extra={"context": {"error": str(exc)}},
            )
            return []

        chunks_with_scores: list[tuple[EvidenceChunk, float]] = []
        ids_list = results.get("ids", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        documents_list = results.get("documents", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]

        for cid, distance, doc_text, meta in zip(
            ids_list, distances_list, documents_list, metadatas_list
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity: 1 - distance (for normalized cosine, range 0–1)
            similarity = max(0.0, 1.0 - distance)

            chunk = self._reconstruct_chunk(cid, doc_text, meta)
            chunks_with_scores.append((chunk, similarity))

        logger.debug(
            "Vector results",
            extra={"context": {
                "result_count": len(chunks_with_scores),
                "top_score": chunks_with_scores[0][1] if chunks_with_scores else 0,
                "bottom_score": chunks_with_scores[-1][1] if chunks_with_scores else 0,
            }},
        )

        return chunks_with_scores

    def _id_match_retrieve(
        self,
        requirement_ids: list[str],
        where_filter: dict | None,
    ) -> list[EvidenceChunk]:
        """
        Search for chunks that explicitly mention requirement identifiers.

        For each requirement ID (e.g., "R3"), searches for variants:
        "R3", "Requirement 3", "req-3", "req_3", etc.
        """
        matched_chunks: list[EvidenceChunk] = []
        seen_ids: set[str] = set()

        for req_id in requirement_ids:
            # Extract numeric part if present (e.g., "R3" → "3")
            num_match = re.search(r"(\d+)", req_id)
            num = num_match.group(1) if num_match else req_id

            patterns = [
                req_id,  # Exact ID
                f"Requirement {num}",
                f"requirement {num}",
                f"req-{num}",
                f"req_{num}",
                f"REQ{num}",
            ]

            # Search ChromaDB by querying with the ID as text
            for pattern in patterns:
                try:
                    query_emb = self._embedder.embed_query(pattern)
                    results = self._chroma.query(
                        query_embedding=query_emb,
                        n_results=5,
                        where=where_filter,
                    )
                    for cid, doc_text, meta in zip(
                        results.get("ids", [[]])[0],
                        results.get("documents", [[]])[0],
                        results.get("metadatas", [[]])[0],
                    ):
                        # Only keep if the ID literally appears in the text
                        if (
                            cid not in seen_ids
                            and req_id.lower() in doc_text.lower()
                        ):
                            seen_ids.add(cid)
                            matched_chunks.append(
                                self._reconstruct_chunk(cid, doc_text, meta)
                            )
                except Exception:
                    continue

        logger.debug(
            "ID match results",
            extra={"context": {
                "result_count": len(matched_chunks),
                "matched_ids": list(seen_ids)[:5],
            }},
        )

        return matched_chunks

    @staticmethod
    def _reconstruct_chunk(
        chunk_id: str, doc_text: str, meta: dict[str, Any]
    ) -> EvidenceChunk:
        """Reconstruct an EvidenceChunk from ChromaDB query results."""
        source_type_str = meta.get("source_type", "MARKDOWN")
        try:
            source_type = SourceType(source_type_str)
        except ValueError:
            source_type = SourceType.MARKDOWN

        return EvidenceChunk(
            chunk_id=chunk_id,
            source_document=meta.get("source_document", ""),
            source_hash=meta.get("source_hash", ""),
            source_type=source_type,
            chunk_index=meta.get("chunk_index", 0),
            source_locator=meta.get("source_locator", ""),
            normalized_text=doc_text,
            char_count=len(doc_text),
            metadata={k: v for k, v in meta.items()
                      if k not in ("source_document", "source_type", "source_hash",
                                   "chunk_index", "source_locator", "char_count", "created_at")},
        )

    def rebuild_bm25_index(self, chunks: list[EvidenceChunk]) -> None:
        """
        Rebuild the BM25 index with the given chunks.

        Call this after ingesting new documents.
        """
        self._bm25.build(chunks)
