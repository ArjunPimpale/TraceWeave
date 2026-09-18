"""
ChromaDB vector store operations.

Single collection 'pecs_evidence' with cosine distance.
Stores embeddings + metadata for all EvidenceChunks.
Provides upsert (idempotent by chunk_id) and query operations.
"""

from __future__ import annotations

import time
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk

logger = get_logger(__name__)


class ChromaStore:
    """
    ChromaDB vector store wrapper.

    Operations:
    - upsert_chunks: Store EvidenceChunks with their embeddings.
    - query: Retrieve top-K similar chunks for a query vector.
    - delete_by_source: Remove all chunks from a given source document.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._persist_dir = persist_directory or str(settings.chromadb_path)
        self._collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy-initialize ChromaDB persistent client."""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            logger.debug(
                "ChromaDB client initialized",
                extra={"context": {"path": self._persist_dir}},
            )
        return self._client

    def _get_collection(self):
        """Lazy-initialize or get the pecs_evidence collection."""
        if self._collection is None:
            client = self._get_client()
            import chromadb
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection ready",
                extra={"context": {"collection_name": self._collection_name}},
            )
        return self._collection

    def upsert_chunks(
        self,
        chunks: list[EvidenceChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert EvidenceChunks with their embeddings into ChromaDB.

        Upsert semantics: if a chunk_id already exists, it is updated.
        This enables idempotent re-ingestion.

        Args:
            chunks: List of EvidenceChunk objects.
            embeddings: Corresponding embedding vectors (same order as chunks).
        """
        if not chunks:
            return

        assert len(chunks) == len(embeddings), (
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

        collection = self._get_collection()

        ids = [c.chunk_id for c in chunks]
        documents = [c.normalized_text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        t0 = time.monotonic()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "ChromaDB upsert completed",
            extra={"context": {
                "chunk_count": len(chunks),
                "collection_name": self._collection_name,
                "latency_ms": elapsed_ms,
            }},
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query ChromaDB for the top-K most similar chunks.

        Args:
            query_embedding: 768-dim embedding vector for the query.
            n_results: Number of results to return (default: settings.RETRIEVAL_TOP_K).
            where: Optional ChromaDB metadata filter dict.
            include: Fields to include in results (default: documents, metadatas, distances).

        Returns:
            Raw ChromaDB query result dict with keys: ids, distances, documents, metadatas.
        """
        collection = self._get_collection()
        n_results = n_results or settings.RETRIEVAL_TOP_K
        include = include or ["documents", "metadatas", "distances"]

        t0 = time.monotonic()
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.count()),  # Can't request more than we have
            "include": include,
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        result_count = len(results.get("ids", [[]])[0])
        logger.debug(
            "ChromaDB query executed",
            extra={"context": {
                "n_results": result_count,
                "latency_ms": elapsed_ms,
                "where": str(where)[:100] if where else None,
            }},
        )

        return results

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def count_strict(self) -> int:
        """Return a count or raise, so outage is distinct from empty search."""
        return self._get_collection().count()

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, Any]:
        """Retrieve specific chunks by their IDs."""
        collection = self._get_collection()
        return collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )

    def get_all_records(self) -> dict[str, Any]:
        """Retrieve all stored chunk IDs, documents, and metadata records."""
        return self._get_collection().get(include=["documents", "metadatas"])

    def delete_by_source(self, source_document: str) -> None:
        """
        Delete all chunks from a specific source document.

        Used when re-ingesting a modified file.
        """
        collection = self._get_collection()
        collection.delete(
            where={"source_document": source_document}
        )
        logger.info(
            "ChromaDB chunks deleted by source",
            extra={"context": {"source_document": source_document}},
        )

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.

        WARNING: This destroys all stored vectors. Use only for debugging.
        """
        client = self._get_client()
        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None
        self._get_collection()
        logger.warning("ChromaDB collection reset", extra={"context": {"collection": self._collection_name}})
