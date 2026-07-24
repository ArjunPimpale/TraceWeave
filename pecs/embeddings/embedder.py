"""
Embedding generation using nomic-embed-text via Ollama.

Generates 768-dimensional embedding vectors for EvidenceChunk text.
Uses batch processing to minimize HTTP round-trips.
Implements retry with exponential backoff on API failures.
"""

from __future__ import annotations

import time
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk

logger = get_logger(__name__)

_BATCH_SIZE = 32
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds


class Embedder:
    """
    Generates text embeddings using nomic-embed-text via Ollama.

    Supports:
    - Batch embedding of EvidenceChunk objects
    - Single-query embedding for retrieval
    - Startup health checks (Ollama running? model pulled?)
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.batch_size = batch_size
        self._client = self._init_client()

    def _init_client(self):
        """Initialize the Ollama client."""
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            return client
        except ImportError as exc:
            raise RuntimeError(
                "ollama package not installed. Run: uv pip install ollama"
            ) from exc

    def check_health(self) -> dict[str, Any]:
        """
        Check whether Ollama is running and the embedding model is available.

        Returns:
            Dict with 'ollama_running' (bool), 'model_available' (bool),
            'error' (str|None).
        """
        result: dict[str, Any] = {
            "ollama_running": False,
            "model_available": False,
            "error": None,
        }
        try:
            models = self._client.list()
            result["ollama_running"] = True
            model_names = [m.model for m in (models.models or [])]
            result["model_available"] = any(
                self.model in name for name in model_names
            )
            if not result["model_available"]:
                result["error"] = (
                    f"Model {self.model!r} not found. "
                    f"Run: ollama pull {self.model}"
                )
        except Exception as exc:
            result["error"] = (
                f"Ollama not running at {self.base_url}. "
                f"Start with: ollama serve\n({exc})"
            )
        return result

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Args:
            text: The query text to embed.

        Returns:
            768-dimensional float vector.
        """
        return self._embed_with_retry([text])[0]

    def embed_chunks(
        self, chunks: list[EvidenceChunk]
    ) -> list[tuple[str, list[float]]]:
        """
        Embed a list of EvidenceChunks.

        Processes in batches to minimize HTTP round-trips.

        Args:
            chunks: List of EvidenceChunk objects.

        Returns:
            List of (chunk_id, embedding_vector) pairs.
        """
        if not chunks:
            return []

        results: list[tuple[str, list[float]]] = []

        for batch_num, batch_start in enumerate(range(0, len(chunks), self.batch_size)):
            batch = chunks[batch_start: batch_start + self.batch_size]
            texts = [c.normalized_text for c in batch]

            logger.debug(
                "Embedding batch started",
                extra={"context": {
                    "batch_size": len(batch),
                    "batch_number": batch_num + 1,
                }},
            )

            t0 = time.monotonic()
            vectors = self._embed_with_retry(texts)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            logger.info(
                "Embedding batch completed",
                extra={"context": {
                    "batch_size": len(batch),
                    "batch_number": batch_num + 1,
                    "latency_ms": elapsed_ms,
                }},
            )

            for chunk, vector in zip(batch, vectors):
                results.append((chunk.chunk_id, vector))

        return results

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """
        Call Ollama embedding API with exponential backoff retry.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: After all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                vectors: list[list[float]] = []
                for text in texts:
                    # Truncate very long text to avoid context window overflow
                    truncated = text[:8000] if len(text) > 8000 else text
                    response = self._client.embeddings(
                        model=self.model,
                        prompt=truncated,
                    )
                    vectors.append(response.embedding)
                return vectors

            except Exception as exc:
                last_error = exc
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding API retry",
                    extra={"context": {
                        "attempt": attempt,
                        "max_retries": _MAX_RETRIES,
                        "wait_s": wait,
                        "error": str(exc),
                    }},
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"Embedding API failed after {_MAX_RETRIES} attempts: {last_error}"
        )
