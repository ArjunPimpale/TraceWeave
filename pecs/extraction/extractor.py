"""
Stage 1 extractor — calls Phi-4-mini to extract structured entities from chunks.

Implements the retry strategy from Section 8.5:
- Attempt 1: Standard prompt
- Attempt 2: Standard prompt + validation errors
- Attempt 3: Simplified prompt
- Failure: Log as extraction_failed, skip chunk, continue

Processes chunks INDIVIDUALLY (not batched) per Section 8.6 rationale.
Temperature ≈ 0 for deterministic output.

Concurrency:
    extract_batch_concurrent() uses ThreadPoolExecutor for I/O-bound parallelism.
    Each worker thread gets its own Ollama client via threading.local() to avoid
    sharing HTTP connections across threads.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any, Callable

from pecs.config import settings
from pecs.extraction.prompts import (
    STAGE1_SYSTEM_PROMPT,
    STAGE1_FEW_SHOT_EXAMPLES,
    STAGE1_USER_TEMPLATE,
    STAGE1_RETRY_TEMPLATE,
    STAGE1_SIMPLIFIED_TEMPLATE,
)
from pecs.extraction.validator import ExtractionValidator
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk
from pecs.models.extraction_result import ExtractionResult  # noqa: F401 — re-exported

logger = get_logger(__name__)

# Thread-local storage: each worker thread gets its own Ollama client instance.
# This prevents sharing HTTP connection state (sockets, headers) across threads.
_thread_local = threading.local()


class Extractor:
    """
    Stage 1 LLM extractor using Phi-4-mini via Ollama.

    For each EvidenceChunk, calls the LLM with a constrained extraction prompt
    and validates the output with up to MAX_EXTRACTION_RETRIES attempts.

    Concurrency:
        Use extract_batch_concurrent() to process multiple chunks in parallel
        using ThreadPoolExecutor. The bottleneck is I/O (HTTP calls to Ollama),
        not CPU, so threads — not processes — are the right primitive.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_retries = max_retries or settings.MAX_EXTRACTION_RETRIES
        self._validator = ExtractionValidator()
        # Main-thread client — used by extract_chunk() when called directly.
        self._client = self._make_client()

    def _make_client(self):
        """Create a new Ollama client pointing at self.base_url."""
        try:
            import ollama
            return ollama.Client(host=self.base_url)
        except ImportError as exc:
            raise RuntimeError("ollama package not installed") from exc

    def _get_thread_client(self):
        """
        Return a per-thread Ollama client, creating one if needed.

        Using thread-local storage ensures each worker thread owns an
        independent client with its own HTTP connection pool, preventing
        race conditions on shared socket state.
        """
        if not hasattr(_thread_local, "client"):
            _thread_local.client = self._make_client()
        return _thread_local.client

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_chunk(self, chunk: EvidenceChunk) -> list[Any]:
        """
        Extract structured entities from a single EvidenceChunk.

        Uses the main-thread Ollama client. Safe to call directly from tests
        or from the sequential extract_batch() fallback.

        Args:
            chunk: The EvidenceChunk to process.

        Returns:
            List of validated ExtractionResult objects.
            Empty list if extraction failed after all retries.
        """
        return self._extract_with_client(chunk, self._client)

    def extract_batch(self, chunks: list[EvidenceChunk]) -> dict[str, list[Any]]:
        """
        Extract entities from multiple chunks sequentially (original behaviour).

        Kept for backwards compatibility and as a fallback when concurrency is
        not desired (e.g. during unit tests).

        Returns:
            Dict mapping chunk_id → list of ExtractionResult objects.
        """
        results: dict[str, list[Any]] = {}
        for chunk in chunks:
            results[chunk.chunk_id] = self.extract_chunk(chunk)
        return results

    def extract_batch_concurrent(
        self,
        chunks: list[EvidenceChunk],
        max_workers: int | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, list[Any]]:
        """
        Extract entities from multiple chunks concurrently using a thread pool.

        Each worker thread obtains its own Ollama client via thread-local storage
        so that HTTP connections are never shared across threads.

        Results are collected in completion order (as_completed), not submission
        order, maximising throughput. The returned dict preserves chunk identity
        via chunk_id keys regardless of completion order.

        Args:
            chunks: List of EvidenceChunks to process.
            max_workers: Number of concurrent worker threads.
                         Defaults to settings.EXTRACTION_WORKERS (= 6).
            on_progress: Optional callback invoked on the calling thread after
                         each chunk completes. Signature:
                             on_progress(completed: int, total: int, source_doc: str)
                         Use this to drive a UI progress bar.

        Returns:
            Dict mapping chunk_id → list of ExtractionResult objects.
        """
        if not chunks:
            return {}

        max_workers = max_workers or settings.EXTRACTION_WORKERS
        total = len(chunks)
        results: dict[str, list[Any]] = {}
        completed_count = 0

        logger.info(
            "Concurrent extraction started",
            extra={"context": {
                "chunk_count": total,
                "max_workers": max_workers,
            }},
        )

        t_start = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit all chunks; store future → chunk mapping for error attribution.
            future_to_chunk: dict[concurrent.futures.Future, EvidenceChunk] = {
                pool.submit(self._extract_chunk_thread_safe, chunk): chunk
                for chunk in chunks
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                except Exception as exc:
                    logger.error(
                        "Worker thread extraction failed",
                        extra={"context": {
                            "chunk_id": chunk.chunk_id[:8],
                            "source_document": chunk.source_document,
                            "error": str(exc),
                        }},
                    )
                    chunk_results = []

                results[chunk.chunk_id] = chunk_results
                completed_count += 1

                if on_progress is not None:
                    try:
                        on_progress(completed_count, total, chunk.source_document)
                    except Exception:
                        pass  # Never let a progress callback crash the extraction

        elapsed_s = time.monotonic() - t_start
        logger.info(
            "Concurrent extraction complete",
            extra={"context": {
                "chunk_count": total,
                "elapsed_s": round(elapsed_s, 2),
                "avg_s_per_chunk": round(elapsed_s / total, 2) if total else 0,
            }},
        )

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_chunk_thread_safe(self, chunk: EvidenceChunk) -> list[Any]:
        """
        Thread-safe extraction using a per-thread Ollama client.

        Called by worker threads inside extract_batch_concurrent().
        Retrieves (or lazily creates) a thread-local client so that no two
        threads share the same HTTP connection pool.
        """
        client = self._get_thread_client()
        return self._extract_with_client(chunk, client)

    def _extract_with_client(self, chunk: EvidenceChunk, client) -> list[Any]:
        """
        Core extraction logic — runs the retry loop using the supplied client.

        Separated from extract_chunk() so that both the main-thread path and
        the per-thread path share identical logic without code duplication.

        Args:
            chunk: The EvidenceChunk to process.
            client: The ollama.Client instance to use for this call.

        Returns:
            List of validated ExtractionResult objects (may be empty on failure).
        """
        logger.info(
            "Extraction started",
            extra={"context": {
                "chunk_id": chunk.chunk_id[:8],
                "source_document": chunk.source_document,
                "thread": threading.current_thread().name,
            }},
        )

        last_errors: list[str] = []

        for attempt in range(1, self.max_retries + 1):
            prompt = self._build_prompt(chunk, attempt, last_errors)

            t0 = time.monotonic()
            raw_response = self._call_llm(prompt, attempt, client)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            logger.debug(
                "LLM response received",
                extra={"context": {
                    "chunk_id": chunk.chunk_id[:8],
                    "response_length": len(raw_response),
                    "latency_ms": elapsed_ms,
                    "attempt": attempt,
                }},
            )

            validation = self._validator.validate(
                raw_response=raw_response,
                source_document=chunk.source_document,
                chunk_id=chunk.chunk_id,
                source_text=chunk.normalized_text,
            )

            if validation.success:
                logger.info(
                    "Extraction validation passed",
                    extra={"context": {
                        "chunk_id": chunk.chunk_id[:8],
                        "entity_count": len(validation.results),
                        "attempt": attempt,
                    }},
                )
                return validation.results

            last_errors = validation.errors
            logger.warning(
                "Extraction validation failed",
                extra={"context": {
                    "chunk_id": chunk.chunk_id[:8],
                    "errors": last_errors[:3],
                    "attempt": attempt,
                    "max_retries": self.max_retries,
                }},
            )

        # All retries exhausted
        logger.error(
            "Extraction failed — max retries exhausted",
            extra={"context": {
                "chunk_id": chunk.chunk_id[:8],
                "source_document": chunk.source_document,
                "all_errors": last_errors,
            }},
        )
        return []

    def _build_prompt(
        self,
        chunk: EvidenceChunk,
        attempt: int,
        last_errors: list[str],
    ) -> list[dict[str, str]]:
        """Build the message list for the LLM call."""
        if attempt == 1:
            # Standard prompt with system + examples + user
            system = STAGE1_SYSTEM_PROMPT + "\n\n" + STAGE1_FEW_SHOT_EXAMPLES
            user = STAGE1_USER_TEMPLATE.format(
                source_document=chunk.source_document,
                source_locator=chunk.source_locator,
                chunk_text=chunk.normalized_text[:3000],  # Truncate very long chunks
            )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        elif attempt == 2:
            # Retry with error feedback
            system = STAGE1_SYSTEM_PROMPT
            error_text = self._validator.format_errors_for_retry(last_errors)
            user = STAGE1_RETRY_TEMPLATE.format(
                errors=error_text,
                source_document=chunk.source_document,
                source_locator=chunk.source_locator,
                chunk_text=chunk.normalized_text[:3000],
            )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        else:
            # Simplified prompt for final attempt
            user = STAGE1_SIMPLIFIED_TEMPLATE.format(
                chunk_text=chunk.normalized_text[:2000],
            )
            return [{"role": "user", "content": user}]

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        attempt: int,
        client=None,
    ) -> str:
        """
        Call the Ollama LLM API.

        Args:
            messages: Chat message list.
            attempt: Current attempt number (for logging).
            client: Ollama client to use. Defaults to self._client (main thread).
        """
        active_client = client if client is not None else self._client
        try:
            response = active_client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature},
            )
            return response.message.content or ""
        except Exception as exc:
            logger.error(
                "LLM call failed",
                extra={"context": {
                    "model": self.model,
                    "attempt": attempt,
                    "error": str(exc),
                }},
            )
            return ""
