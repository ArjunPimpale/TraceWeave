"""
Stage 1 extractor — calls Phi-4-mini to extract structured entities from chunks.

Implements the retry strategy from Section 8.5:
- Attempt 1: Standard prompt
- Attempt 2: Standard prompt + validation errors
- Attempt 3: Simplified prompt
- Failure: Log as extraction_failed, skip chunk, continue

Processes chunks INDIVIDUALLY (not batched) per Section 8.6 rationale.
Temperature ≈ 0 for deterministic output.
"""

from __future__ import annotations

import time
from typing import Any

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


class Extractor:
    """
    Stage 1 LLM extractor using Phi-4-mini via Ollama.

    For each EvidenceChunk, calls the LLM with a constrained extraction prompt
    and validates the output with up to MAX_EXTRACTION_RETRIES attempts.
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
        self._client = self._init_client()

    def _init_client(self):
        try:
            import ollama
            return ollama.Client(host=self.base_url)
        except ImportError as exc:
            raise RuntimeError("ollama package not installed") from exc

    def extract_chunk(self, chunk: EvidenceChunk) -> list[Any]:
        """
        Extract structured entities from a single EvidenceChunk.

        Args:
            chunk: The EvidenceChunk to process.

        Returns:
            List of validated ExtractionResult objects.
            Empty list if extraction failed after all retries.
        """
        logger.info(
            "Extraction started",
            extra={"context": {
                "chunk_id": chunk.chunk_id[:8],
                "source_document": chunk.source_document,
            }},
        )

        last_errors: list[str] = []

        for attempt in range(1, self.max_retries + 1):
            prompt = self._build_prompt(chunk, attempt, last_errors)

            t0 = time.monotonic()
            raw_response = self._call_llm(prompt, attempt)
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

    def extract_batch(self, chunks: list[EvidenceChunk]) -> dict[str, list[Any]]:
        """
        Extract entities from multiple chunks sequentially.

        Returns:
            Dict mapping chunk_id → list of ExtractionResult objects.
        """
        results: dict[str, list[Any]] = {}
        for chunk in chunks:
            results[chunk.chunk_id] = self.extract_chunk(chunk)
        return results

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

    def _call_llm(self, messages: list[dict[str, str]], attempt: int) -> str:
        """Call the Ollama LLM API."""
        try:
            response = self._client.chat(
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
