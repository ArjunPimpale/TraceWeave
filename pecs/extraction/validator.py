"""
Pydantic validation for Stage 1 extraction output.

Validates raw JSON strings from Phi-4-mini against the ExtractionResult schema.
Handles:
- JSON parsing failures (including extraction from markdown code fences)
- Schema validation with specific error messages for retry prompts
- Both single-object and array responses
- Semantic validation (word overlap check for hallucination detection)
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.extraction_result import ExtractionResult

logger = get_logger(__name__)

# Regex to extract JSON from markdown code fences: ```json ... ``` or ``` ... ```
_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```",
    re.DOTALL,
)

# Also try to find bare JSON arrays/objects
_BARE_JSON_PATTERN = re.compile(
    r"(\[[\s\S]*?\]|\{[\s\S]*?\})",
    re.DOTALL,
)


class ValidationResult:
    """Result of a validation attempt."""

    def __init__(
        self,
        results: list[ExtractionResult] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.results = results or []
        self.errors = errors or []
        self.success = len(errors or []) == 0 and results is not None


class ExtractionValidator:
    """
    Validates Stage 1 LLM output against the ExtractionResult schema.

    Pipeline:
    1. Parse JSON (with code-fence extraction fallback)
    2. Validate each object against ExtractionResult Pydantic model
    3. Perform semantic validation (hallucination check)
    """

    def validate(
        self,
        raw_response: str,
        source_document: str,
        chunk_id: str,
        source_text: str = "",
    ) -> ValidationResult:
        """
        Validate a raw LLM response string.

        Args:
            raw_response: Raw string from Phi-4-mini.
            source_document: Source document name (injected into each result).
            chunk_id: Chunk ID (injected into each result).
            source_text: Original chunk text (for hallucination detection).

        Returns:
            ValidationResult with parsed ExtractionResult objects or error messages.
        """
        # ── Step 1: Parse JSON ─────────────────────────────────────────────
        json_data, parse_error = self._parse_json(raw_response)
        if json_data is None:
            return ValidationResult(errors=[f"JSON parse failed: {parse_error}"])

        # ── Step 2: Normalize to a list ────────────────────────────────────
        if isinstance(json_data, dict):
            items = [json_data]
        elif isinstance(json_data, list):
            items = json_data
        else:
            return ValidationResult(
                errors=[f"Expected JSON array or object, got {type(json_data).__name__}"]
            )

        # ── Step 3: Validate each item ─────────────────────────────────────
        results: list[ExtractionResult] = []
        errors: list[str] = []

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"Item {i} is not a JSON object: {type(item).__name__}")
                continue

            # Inject provenance fields
            item.setdefault("source_document", source_document)
            item.setdefault("chunk_id", chunk_id)

            try:
                result = ExtractionResult.model_validate(item)

                # ── Step 3.5: Quality filter ───────────────────────────────
                if not self._quality_filter(result, source_document):
                    logger.warning(
                        "Extraction rejected by quality filter",
                        extra={"context": {
                            "chunk_id": chunk_id[:8],
                            "entity_id": result.entity_id,
                            "type": result.entity_type.value,
                        }},
                    )
                    errors.append(f"Item {i} rejected: Failed quality checks (too short or vague).")
                    continue

                # ── Step 4: Semantic validation ────────────────────────────
                if source_text:
                    overlap = self._word_overlap(result.text, source_text)
                    if overlap < settings.MIN_WORD_OVERLAP_RATIO:
                        logger.warning(
                            "Hallucination suspected",
                            extra={"context": {
                                "chunk_id": chunk_id[:8],
                                "entity_id": result.entity_id,
                                "word_overlap_ratio": round(overlap, 3),
                            }},
                        )
                        # Don't reject — flag in metadata
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata["hallucination_risk"] = True
                        result.metadata["word_overlap_ratio"] = round(overlap, 3)

                results.append(result)

            except ValidationError as exc:
                error_msgs = [f"Field '{e['loc'][0]}': {e['msg']}" for e in exc.errors()]
                errors.append(f"Item {i} validation failed: {'; '.join(error_msgs)}")

        if errors and not results:
            return ValidationResult(errors=errors)
        if errors:
            # Partial success — return what we have and log errors
            logger.warning(
                "Partial validation success",
                extra={"context": {
                    "valid_count": len(results),
                    "error_count": len(errors),
                    "errors": errors[:3],
                }},
            )

        return ValidationResult(results=results, errors=errors if not results else [])

    @staticmethod
    def _parse_json(text: str) -> tuple[Any, str]:
        """
        Parse JSON from LLM response text.

        Attempts:
        1. Direct json.loads()
        2. Extract from markdown code fences
        3. Find first bare JSON array/object
        """
        text = text.strip()

        # Attempt 1: Direct parse
        try:
            return json.loads(text), ""
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract from code fences
        fence_match = _JSON_FENCE_PATTERN.search(text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1)), ""
            except json.JSONDecodeError:
                pass

        # Attempt 3: Find bare JSON
        bare_match = _BARE_JSON_PATTERN.search(text)
        if bare_match:
            try:
                return json.loads(bare_match.group(1)), ""
            except json.JSONDecodeError:
                pass

        return None, f"Could not extract valid JSON from response (length={len(text)})"

    @staticmethod
    def _word_overlap(extracted_text: str, source_text: str) -> float:
        """
        Compute word overlap ratio between extracted text and source text.

        Returns the fraction of words in extracted_text that also appear in source_text.
        Used as a rough hallucination detector.
        """
        if not extracted_text or not source_text:
            return 1.0

        extracted_words = set(re.findall(r"\w+", extracted_text.lower()))
        source_words = set(re.findall(r"\w+", source_text.lower()))

        if not extracted_words:
            return 1.0

        overlap = extracted_words & source_words
        return len(overlap) / len(extracted_words)

    @staticmethod
    def _quality_filter(result: ExtractionResult, source_document: str) -> bool:
        """Return True if entity passes quality checks, False to discard."""
        if result.entity_type.value == "REQUIREMENT":
            # Reject very short requirements
            if len(result.text.split()) < 10:
                return False
            # Reject vague requirements
            vague_markers = ["should be", "could be", "maybe", "consider", "try to"]
            if any(m in result.text.lower() for m in vague_markers):
                return False
        return True

    @staticmethod
    def format_errors_for_retry(errors: list[str]) -> str:
        """Format validation errors for inclusion in a retry prompt."""
        return "\n".join(f"- {e}" for e in errors)
