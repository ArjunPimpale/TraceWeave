"""
Stage 2 LLM classifier for ambiguous requirement-evidence pairs.

Only processes pairs that the deterministic rule engine could not resolve.
Classifies each pair into exactly one of the 7 CorrelationStatus labels.
Output is constrained to the label set — no free-text generation.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus

logger = get_logger(__name__)

# Valid status labels
_VALID_LABELS = {s.value for s in CorrelationStatus}

_STAGE2_SYSTEM = """You are a requirement-evidence correlation classifier.

Given a requirement and evidence text, classify their relationship into EXACTLY ONE of these 7 labels:
- IMPLEMENTED_AND_VALIDATED: Requirement is implemented AND evaluated positively
- IMPLEMENTED_BUT_NEGATIVELY_EVALUATED: Requirement is implemented but received negative feedback
- IMPLEMENTED_WITHOUT_EVALUATION: Requirement is implemented but not evaluated
- PARTIALLY_IMPLEMENTED: Requirement is only partially addressed
- CLAIMED_BUT_NO_EVIDENCE: Someone claimed it was done but there is no concrete proof
- EVALUATION_WITHOUT_REQUIREMENT: Evaluation exists but doesn't map to this requirement
- REQUIREMENT_NOT_IMPLEMENTED: No evidence this requirement was addressed

Respond ONLY with a JSON object:
{"status": "<LABEL>", "reasoning": "<1-2 sentences citing specific evidence text>"}

Do NOT add any text outside the JSON object."""

_STAGE2_USER = """Requirement:
ID: {req_id}
Text: {req_text}

Evidence:
Type: {evidence_type}
Text: {evidence_text}

Hint: {hint}

Classify this requirement-evidence pair."""


class Stage2Classifier:
    """
    LLM-based classifier for ambiguous requirement-evidence pairs.

    Called only by the correlation orchestrator for pairs not resolved
    deterministically. Outputs a CorrelationStatus label + brief reasoning.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._client = self._init_client()

    def _init_client(self):
        try:
            import ollama
            return ollama.Client(host=self.base_url)
        except ImportError as exc:
            raise RuntimeError("ollama package not installed") from exc

    def classify(
        self, ambiguous_pair: dict[str, Any]
    ) -> CorrelationResult | None:
        """
        Classify a single ambiguous requirement-evidence pair.

        Args:
            ambiguous_pair: Dict from the rule engine with keys:
                            requirement, evidence, hint, type.

        Returns:
            CorrelationResult with status and reasoning, or None on failure.
        """
        req = ambiguous_pair.get("requirement") or {}
        evidence = ambiguous_pair.get("evidence") or {}
        hint = ambiguous_pair.get("hint", "")

        req_id = req.get("entity_id", "(unknown)")
        req_text = req.get("text", "")
        evidence_type = evidence.get("entity_type", "")
        evidence_text = evidence.get("text", "")

        if not req_text or not evidence_text:
            return None

        user_content = _STAGE2_USER.format(
            req_id=req_id,
            req_text=req_text[:500],
            evidence_type=evidence_type,
            evidence_text=evidence_text[:800],
            hint=hint,
        )

        messages = [
            {"role": "system", "content": _STAGE2_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        for attempt in range(1, 3):
            try:
                t0 = time.monotonic()
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature},
                )
                raw = response.message.content or ""
                elapsed_ms = int((time.monotonic() - t0) * 1000)

                result = self._parse_response(
                    raw, req_id, evidence.get("entity_id", ""), evidence.get("chunk_id", "")
                )

                if result:
                    logger.info(
                        "Stage 2 classification",
                        extra={"context": {
                            "req_id": req_id,
                            "status": result.status.value,
                            "latency_ms": elapsed_ms,
                        }},
                    )
                    return result

            except Exception as exc:
                logger.warning(
                    "Stage 2 LLM call failed",
                    extra={"context": {"error": str(exc), "attempt": attempt}},
                )

        return None

    def classify_batch(
        self, ambiguous_pairs: list[dict[str, Any]]
    ) -> list[CorrelationResult]:
        """Classify a batch of ambiguous pairs sequentially."""
        results: list[CorrelationResult] = []
        for pair in ambiguous_pairs:
            result = self.classify(pair)
            if result:
                results.append(result)
        return results

    @staticmethod
    def _parse_response(
        raw: str,
        req_entity_id: str,
        evidence_entity_id: str,
        chunk_id: str,
    ) -> CorrelationResult | None:
        """Parse and validate Stage 2 LLM response."""
        # Extract JSON
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*?\}", raw)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except Exception:
                    return None
            else:
                return None

        status_str = data.get("status", "").strip()
        if status_str not in _VALID_LABELS:
            # Try to find a matching label (case-insensitive)
            for label in _VALID_LABELS:
                if label.lower() in status_str.lower():
                    status_str = label
                    break
            else:
                return None

        status = CorrelationStatus(status_str)
        reasoning = data.get("reasoning", "")

        return CorrelationResult(
            requirement_entity_id=req_entity_id,
            evidence_entity_id=evidence_entity_id,
            status=status,
            resolution_method="llm_stage2",
            rule_name=None,
            confidence=0.0,  # Will be set by ConfidenceScorer
            supporting_chunk_ids=[chunk_id] if chunk_id else [],
            reasoning=reasoning,
        )
