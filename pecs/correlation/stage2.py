"""
Stage 2 LLM classifier for ambiguous requirement-evidence pairs.

Only processes pairs that the deterministic rule engine could not resolve.
Classifies each pair into exactly one of the 7 CorrelationStatus labels.
Output is constrained to the label set — no free-text generation.

Key improvement: Stage 2 now receives the top retrieved chunks from the
vector store, not just pre-extracted evidence rows. This gives the LLM
real evidence text to work with for requirements without explicit links.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus
from pecs.models.retrieved_evidence import RetrievedEvidence

logger = get_logger(__name__)

# Valid status labels
_VALID_LABELS = {s.value for s in CorrelationStatus}

_STAGE2_SYSTEM = """You are a requirement-evidence correlation classifier for a software project traceability system.

Given a requirement and one or more evidence chunks retrieved from the project's codebase and documentation, classify their relationship into EXACTLY ONE of these 7 labels:

- IMPLEMENTED_AND_VALIDATED: Requirement is implemented AND there is positive evaluation/feedback
- IMPLEMENTED_BUT_NEGATIVELY_EVALUATED: Requirement is implemented but received negative feedback or failed tests
- IMPLEMENTED_WITHOUT_EVALUATION: Requirement is clearly implemented but has no evaluation/feedback
- PARTIALLY_IMPLEMENTED: Requirement is only partially addressed by the evidence
- CLAIMED_BUT_NO_EVIDENCE: Someone claimed it was done but the evidence does not show concrete implementation
- EVALUATION_WITHOUT_REQUIREMENT: The evidence is an evaluation but does not relate to this specific requirement
- REQUIREMENT_NOT_IMPLEMENTED: No evidence this requirement was addressed at all

Rules:
- Base your decision ONLY on the provided evidence text.
- If evidence is from code/git, look for functions, classes, or logic that implements the requirement.
- If evidence is from emails/chats, verbal claims do NOT count as implementation.
- Consider partial implementation if only some aspects are covered.

Respond ONLY with a JSON object:
{"status": "<LABEL>", "reasoning": "<1-2 sentences citing specific evidence text>"}

Do NOT add any text outside the JSON object."""

_STAGE2_USER = """Requirement:
ID: {req_id}
Text: {req_text}

Retrieved Evidence ({candidate_count} chunk(s)):
{evidence_blocks}

Hint: {hint}

Classify this requirement against the retrieved evidence."""

_EVIDENCE_BLOCK_TEMPLATE = """--- Chunk {n} | Source: {source} | Score: {score:.2f} ---
{text}"""


class Stage2Classifier:
    """
    LLM-based classifier for ambiguous requirement-evidence pairs.

    Called only by the correlation orchestrator for pairs not resolved
    deterministically. Receives the top retrieved evidence chunks from
    the vector store and outputs a CorrelationStatus label + reasoning.
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
                            requirement, evidence, retrieval_candidates, hint, type.

        Returns:
            CorrelationResult with status and reasoning, or None on failure.
        """
        req = ambiguous_pair.get("requirement") or {}
        hint = ambiguous_pair.get("hint", "")
        pair_type = ambiguous_pair.get("type", "")

        req_id = req.get("entity_id", "(unknown)")
        req_text = req.get("text", "")

        if not req_text:
            return None

        # Build evidence blocks from retrieval candidates (primary path)
        retrieval_candidates: list[RetrievedEvidence] = ambiguous_pair.get("retrieval_candidates", [])
        
        # Also include the pre-extracted evidence row if present (orphan eval, verbal claim paths)
        pre_extracted = ambiguous_pair.get("evidence")

        evidence_blocks, all_chunk_ids = self._build_evidence_blocks(
            retrieval_candidates, pre_extracted
        )

        if not evidence_blocks:
            # No evidence at all — can only classify as NOT_IMPLEMENTED
            logger.warning(
                "Stage 2 received pair with no evidence",
                extra={"context": {"req_id": req_id, "type": pair_type}},
            )
            return CorrelationResult(
                requirement_entity_id=req_id,
                evidence_entity_id="(none)",
                status=CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED,
                resolution_method="llm_stage2",
                rule_name=None,
                confidence=0.0,
                supporting_chunk_ids=[],
                reasoning="No evidence provided to Stage 2 classifier.",
            )

        evidence_blocks_text = "\n\n".join(evidence_blocks)

        user_content = _STAGE2_USER.format(
            req_id=req_id,
            req_text=req_text[:600],
            candidate_count=len(retrieval_candidates) + (1 if pre_extracted else 0),
            evidence_blocks=evidence_blocks_text,
            hint=hint,
        )

        # Primary evidence entity ID: first candidate or pre-extracted
        if retrieval_candidates:
            primary_evidence_id = retrieval_candidates[0].chunk.chunk_id
        elif pre_extracted:
            primary_evidence_id = pre_extracted.get("entity_id", "")
        else:
            primary_evidence_id = ""

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

                result = self._parse_response(raw, req_id, primary_evidence_id, all_chunk_ids)

                if result:
                    logger.info(
                        "Stage 2 classification",
                        extra={"context": {
                            "req_id": req_id,
                            "status": result.status.value,
                            "latency_ms": elapsed_ms,
                            "evidence_chunks": len(retrieval_candidates),
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
    def _build_evidence_blocks(
        retrieval_candidates: list[RetrievedEvidence],
        pre_extracted: dict[str, Any] | None,
    ) -> tuple[list[str], list[str]]:
        """
        Build formatted evidence text blocks for the LLM prompt.

        Returns (blocks, all_chunk_ids).
        """
        blocks: list[str] = []
        chunk_ids: list[str] = []

        for i, candidate in enumerate(retrieval_candidates, start=1):
            chunk = candidate.chunk
            text = chunk.normalized_text[:800]
            block = _EVIDENCE_BLOCK_TEMPLATE.format(
                n=i,
                source=chunk.source_document,
                score=candidate.combined_score,
                text=text,
            )
            blocks.append(block)
            chunk_ids.append(chunk.chunk_id)

        # Append pre-extracted evidence (e.g., evaluation row) if present
        if pre_extracted:
            n = len(retrieval_candidates) + 1
            text = pre_extracted.get("text", "")[:800]
            source = pre_extracted.get("source_document", "unknown")
            block = _EVIDENCE_BLOCK_TEMPLATE.format(
                n=n, source=source, score=1.0, text=text
            )
            blocks.append(block)
            if pre_extracted.get("chunk_id"):
                chunk_ids.append(pre_extracted["chunk_id"])

        return blocks, chunk_ids

    @staticmethod
    def _parse_response(
        raw: str,
        req_entity_id: str,
        evidence_entity_id: str,
        chunk_ids: list[str],
    ) -> CorrelationResult | None:
        """Parse and validate Stage 2 LLM response."""
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
            confidence=0.0,  # Set by ConfidenceScorer
            supporting_chunk_ids=chunk_ids,
            reasoning=reasoning,
        )
