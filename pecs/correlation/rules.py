"""
Deterministic rule engine — resolves unambiguous correlations without the LLM.

Rules are applied in order. Each resolved pair is removed from the candidate set
so later rules don't re-process it.

Rules:
1. exact_requirement_id_match
2. requirement_with_no_evidence
3. implementation_plus_positive_evaluation
4. implementation_plus_negative_evaluation
5. orphan_evaluation
6. claim_without_evidence (annotates but does NOT resolve — passes to LLM)
"""

from __future__ import annotations

import re
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus

logger = get_logger(__name__)

# ── Sentiment word lists ───────────────────────────────────────────────────────

POSITIVE_MARKERS = frozenset([
    "excellent", "well-implemented", "well implemented", "good", "satisfactory",
    "meets expectations", "meets the requirements", "correct", "correctly",
    "complete", "completed", "well-structured", "well structured", "thorough",
    "impressive", "solid", "strong", "effective", "efficient", "appropriate",
    "demonstrates", "clear understanding", "proficient", "successful", "success",
])

NEGATIVE_MARKERS = frozenset([
    "poor", "not implemented", "not complete", "missing", "incomplete", "absent",
    "unsatisfactory", "inadequate", "lacks", "lacking", "failed", "failure",
    "incorrect", "wrong", "error", "bug", "does not", "didn't", "not found",
    "not present", "rudimentary", "insufficient", "weak", "minimal",
])

CLAIM_MARKERS = frozenset([
    "we implemented", "we built", "we developed", "we created", "we designed",
    "was implemented", "was built", "was completed", "was developed", "was created",
    "implemented by", "built by", "is implemented", "has been implemented",
])

# Score threshold below which a numeric score is considered negative
NEGATIVE_SCORE_THRESHOLD = 5.0

# Numeric score pattern (e.g., "8.5/10", "7/10", "Score: 6")
_SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10|Score:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# Min vector score for a requirement to have "found" evidence via retrieval
MIN_RETRIEVAL_SCORE = 0.15


# ── Module-level convenience wrappers (importable by tests) ───────────────────

def _is_positive_eval(text: str) -> bool:
    """Return True if the text contains positive evaluation markers."""
    text_lower = text.lower()
    if any(marker in text_lower for marker in POSITIVE_MARKERS):
        return True
    for match in _SCORE_PATTERN.finditer(text):
        score_str = match.group(1) or match.group(2)
        if score_str and float(score_str) >= NEGATIVE_SCORE_THRESHOLD:
            return True
    return False


def _is_negative_eval(text: str) -> bool:
    """Return True if the text contains negative evaluation markers."""
    text_lower = text.lower()
    if any(marker in text_lower for marker in NEGATIVE_MARKERS):
        return True
    for match in _SCORE_PATTERN.finditer(text):
        score_str = match.group(1) or match.group(2)
        if score_str and float(score_str) < NEGATIVE_SCORE_THRESHOLD:
            return True
    return False


class RuleEngine:
    """
    Deterministic correlation rule engine.

    Takes requirements and evidence from SQLite and resolves as many
    correlations as possible without the LLM.
    """

    def apply_rules(
        self,
        requirements: list[dict[str, Any]],
        implementations: list[dict[str, Any]],
        evaluations: list[dict[str, Any]],
        retrieval_scores: dict[str, float],  # req_entity_id → best retrieval score
    ) -> tuple[list[CorrelationResult], list[dict[str, Any]]]:
        """
        Apply all rules to generate CorrelationResults.

        Args:
            requirements: All REQUIREMENT rows from evidence table.
            implementations: All IMPLEMENTATION rows from evidence table.
            evaluations: All EVALUATION rows from evidence table.
            retrieval_scores: Best retrieval score found for each requirement.

        Returns:
            (resolved_correlations, ambiguous_pairs)
            - resolved_correlations: CorrelationResult objects from rules.
            - ambiguous_pairs: Dicts describing req+evidence pairs for Stage 2.
        """
        resolved: list[CorrelationResult] = []
        resolved_req_ids: set[str] = set()  # Track resolved requirements
        ambiguous: list[dict[str, Any]] = []

        # Build lookup maps
        impl_by_req: dict[str, list[dict]] = {}
        eval_by_req: dict[str, list[dict]] = {}

        for impl in implementations:
            linked = impl.get("linked_requirement")
            if linked:
                impl_by_req.setdefault(linked, []).append(impl)

        for ev in evaluations:
            linked = ev.get("linked_requirement")
            if linked:
                eval_by_req.setdefault(linked, []).append(ev)

        # ── Rule 1: exact_requirement_id_match ────────────────────────────
        for req in requirements:
            req_id = req["entity_id"]
            linked_impls = impl_by_req.get(req_id, [])
            linked_evals = eval_by_req.get(req_id, [])

            if linked_impls:
                status = self._determine_status_with_evals(linked_evals)
                chunk_ids = [i["chunk_id"] for i in linked_impls] + [e["chunk_id"] for e in linked_evals]

                resolved.append(CorrelationResult(
                    requirement_entity_id=req_id,
                    evidence_entity_id=linked_impls[0]["entity_id"],
                    status=status,
                    resolution_method="deterministic_rule",
                    rule_name="exact_requirement_id_match",
                    confidence=0.95 if linked_evals else 0.85,
                    supporting_chunk_ids=chunk_ids,
                    reasoning=(
                        f"Rule 'exact_requirement_id_match': "
                        f"{len(linked_impls)} implementation(s) explicitly link to {req_id}. "
                        f"Status: {status.value}."
                    ),
                ))
                resolved_req_ids.add(req_id)
                logger.info(
                    "Deterministic rule fired",
                    extra={"context": {
                        "requirement_id": req_id,
                        "rule": "exact_requirement_id_match",
                        "status": status.value,
                    }},
                )

        # ── Rule 2: requirement_with_no_evidence ──────────────────────────
        for req in requirements:
            req_id = req["entity_id"]
            if req_id in resolved_req_ids:
                continue

            has_any_impl = bool(impl_by_req.get(req_id))
            best_retrieval = retrieval_scores.get(req_id, 0.0)

            if not has_any_impl:
                if best_retrieval < MIN_RETRIEVAL_SCORE:
                    resolved.append(CorrelationResult(
                        requirement_entity_id=req_id,
                        evidence_entity_id="(none)",
                        status=CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED,
                        resolution_method="deterministic_rule",
                        rule_name="requirement_with_no_evidence",
                        confidence=0.85,
                        supporting_chunk_ids=[],
                        reasoning=(
                            f"Rule 'requirement_with_no_evidence': "
                            f"No linked implementation and no retrieval candidate "
                            f"above threshold (best score: {best_retrieval:.2f})."
                        ),
                    ))
                    resolved_req_ids.add(req_id)
                    logger.info(
                        "Deterministic rule fired",
                        extra={"context": {
                            "requirement_id": req_id,
                            "rule": "requirement_with_no_evidence",
                            "status": "REQUIREMENT_NOT_IMPLEMENTED",
                        }},
                    )
                else:
                    # Score is >= 0.15 — let Stage 2 decide based on retrieved chunk
                    ambiguous.append({
                        "type": "retrieved_evidence_ambiguous",
                        "requirement": req,
                        "evidence": None,  # Stage 2 will retrieve and inspect
                        "hint": f"Retrieval found candidate evidence (score={best_retrieval:.2f}) but no explicit link.",
                    })
                    resolved_req_ids.add(req_id)  # Prevents Rule 6/unresolved from processing it again

        # ── Rule 3 & 4: implementation_plus_{positive|negative}_evaluation ──
        # Already handled in Rule 1 via _determine_status_with_evals.
        # These rules handle cases where evaluation was linked separately.

        # ── Rule 5: orphan_evaluation ─────────────────────────────────────
        for ev in evaluations:
            req_id = ev.get("linked_requirement")
            if req_id:
                continue  # Already has a requirement link

            # Check if any requirement can be matched by keyword overlap
            req_ids = [r["entity_id"] for r in requirements]
            matched = self._find_matching_req(ev["text"], requirements)

            if not matched:
                resolved.append(CorrelationResult(
                    requirement_entity_id="(unlinked)",
                    evidence_entity_id=ev["entity_id"],
                    status=CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT,
                    resolution_method="deterministic_rule",
                    rule_name="orphan_evaluation",
                    confidence=0.80,
                    supporting_chunk_ids=[ev["chunk_id"]],
                    reasoning=(
                        f"Rule 'orphan_evaluation': "
                        f"Evaluation evidence '{ev['entity_id']}' has no linked requirement "
                        f"and no keyword-matching requirement was found."
                    ),
                ))
                logger.info(
                    "Deterministic rule fired",
                    extra={"context": {
                        "evidence_id": ev["entity_id"],
                        "rule": "orphan_evaluation",
                        "status": "EVALUATION_WITHOUT_REQUIREMENT",
                    }},
                )
            else:
                # Mark as ambiguous — needs LLM to classify
                ambiguous.append({
                    "type": "claim_ambiguous",
                    "requirement": matched,
                    "evidence": ev,
                    "hint": "Evaluation with possible requirement match but not explicitly linked",
                })

        # ── Rule 6: claim_without_evidence (annotate → pass to LLM) ───────
        for impl in implementations:
            if self._is_claim_without_concrete_evidence(impl, requirements):
                req_id = impl.get("linked_requirement") or "(unknown)"
                ambiguous.append({
                    "type": "claim_without_evidence",
                    "requirement": next(
                        (r for r in requirements if r["entity_id"] == req_id), None
                    ),
                    "evidence": impl,
                    "hint": "claim_without_evidence: verbal claim but no concrete implementation",
                })

        # ── Remaining unresolved requirements → ambiguous ─────────────────
        for req in requirements:
            req_id = req["entity_id"]
            if req_id in resolved_req_ids:
                continue
            # Gather all possible evidence
            possible_impls = impl_by_req.get(req_id, [])
            possible_evals = eval_by_req.get(req_id, [])
            if possible_impls or possible_evals:
                ambiguous.append({
                    "type": "unresolved_with_candidates",
                    "requirement": req,
                    "evidence": possible_impls + possible_evals,
                    "hint": "Has candidate evidence but not resolved deterministically",
                })

        logger.info(
            "Rule engine complete",
            extra={"context": {
                "total_requirements": len(requirements),
                "deterministic_resolved": len(resolved_req_ids),
                "ambiguous_pairs": len(ambiguous),
            }},
        )

        return resolved, ambiguous

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _determine_status_with_evals(
        self, evaluations: list[dict[str, Any]]
    ) -> CorrelationStatus:
        """
        Determine correlation status based on linked evaluation evidence.

        Returns IMPLEMENTED_AND_VALIDATED, IMPLEMENTED_BUT_NEGATIVELY_EVALUATED,
        or IMPLEMENTED_WITHOUT_EVALUATION.
        """
        if not evaluations:
            return CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION

        positive = any(self._is_positive_eval(e["text"]) for e in evaluations)
        negative = any(self._is_negative_eval(e["text"]) for e in evaluations)

        if positive and not negative:
            return CorrelationStatus.IMPLEMENTED_AND_VALIDATED
        if negative:
            return CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED
        return CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION

    @staticmethod
    def _is_positive_eval(text: str) -> bool:
        text_lower = text.lower()
        if any(marker in text_lower for marker in POSITIVE_MARKERS):
            return True
        # Check numeric score
        for match in _SCORE_PATTERN.finditer(text):
            score_str = match.group(1) or match.group(2)
            if score_str and float(score_str) >= NEGATIVE_SCORE_THRESHOLD:
                return True
        return False

    @staticmethod
    def _is_negative_eval(text: str) -> bool:
        text_lower = text.lower()
        if any(marker in text_lower for marker in NEGATIVE_MARKERS):
            return True
        # Check numeric score below threshold
        for match in _SCORE_PATTERN.finditer(text):
            score_str = match.group(1) or match.group(2)
            if score_str and float(score_str) < NEGATIVE_SCORE_THRESHOLD:
                return True
        return False

    @staticmethod
    def _find_matching_req(
        text: str, requirements: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Find a requirement with keyword overlap > 30% with the given text."""
        text_words = set(re.findall(r"\w+", text.lower()))
        if not text_words:
            return None

        best_req = None
        best_overlap = 0.0

        for req in requirements:
            req_words = set(re.findall(r"\w+", req.get("text", "").lower()))
            if not req_words:
                continue
            overlap = len(text_words & req_words) / len(text_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_req = req

        return best_req if best_overlap >= 0.30 else None

    @staticmethod
    def _is_claim_without_concrete_evidence(
        impl: dict[str, Any], requirements: list[dict[str, Any]]
    ) -> bool:
        """
        Detect if an implementation evidence row is a verbal claim
        without concrete implementation (from a chat/email source).
        """
        text = impl.get("text", "").lower()
        source = impl.get("source_document", "").lower()
        source_type = impl.get("metadata", {})
        entity_type = impl.get("entity_type", "")

        # Only flag if it's a claim-marker sentence
        has_claim = any(marker in text for marker in CLAIM_MARKERS)
        # And it's from a conversational source
        is_conversational = any(
            s in source for s in [".txt", "whatsapp", "chat", "email", ".eml"]
        )

        return has_claim and is_conversational
