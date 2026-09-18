"""
Deterministic rule engine — evidence-first correlation without the LLM.

Rules are applied in order. Each resolved pair is removed from the candidate
set so later rules don't re-process it.

Rules (revised):
R1. exact_requirement_id_match
    – An extracted IMPLEMENTATION row has linked_requirement = req_id.
    – Deterministically resolved; sub-checks evaluation sentiment for status.

R2. strong_retrieval_with_id_mention
    – Top retrieved chunk scores ≥ HIGH_SCORE_THRESHOLD (default 0.70) AND
      the requirement ID literally appears in the chunk text.
    – Deterministically resolved as IMPLEMENTED_WITHOUT_EVALUATION.

R3. requirement_with_no_evidence
    – No explicit link AND top retrieved score < NO_EVIDENCE_FLOOR (default 0.15).
    – Deterministically resolved as REQUIREMENT_NOT_IMPLEMENTED.

R4. ambiguous_with_retrieval_evidence  (→ Stage 2 LLM)
    – Has retrieval candidates but below the strong-resolve threshold.
    – Sends requirement + top retrieved chunks to the LLM.

R5. orphan_evaluation
    – EVALUATION row with no linked_requirement and no keyword-matching req.
    – Deterministically resolved as EVALUATION_WITHOUT_REQUIREMENT.

R6. verbal_claim_without_evidence  (→ Stage 2 LLM)
    – IMPLEMENTATION from conversational source containing claim markers.
    – Sent to LLM to validate whether real evidence exists.
"""

from __future__ import annotations

import re
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus
from pecs.models.retrieved_evidence import RetrievedEvidence

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

# Numeric score pattern (e.g., "8.5/10", "7/10", "Score: 6")
_SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10|Score:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# Threshold below which a numeric score is considered negative
NEGATIVE_SCORE_THRESHOLD = 5.0

# Retrieval thresholds
# R2: deterministic resolve if top score >= this (semantic match alone is enough;
# we do NOT require the literal req ID in the text because IDs like
# 'R1-gnn-exam-proctoring-system-design' are never literally in chunk text)
HIGH_SCORE_THRESHOLD = 0.75   # High confidence semantic match → resolve directly
NO_EVIDENCE_FLOOR = 0.30      # Below this → NOT_IMPLEMENTED; between floor and HIGH → LLM


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
    Evidence-first deterministic correlation rule engine.

    Takes requirements and evidence from SQLite plus retrieval candidates
    from the vector store, and resolves as many correlations as possible
    without the LLM.
    """

    def apply_rules(
        self,
        requirements: list[dict[str, Any]],
        implementations: list[dict[str, Any]],
        evaluations: list[dict[str, Any]],
        retrieval_candidates: dict[str, list[RetrievedEvidence]],
    ) -> tuple[list[CorrelationResult], list[dict[str, Any]]]:
        """
        Apply all rules to generate CorrelationResults.

        Args:
            requirements: All REQUIREMENT rows from evidence table.
            implementations: All IMPLEMENTATION rows from evidence table.
            evaluations: All EVALUATION rows from evidence table.
            retrieval_candidates: Maps req_entity_id → top-K RetrievedEvidence
                                  from the vector store (sorted by score desc).

        Returns:
            (resolved_correlations, ambiguous_pairs)
            - resolved_correlations: CorrelationResult objects from deterministic rules.
            - ambiguous_pairs: Dicts with req + top candidates for Stage 2 LLM.
        """
        resolved: list[CorrelationResult] = []
        resolved_req_ids: set[str] = set()
        ambiguous: list[dict[str, Any]] = []
        impl_by_req = self._index_by_linked_requirement(implementations)
        eval_by_req = self._index_by_linked_requirement(evaluations)

        # Keep the six passes in their original order: each pass relies on the
        # requirement IDs resolved by the preceding passes.
        self._apply_exact_requirement_matches(
            requirements, impl_by_req, eval_by_req, resolved, resolved_req_ids
        )
        self._apply_strong_semantic_matches(
            requirements, eval_by_req, retrieval_candidates, resolved, resolved_req_ids
        )
        self._apply_no_evidence_matches(
            requirements, impl_by_req, retrieval_candidates, resolved, resolved_req_ids
        )
        self._collect_ambiguous_retrieval_matches(
            requirements, eval_by_req, retrieval_candidates, resolved, resolved_req_ids, ambiguous
        )
        self._apply_orphan_evaluations(requirements, evaluations, resolved, ambiguous)
        self._collect_verbal_claims(requirements, implementations, retrieval_candidates, ambiguous)

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

    @staticmethod
    def _index_by_linked_requirement(
        evidence: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group evidence by its explicitly extracted requirement link."""
        indexed: dict[str, list[dict[str, Any]]] = {}
        for item in evidence:
            linked_requirement = item.get("linked_requirement")
            if linked_requirement:
                indexed.setdefault(linked_requirement, []).append(item)
        return indexed

    def _apply_exact_requirement_matches(
        self,
        requirements: list[dict[str, Any]],
        implementations: dict[str, list[dict[str, Any]]],
        evaluations: dict[str, list[dict[str, Any]]],
        resolved: list[CorrelationResult],
        resolved_ids: set[str],
    ) -> None:
        """Apply rule 1 for explicit implementation-to-requirement links."""
        for requirement in requirements:
            requirement_id = requirement["entity_id"]
            linked_implementations = implementations.get(requirement_id, [])
            linked_evaluations = evaluations.get(requirement_id, [])
            if not linked_implementations:
                continue
            status = self._determine_status_with_evals(linked_evaluations)
            chunk_ids = [item["chunk_id"] for item in linked_implementations]
            chunk_ids += [item["chunk_id"] for item in linked_evaluations]
            resolved.append(
                CorrelationResult(
                    requirement_entity_id=requirement_id,
                    evidence_entity_id=linked_implementations[0]["entity_id"],
                    status=status,
                    resolution_method="deterministic_rule",
                    rule_name="exact_requirement_id_match",
                    confidence=0.95 if linked_evaluations else 0.85,
                    supporting_chunk_ids=chunk_ids,
                    reasoning=(
                        f"Rule 'exact_requirement_id_match': "
                        f"{len(linked_implementations)} implementation(s) explicitly link to {requirement_id}. "
                        f"Status determined by evaluation sentiment: {status.value}."
                    ),
                )
            )
            resolved_ids.add(requirement_id)
            logger.info(
                "Deterministic rule fired",
                extra={"context": {
                    "requirement_id": requirement_id,
                    "rule": "exact_requirement_id_match",
                    "status": status.value,
                }},
            )

    def _apply_strong_semantic_matches(
        self,
        requirements: list[dict[str, Any]],
        evaluations: dict[str, list[dict[str, Any]]],
        retrieval_candidates: dict[str, list[RetrievedEvidence]],
        resolved: list[CorrelationResult],
        resolved_ids: set[str],
    ) -> None:
        """Apply rule 2 to unresolved requirements with strong retrieval."""
        for requirement in requirements:
            requirement_id = requirement["entity_id"]
            if requirement_id in resolved_ids:
                continue
            candidates = retrieval_candidates.get(requirement_id, [])
            if not candidates:
                continue
            best = candidates[0]
            if best.combined_score < HIGH_SCORE_THRESHOLD:
                continue
            linked_evaluations = evaluations.get(requirement_id, [])
            status = self._determine_status_with_evals(linked_evaluations)
            high_quality = [
                candidate
                for candidate in candidates
                if candidate.combined_score >= HIGH_SCORE_THRESHOLD
            ]
            chunk_ids = [candidate.chunk.chunk_id for candidate in high_quality]
            chunk_ids += [item["chunk_id"] for item in linked_evaluations]
            resolved.append(
                CorrelationResult(
                    requirement_entity_id=requirement_id,
                    evidence_entity_id=best.chunk.chunk_id,
                    status=status,
                    resolution_method="deterministic_rule",
                    rule_name="strong_semantic_match",
                    confidence=round(min(best.combined_score, 1.0), 4),
                    supporting_chunk_ids=chunk_ids,
                    reasoning=(
                        f"Rule 'strong_semantic_match': "
                        f"Top retrieved chunk from '{best.chunk.source_document}' "
                        f"has similarity score {best.combined_score:.2f} (>= {HIGH_SCORE_THRESHOLD}). "
                        f"{len(high_quality)} high-quality chunk(s). Status: {status.value}."
                    ),
                )
            )
            resolved_ids.add(requirement_id)
            logger.info(
                "Deterministic rule fired",
                extra={"context": {
                    "requirement_id": requirement_id,
                    "rule": "strong_semantic_match",
                    "score": round(best.combined_score, 3),
                    "status": status.value,
                }},
            )

    @staticmethod
    def _append_no_evidence_result(
        requirement_id: str,
        confidence: float,
        reasoning: str,
        resolved: list[CorrelationResult],
        resolved_ids: set[str],
    ) -> None:
        """Append the existing deterministic no-evidence correlation result."""
        resolved.append(
            CorrelationResult(
                requirement_entity_id=requirement_id,
                evidence_entity_id="(none)",
                status=CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED,
                resolution_method="deterministic_rule",
                rule_name="requirement_with_no_evidence",
                confidence=confidence,
                supporting_chunk_ids=[],
                reasoning=reasoning,
            )
        )
        resolved_ids.add(requirement_id)

    def _apply_no_evidence_matches(
        self,
        requirements: list[dict[str, Any]],
        implementations: dict[str, list[dict[str, Any]]],
        retrieval_candidates: dict[str, list[RetrievedEvidence]],
        resolved: list[CorrelationResult],
        resolved_ids: set[str],
    ) -> None:
        """Apply rule 3 to unresolved requirements below the retrieval floor."""
        for requirement in requirements:
            requirement_id = requirement["entity_id"]
            if requirement_id in resolved_ids:
                continue
            candidates = retrieval_candidates.get(requirement_id, [])
            best_score = candidates[0].combined_score if candidates else 0.0
            if implementations.get(requirement_id) or best_score >= NO_EVIDENCE_FLOOR:
                continue
            self._append_no_evidence_result(
                requirement_id,
                0.80,
                (
                    f"Rule 'requirement_with_no_evidence': No linked implementation "
                    f"and no retrieval candidate above floor (best score: {best_score:.2f} "
                    f"< {NO_EVIDENCE_FLOOR})."
                ),
                resolved,
                resolved_ids,
            )
            logger.info(
                "Deterministic rule fired",
                extra={"context": {
                    "requirement_id": requirement_id,
                    "rule": "requirement_with_no_evidence",
                    "best_score": round(best_score, 3),
                }},
            )

    def _collect_ambiguous_retrieval_matches(
        self,
        requirements: list[dict[str, Any]],
        evaluations: dict[str, list[dict[str, Any]]],
        retrieval_candidates: dict[str, list[RetrievedEvidence]],
        resolved: list[CorrelationResult],
        resolved_ids: set[str],
        ambiguous: list[dict[str, Any]],
    ) -> None:
        """Apply rule 4, retaining its fallback for unresolved requirements."""
        for requirement in requirements:
            requirement_id = requirement["entity_id"]
            if requirement_id in resolved_ids:
                continue
            candidates = retrieval_candidates.get(requirement_id, [])
            useful_candidates = [
                candidate
                for candidate in candidates
                if candidate.combined_score >= NO_EVIDENCE_FLOOR
            ][:5]
            if useful_candidates:
                ambiguous.append({
                    "type": "ambiguous_with_retrieval_evidence",
                    "requirement": requirement,
                    "evidence": None,
                    "retrieval_candidates": useful_candidates,
                    "linked_evaluations": evaluations.get(requirement_id, []),
                    "hint": (
                        f"Top retrieval score: {useful_candidates[0].combined_score:.2f}. "
                        f"Classify whether these chunks implement the requirement. "
                        f"Consider partial implementation, verbal claims, or missing evidence."
                    ),
                })
                logger.info(
                    "Deferred to Stage 2",
                    extra={"context": {
                        "requirement_id": requirement_id,
                        "rule": "ambiguous_with_retrieval_evidence",
                        "candidate_count": len(useful_candidates),
                        "top_score": round(useful_candidates[0].combined_score, 3),
                    }},
                )
                continue
            self._append_no_evidence_result(
                requirement_id,
                0.75,
                (
                    f"Rule 'requirement_with_no_evidence' (fallback): "
                    f"No retrieval candidates above floor for {requirement_id}."
                ),
                resolved,
                resolved_ids,
            )

    def _apply_orphan_evaluations(
        self,
        requirements: list[dict[str, Any]],
        evaluations: list[dict[str, Any]],
        resolved: list[CorrelationResult],
        ambiguous: list[dict[str, Any]],
    ) -> None:
        """Apply rule 5 to unlinked evaluations."""
        for evaluation in evaluations:
            if evaluation.get("linked_requirement"):
                continue
            matched_requirement = self._find_matching_req(
                evaluation.get("text", ""), requirements
            )
            if matched_requirement:
                ambiguous.append({
                    "type": "orphan_evaluation_with_candidate_req",
                    "requirement": matched_requirement,
                    "evidence": evaluation,
                    "retrieval_candidates": [],
                    "linked_evaluations": [],
                    "hint": "Evaluation evidence with possible requirement match but not explicitly linked.",
                })
                continue
            resolved.append(
                CorrelationResult(
                    requirement_entity_id="(unlinked)",
                    evidence_entity_id=evaluation["entity_id"],
                    status=CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT,
                    resolution_method="deterministic_rule",
                    rule_name="orphan_evaluation",
                    confidence=0.80,
                    supporting_chunk_ids=[evaluation["chunk_id"]],
                    reasoning=(
                        f"Rule 'orphan_evaluation': Evaluation '{evaluation['entity_id']}' "
                        f"has no linked requirement and no keyword-matching requirement was found."
                    ),
                )
            )
            logger.info(
                "Deterministic rule fired",
                extra={"context": {
                    "evidence_id": evaluation["entity_id"],
                    "rule": "orphan_evaluation",
                    "status": "EVALUATION_WITHOUT_REQUIREMENT",
                }},
            )

    def _collect_verbal_claims(
        self,
        requirements: list[dict[str, Any]],
        implementations: list[dict[str, Any]],
        retrieval_candidates: dict[str, list[RetrievedEvidence]],
        ambiguous: list[dict[str, Any]],
    ) -> None:
        """Apply rule 6 to conversational implementation claims."""
        for implementation in implementations:
            if not self._is_claim_without_concrete_evidence(implementation):
                continue
            requirement_id = implementation.get("linked_requirement") or "(unknown)"
            requirement = next(
                (item for item in requirements if item["entity_id"] == requirement_id), None
            )
            candidates = (
                retrieval_candidates.get(requirement_id, [])[:3]
                if requirement_id != "(unknown)"
                else []
            )
            ambiguous.append({
                "type": "verbal_claim_without_evidence",
                "requirement": requirement,
                "evidence": implementation,
                "retrieval_candidates": candidates,
                "linked_evaluations": [],
                "hint": (
                    "Implementation claim from conversational source — verify if "
                    "concrete evidence exists in retrieved chunks."
                ),
            })

    def _determine_status_with_evals(
        self, evaluations: list[dict[str, Any]]
    ) -> CorrelationStatus:
        """
        Determine correlation status based on linked evaluation evidence.
        """
        if not evaluations:
            return CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION

        positive = any(_is_positive_eval(e.get("text", "")) for e in evaluations)
        negative = any(_is_negative_eval(e.get("text", "")) for e in evaluations)

        if positive and not negative:
            return CorrelationStatus.IMPLEMENTED_AND_VALIDATED
        if negative:
            return CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED
        return CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION

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
    def _is_claim_without_concrete_evidence(impl: dict[str, Any]) -> bool:
        """
        Detect if an implementation row is a verbal claim from a conversational source.
        """
        text = impl.get("text", "").lower()
        source = impl.get("source_document", "").lower()

        has_claim = any(marker in text for marker in CLAIM_MARKERS)
        is_conversational = any(
            s in source for s in [".txt", "whatsapp", "chat", "email", ".eml"]
        )

        return has_claim and is_conversational
