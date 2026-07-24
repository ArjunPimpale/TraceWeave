"""
Confidence scorer for CorrelationResult objects.

Computes a confidence score (0.0–1.0) for each correlation based on
four components with configurable weights:
- Retrieval score component (W_R = 0.35)
- Resolution method component (W_M = 0.30)
- Evidence count component (W_E = 0.20)
- Source diversity component (W_D = 0.15)
"""

from __future__ import annotations

import math
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus

logger = get_logger(__name__)


class ConfidenceScorer:
    """
    Computes normalized confidence scores for correlations.

    Confidence = W_R * retrieval_score + W_M * method_score
                 + W_E * evidence_count_score + W_D * diversity_score

    Where:
    - retrieval_score: Best combined retrieval score from the retrieval pipeline.
    - method_score: 1.0 for deterministic_rule, 0.7 for llm_stage2.
    - evidence_count_score: Normalized count of supporting evidence pieces.
    - diversity_score: Number of unique source types / max source types.
    """

    # Resolution method → base score
    _METHOD_SCORES: dict[str, float] = {
        "deterministic_rule": 1.0,
        "llm_stage2": 0.7,
    }

    # Status → believability factor (some statuses need more evidence to believe)
    _STATUS_BELIEVABILITY: dict[CorrelationStatus, float] = {
        CorrelationStatus.IMPLEMENTED_AND_VALIDATED: 1.0,
        CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION: 0.9,
        CorrelationStatus.PARTIALLY_IMPLEMENTED: 0.85,
        CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED: 0.85,
        CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED: 0.8,
        CorrelationStatus.CLAIMED_BUT_NO_EVIDENCE: 0.7,
        CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT: 0.75,
    }

    def __init__(self) -> None:
        self.W_R = settings.CONFIDENCE_W_R
        self.W_M = settings.CONFIDENCE_W_M
        self.W_E = settings.CONFIDENCE_W_E
        self.W_D = settings.CONFIDENCE_W_D

    def score(
        self,
        correlation: CorrelationResult,
        retrieval_score: float = 0.0,
        supporting_evidence: list[dict[str, Any]] | None = None,
    ) -> float:
        """
        Compute the confidence score for a correlation.

        Args:
            correlation: The CorrelationResult to score.
            retrieval_score: Best combined retrieval score (0.0–1.0).
            supporting_evidence: List of evidence rows from SQLite.

        Returns:
            Confidence score (0.0–1.0), rounded to 4 decimal places.
        """
        supporting_evidence = supporting_evidence or []

        # Component 1: Retrieval score
        r = min(max(retrieval_score, 0.0), 1.0)

        # Component 2: Resolution method
        m = self._METHOD_SCORES.get(correlation.resolution_method, 0.5)

        # Component 3: Evidence count (logarithmic scaling, max at 5+ pieces)
        n = len(supporting_evidence)
        if n == 0:
            e = 0.0
        elif n == 1:
            e = 0.5
        else:
            e = min(1.0, math.log(n + 1) / math.log(6))

        # Component 4: Source diversity (unique source types)
        source_types = {ev.get("source_type", "") for ev in supporting_evidence}
        d = min(1.0, len(source_types) / 3.0)  # Normalize to max of 3 source types

        raw_score = self.W_R * r + self.W_M * m + self.W_E * e + self.W_D * d

        # Apply believability factor
        believability = self._STATUS_BELIEVABILITY.get(correlation.status, 0.8)
        adjusted = raw_score * believability

        return round(min(adjusted, 1.0), 4)

    def score_batch(
        self,
        correlations: list[CorrelationResult],
        retrieval_scores: dict[str, float],
        all_evidence: list[dict[str, Any]],
    ) -> list[CorrelationResult]:
        """
        Score a batch of correlations, updating their confidence fields.

        Args:
            correlations: List of CorrelationResult objects.
            retrieval_scores: Dict mapping requirement_entity_id → retrieval score.
            all_evidence: All evidence rows from SQLite (for lookup by chunk_id).

        Returns:
            Updated list of CorrelationResult objects with confidence set.
        """
        # Build lookup: chunk_id → evidence row
        chunk_to_evidence: dict[str, dict[str, Any]] = {
            ev["chunk_id"]: ev for ev in all_evidence
        }

        updated: list[CorrelationResult] = []
        for corr in correlations:
            # Get supporting evidence rows
            supporting = [
                chunk_to_evidence[cid]
                for cid in corr.supporting_chunk_ids
                if cid in chunk_to_evidence
            ]
            retrieval_score = retrieval_scores.get(corr.requirement_entity_id, 0.0)
            confidence = self.score(corr, retrieval_score, supporting)

            updated.append(CorrelationResult(
                correlation_id=corr.correlation_id,
                requirement_entity_id=corr.requirement_entity_id,
                evidence_entity_id=corr.evidence_entity_id,
                status=corr.status,
                resolution_method=corr.resolution_method,
                rule_name=corr.rule_name,
                confidence=confidence,
                supporting_chunk_ids=corr.supporting_chunk_ids,
                reasoning=corr.reasoning,
                created_at=corr.created_at,
            ))

        logger.info(
            "Confidence scoring complete",
            extra={"context": {
                "correlation_count": len(correlations),
                "avg_confidence": round(
                    sum(c.confidence for c in updated) / max(len(updated), 1), 3
                ),
            }},
        )

        return updated
