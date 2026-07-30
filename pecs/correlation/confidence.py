"""
Confidence scorer for CorrelationResult objects.

Computes a confidence score (0.0–1.0) for each correlation based on
four components with configurable weights:
- Retrieval score component (W_R = 0.35): Best combined retrieval score
- Resolution method component (W_M = 0.30): Deterministic > LLM
- Evidence count component (W_E = 0.20): Number of supporting chunks
- Source diversity component (W_D = 0.15): Variety of source types

Key fix: Now accepts RetrievedEvidence objects directly so retrieval
scores are real values from the vector store, not 0.0 from a missing dict.
"""

from __future__ import annotations

import math
from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus
from pecs.models.retrieved_evidence import RetrievedEvidence

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
    - diversity_score: Number of unique source documents / max.
    """

    # Resolution method → base score
    _METHOD_SCORES: dict[str, float] = {
        "deterministic_rule": 1.0,
        "llm_stage2": 0.7,
    }

    # Status → believability factor
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
        retrieval_candidates: list[RetrievedEvidence] | None = None,
    ) -> float:
        """
        Compute the confidence score for a correlation.

        Args:
            correlation: The CorrelationResult to score.
            retrieval_score: Best combined retrieval score (0.0–1.0).
            supporting_evidence: Legacy — list of evidence rows from SQLite.
            retrieval_candidates: New — list of RetrievedEvidence from vector store.

        Returns:
            Confidence score (0.0–1.0), rounded to 4 decimal places.
        """
        supporting_evidence = supporting_evidence or []
        retrieval_candidates = retrieval_candidates or []

        # Component 1: Retrieval score
        # Use actual candidates if available; otherwise fall back to the passed float
        if retrieval_candidates:
            r = min(max(retrieval_candidates[0].combined_score, 0.0), 1.0)
        else:
            r = min(max(retrieval_score, 0.0), 1.0)

        # Component 2: Resolution method
        m = self._METHOD_SCORES.get(correlation.resolution_method, 0.5)

        # Component 3: Evidence count (from candidates or supporting chunk IDs)
        n = len(retrieval_candidates) if retrieval_candidates else len(supporting_evidence)
        if n == 0:
            # Fall back to chunk count from the correlation result
            n = len(correlation.supporting_chunk_ids)
        if n == 0:
            e = 0.0
        elif n == 1:
            e = 0.5
        else:
            e = min(1.0, math.log(n + 1) / math.log(6))

        # Component 4: Source diversity (unique source documents)
        if retrieval_candidates:
            source_docs = {c.chunk.source_document for c in retrieval_candidates}
        else:
            source_docs = {ev.get("source_document", "") for ev in supporting_evidence}
        d = min(1.0, len(source_docs) / 3.0)

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
        retrieval_candidates: dict[str, list[RetrievedEvidence]] | None = None,
    ) -> list[CorrelationResult]:
        """
        Score a batch of correlations, updating their confidence fields.

        Args:
            correlations: List of CorrelationResult objects.
            retrieval_scores: Dict mapping requirement_entity_id → retrieval score.
                              Used as fallback if no retrieval_candidates provided.
            all_evidence: All evidence rows from SQLite (for fallback lookup).
            retrieval_candidates: Dict mapping requirement_entity_id → top-K candidates.
                                  When provided, overrides retrieval_scores for scoring.

        Returns:
            Updated list of CorrelationResult objects with confidence set.
        """
        retrieval_candidates = retrieval_candidates or {}

        # Build lookup: chunk_id → evidence row (for legacy path)
        chunk_to_evidence: dict[str, dict[str, Any]] = {
            ev["chunk_id"]: ev for ev in all_evidence
        }

        updated: list[CorrelationResult] = []
        for corr in correlations:
            req_id = corr.requirement_entity_id

            # Prefer retrieval candidates; fall back to scores dict
            candidates = retrieval_candidates.get(req_id, [])
            retrieval_score = retrieval_scores.get(req_id, 0.0)

            # Legacy supporting evidence lookup
            supporting = [
                chunk_to_evidence[cid]
                for cid in corr.supporting_chunk_ids
                if cid in chunk_to_evidence
            ]

            confidence = self.score(
                corr,
                retrieval_score=retrieval_score,
                supporting_evidence=supporting,
                retrieval_candidates=candidates,
            )

            updated.append(CorrelationResult(
                correlation_id=corr.correlation_id,
                requirement_entity_id=req_id,
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
