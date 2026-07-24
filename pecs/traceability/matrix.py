"""
Traceability matrix builder.

Aggregates correlations from SQLite into a structured matrix where:
- Rows: Requirements
- Columns: Status, Confidence, Evidence count, Supporting sources, Reasoning

Also builds per-requirement citation lists and the summary statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pecs.correlation.confidence import ConfidenceScorer
from pecs.correlation.rules import RuleEngine
from pecs.correlation.stage2 import Stage2Classifier
from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus
from pecs.store.correlation_repo import CorrelationRepo
from pecs.store.evidence_repo import EvidenceRepo

logger = get_logger(__name__)

# Status → display color
STATUS_COLORS: dict[CorrelationStatus, str] = {
    CorrelationStatus.IMPLEMENTED_AND_VALIDATED: "#22c55e",      # Green
    CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION: "#3b82f6", # Blue
    CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED: "#f97316",  # Orange
    CorrelationStatus.PARTIALLY_IMPLEMENTED: "#eab308",          # Yellow
    CorrelationStatus.CLAIMED_BUT_NO_EVIDENCE: "#a855f7",        # Purple
    CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT: "#6b7280", # Gray
    CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED: "#ef4444",    # Red
}


@dataclass
class MatrixRow:
    """A single row in the traceability matrix."""
    requirement_id: str
    requirement_text: str
    status: CorrelationStatus
    confidence: float
    evidence_count: int
    source_documents: list[str]
    reasoning: str
    supporting_chunk_ids: list[str]
    resolution_method: str
    color: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_documents": self.source_documents,
            "reasoning": self.reasoning,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "resolution_method": self.resolution_method,
            "color": self.color,
        }


@dataclass
class TraceabilityMatrix:
    """
    The full traceability matrix.

    Contains:
    - rows: One MatrixRow per requirement
    - summary: Aggregate statistics
    - status_distribution: Count by status
    - per_source_coverage: How many requirements each source document covers
    """
    rows: list[MatrixRow] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    status_distribution: dict[str, int] = field(default_factory=dict)
    per_source_coverage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "summary": self.summary,
            "status_distribution": self.status_distribution,
            "per_source_coverage": self.per_source_coverage,
        }

    @property
    def total_requirements(self) -> int:
        return len(self.rows)

    @property
    def implemented_count(self) -> int:
        return sum(
            1 for r in self.rows
            if r.status in (
                CorrelationStatus.IMPLEMENTED_AND_VALIDATED,
                CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION,
                CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED,
            )
        )

    @property
    def coverage_percentage(self) -> float:
        if not self.rows:
            return 0.0
        return round(self.implemented_count / len(self.rows) * 100, 1)

    @property
    def avg_confidence(self) -> float:
        if not self.rows:
            return 0.0
        return round(sum(r.confidence for r in self.rows) / len(self.rows), 3)


class MatrixBuilder:
    """
    Builds the traceability matrix from SQLite data.

    Full pipeline:
    1. Load all requirements, implementations, evaluations from SQLite.
    2. Run deterministic rule engine.
    3. Run Stage 2 LLM classifier on ambiguous pairs.
    4. Score confidence for all correlations.
    5. Store in SQLite.
    6. Build and return TraceabilityMatrix.
    """

    def __init__(
        self,
        evidence_repo: EvidenceRepo | None = None,
        correlation_repo: CorrelationRepo | None = None,
        rule_engine: RuleEngine | None = None,
        stage2: Stage2Classifier | None = None,
        scorer: ConfidenceScorer | None = None,
    ) -> None:
        self._evidence_repo = evidence_repo or EvidenceRepo()
        self._correlation_repo = correlation_repo or CorrelationRepo()
        self._rules = rule_engine or RuleEngine()
        self._stage2 = stage2 or Stage2Classifier()
        self._scorer = scorer or ConfidenceScorer()

    def build(
        self,
        retrieval_scores: dict[str, float] | None = None,
        use_stage2: bool = True,
    ) -> TraceabilityMatrix:
        """
        Build the complete traceability matrix.

        Args:
            retrieval_scores: Dict mapping requirement_entity_id → best retrieval score.
                              Used by the confidence scorer.
            use_stage2: If False, only deterministic rules are applied.

        Returns:
            TraceabilityMatrix with all rows and statistics.
        """
        retrieval_scores = retrieval_scores or {}

        logger.info("Traceability matrix build started")

        # ── Load evidence ─────────────────────────────────────────────────
        requirements = self._evidence_repo.get_all_requirements()
        implementations = self._evidence_repo.get_all_implementations()
        evaluations = self._evidence_repo.get_all_evaluations()

        logger.info(
            "Evidence loaded",
            extra={"context": {
                "requirements": len(requirements),
                "implementations": len(implementations),
                "evaluations": len(evaluations),
            }},
        )

        if not requirements:
            logger.warning("No requirements found — empty matrix")
            return TraceabilityMatrix(
                summary={"warning": "No requirements found in evidence store."}
            )

        # ── Deterministic rules ───────────────────────────────────────────
        resolved, ambiguous = self._rules.apply_rules(
            requirements=requirements,
            implementations=implementations,
            evaluations=evaluations,
            retrieval_scores=retrieval_scores,
        )

        # ── Stage 2 LLM classification ────────────────────────────────────
        if use_stage2 and ambiguous:
            logger.info(
                "Stage 2 classification started",
                extra={"context": {"ambiguous_pairs": len(ambiguous)}},
            )
            stage2_results = self._stage2.classify_batch(ambiguous)
            resolved.extend(stage2_results)

        all_evidence = requirements + implementations + evaluations

        # ── Confidence scoring ────────────────────────────────────────────
        scored_correlations = self._scorer.score_batch(
            correlations=resolved,
            retrieval_scores=retrieval_scores,
            all_evidence=all_evidence,
        )

        # ── Store in SQLite ───────────────────────────────────────────────
        self._correlation_repo.insert_batch(scored_correlations)

        # ── Build matrix ──────────────────────────────────────────────────
        matrix = self._build_matrix_from_correlations(
            requirements=requirements,
            correlations=scored_correlations,
            all_evidence=all_evidence,
        )

        logger.info(
            "Traceability matrix built",
            extra={"context": {
                "total_requirements": matrix.total_requirements,
                "implemented": matrix.implemented_count,
                "coverage": f"{matrix.coverage_percentage}%",
                "avg_confidence": matrix.avg_confidence,
            }},
        )

        return matrix

    def load_latest(self) -> TraceabilityMatrix:
        """
        Load the most recent traceability matrix from SQLite.

        Returns the existing matrix without re-running the pipeline.
        """
        requirements = self._evidence_repo.get_all_requirements()
        all_evidence = self._evidence_repo.get_all()
        correlations_raw = self._correlation_repo.get_all_latest()

        # Convert raw dicts to CorrelationResult objects
        correlations: list[CorrelationResult] = []
        for row in correlations_raw:
            try:
                chunk_ids = json.loads(row.get("supporting_chunk_ids", "[]") or "[]")
                cr = CorrelationResult(
                    correlation_id=row.get("correlation_id", ""),
                    requirement_entity_id=row.get("requirement_entity_id", ""),
                    evidence_entity_id=row.get("evidence_entity_id", ""),
                    status=CorrelationStatus(row.get("status", "REQUIREMENT_NOT_IMPLEMENTED")),
                    resolution_method=row.get("resolution_method", "deterministic_rule"),
                    rule_name=row.get("rule_name"),
                    confidence=float(row.get("confidence", 0.0)),
                    supporting_chunk_ids=chunk_ids,
                    reasoning=row.get("reasoning", ""),
                )
                correlations.append(cr)
            except Exception as exc:
                logger.warning(
                    "Could not load correlation row",
                    extra={"context": {"error": str(exc)}},
                )

        return self._build_matrix_from_correlations(
            requirements=requirements,
            correlations=correlations,
            all_evidence=all_evidence,
        )

    def _build_matrix_from_correlations(
        self,
        requirements: list[dict[str, Any]],
        correlations: list[CorrelationResult],
        all_evidence: list[dict[str, Any]],
    ) -> TraceabilityMatrix:
        """Build TraceabilityMatrix from requirements + correlations."""

        # Build lookup: requirement_entity_id → correlation
        corr_by_req: dict[str, CorrelationResult] = {}
        for corr in correlations:
            req_id = corr.requirement_entity_id
            if req_id not in corr_by_req or corr.confidence > corr_by_req[req_id].confidence:
                corr_by_req[req_id] = corr

        # Build lookup: chunk_id → evidence row
        chunk_to_evidence: dict[str, dict[str, Any]] = {
            ev["chunk_id"]: ev for ev in all_evidence
        }

        rows: list[MatrixRow] = []
        status_distribution: dict[str, int] = {}
        per_source_coverage: dict[str, int] = {}

        for req in requirements:
            req_id = req["entity_id"]
            corr = corr_by_req.get(req_id)

            if corr is None:
                status = CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED
                confidence = 0.0
                supporting_chunks: list[str] = []
                reasoning = "No correlation computed."
                resolution_method = "none"
            else:
                status = corr.status
                confidence = corr.confidence
                supporting_chunks = corr.supporting_chunk_ids
                reasoning = corr.reasoning
                resolution_method = corr.resolution_method

            # Get source documents from supporting chunks
            source_docs = list(set(
                chunk_to_evidence[cid].get("source_document", "")
                for cid in supporting_chunks
                if cid in chunk_to_evidence
            ))

            evidence_count = len(supporting_chunks)

            for doc in source_docs:
                per_source_coverage[doc] = per_source_coverage.get(doc, 0) + 1

            status_distribution[status.value] = status_distribution.get(status.value, 0) + 1

            rows.append(MatrixRow(
                requirement_id=req_id,
                requirement_text=req.get("text", ""),
                status=status,
                confidence=confidence,
                evidence_count=evidence_count,
                source_documents=source_docs,
                reasoning=reasoning,
                supporting_chunk_ids=supporting_chunks,
                resolution_method=resolution_method,
                color=STATUS_COLORS.get(status, "#6b7280"),
            ))

        # Sort by requirement_id
        rows.sort(key=lambda r: r.requirement_id)

        implemented = sum(
            1 for r in rows
            if r.status in (
                CorrelationStatus.IMPLEMENTED_AND_VALIDATED,
                CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION,
                CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED,
            )
        )

        summary = {
            "total_requirements": len(rows),
            "implemented_count": implemented,
            "coverage_percentage": round(implemented / max(len(rows), 1) * 100, 1),
            "avg_confidence": round(sum(r.confidence for r in rows) / max(len(rows), 1), 3),
            "deterministic_resolved": sum(1 for r in rows if r.resolution_method == "deterministic_rule"),
            "llm_resolved": sum(1 for r in rows if r.resolution_method == "llm_stage2"),
        }

        return TraceabilityMatrix(
            rows=rows,
            summary=summary,
            status_distribution=status_distribution,
            per_source_coverage=per_source_coverage,
        )
