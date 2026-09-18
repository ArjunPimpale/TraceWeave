"""
Traceability matrix builder.

Aggregates correlations from SQLite into a structured matrix where:
- Rows: Requirements
- Columns: Status, Confidence, Evidence count, Supporting sources, Reasoning

Full pipeline:
1. Load all requirements, implementations, evaluations from SQLite.
2. Run retrieval pipeline to get top-K evidence candidates per requirement.
3. Run deterministic rule engine (evidence-first).
4. Run Stage 2 LLM classifier on ambiguous pairs with real evidence.
5. Score confidence using real retrieval scores.
6. Store in SQLite.
7. Build and return TraceabilityMatrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pecs.correlation.confidence import ConfidenceScorer
from pecs.correlation.rules import RuleEngine
from pecs.correlation.stage2 import Stage2Classifier
from pecs.models.correlation_result import CorrelationStatus
from pecs.store.correlation_repo import CorrelationRepo
from pecs.store.evidence_repo import EvidenceRepo

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
    status: CorrelationStatus | None
    confidence: float
    evidence_count: int
    source_documents: list[str]
    reasoning: str
    supporting_chunk_ids: list[str]
    resolution_method: str
    color: str
    requirement_key: str = ""
    run_id: str = "legacy"
    workflow_state: str = "assessed"
    assessment_ids: list[str] = field(default_factory=list)
    conflicting: bool = False
    implementation_entities: int = 0
    evaluation_entities: int = 0
    assessment_context_chunks: int = 0
    retrieved_candidate_chunks: int = 0
    unresolved_references: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "status": self.status.value if self.status else None,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_documents": self.source_documents,
            "reasoning": self.reasoning,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "resolution_method": self.resolution_method,
            "color": self.color,
            "requirement_key": self.requirement_key,
            "run_id": self.run_id,
            "workflow_state": self.workflow_state,
            "assessment_ids": self.assessment_ids,
            "conflicting": self.conflicting,
            "implementation_entities": self.implementation_entities,
            "evaluation_entities": self.evaluation_entities,
            "assessment_context_chunks": self.assessment_context_chunks,
            "retrieved_candidate_chunks": self.retrieved_candidate_chunks,
            "unresolved_references": self.unresolved_references,
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
    run_id: str = "legacy"
    snapshot_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "summary": self.summary,
            "status_distribution": self.status_distribution,
            "per_source_coverage": self.per_source_coverage,
            "run_id": self.run_id,
            "snapshot_key": self.snapshot_key,
            "schema_version": 2,
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
        assessed = [r for r in self.rows if r.status is not None]
        return round(sum(r.confidence for r in assessed) / len(assessed), 3) if assessed else 0.0


class MatrixBuilder:
    """
    Builds the traceability matrix from SQLite data + retrieval pipeline.
    """

    def __init__(
        self,
        evidence_repo: EvidenceRepo | None = None,
        correlation_repo: CorrelationRepo | None = None,
        rule_engine: RuleEngine | None = None,
        stage2: Stage2Classifier | None = None,
        scorer: ConfidenceScorer | None = None,
        chroma_store=None,
    ) -> None:
        self._evidence_repo = evidence_repo or EvidenceRepo()
        self._correlation_repo = correlation_repo or CorrelationRepo()
        self._rules = rule_engine or RuleEngine()
        self._stage2 = stage2 or Stage2Classifier()
        self._scorer = scorer or ConfidenceScorer()
        self._chroma_store = chroma_store

    def build(
        self,
        retrieval_candidates: dict | None = None,
        use_stage2: bool = True,
    ) -> TraceabilityMatrix:
        """
        Build the complete traceability matrix.

        Args:
            retrieval_candidates: Dict mapping requirement_entity_id → top-K
                                  RetrievedEvidence from the vector store.
                                  If None, the pipeline runs without retrieval context
                                  (rules will fall back to metadata-only mode).
            use_stage2: If False, only deterministic rules are applied.

        Returns:
            TraceabilityMatrix with all rows and statistics.
        """
        from pecs.models.traceability import RetrievalOutcome
        from pecs.store.traceability_repo import TraceabilityRepo
        from pecs.traceability.service import TraceabilityService

        requirements = self._evidence_repo.get_all_requirements()
        names: dict[str, int] = {}
        duplicates: set[str] = set()
        for row in requirements:
            if row["entity_id"] in names:
                duplicates.add(row["entity_id"])
            names[row["entity_id"]] = row["id"]
        outcomes = {}
        for row in requirements:
            key = row["id"]
            value = (retrieval_candidates or {}).get(key)
            if value is None and row["entity_id"] not in duplicates:
                value = (retrieval_candidates or {}).get(row["entity_id"])
            outcomes[key] = value if isinstance(value, RetrievalOutcome) else RetrievalOutcome(value or [])
        service = TraceabilityService(
            evidence_repo=self._evidence_repo, rules=self._rules,
            stage2=self._stage2, scorer=self._scorer,
            trace_repo=TraceabilityRepo(db=self._evidence_repo._db),
            chroma=self._chroma_store,
        )
        run_id = service.run(outcomes, use_stage2=use_stage2)
        return self.load_latest(run_id)

    def load_latest(self, run_id: str | None = None) -> TraceabilityMatrix:
        """
        Load the most recent traceability matrix from SQLite.
        Returns the existing matrix without re-running the pipeline.
        """
        from pecs.traceability.reader import TraceabilityReader
        from pecs.store.traceability_repo import TraceabilityRepo
        return self._matrix_from_snapshot(TraceabilityReader(
            TraceabilityRepo(db=self._evidence_repo._db)).load(run_id))

    @staticmethod
    def _matrix_from_snapshot(snapshot: dict) -> TraceabilityMatrix:
        rows: list[MatrixRow] = []
        distribution: dict[str, int] = {}
        source_coverage: dict[str, int] = {}
        for req in snapshot["requirements"]:
            status = CorrelationStatus(req["status"]) if req["status"] else None
            row = MatrixRow(
                requirement_id=req["entity_id"], requirement_text=req["text"],
                status=status, confidence=req["confidence"], evidence_count=req["evidence_count"],
                source_documents=req["source_documents"], reasoning=req["reasoning"],
                supporting_chunk_ids=req["supporting_chunk_ids"],
                resolution_method=req["resolution_method"],
                color=STATUS_COLORS.get(status, "#6b7280"),
                requirement_key=f"ev:{snapshot['dataset_uuid']}:{req['row_id']}",
                run_id=snapshot["run_id"], workflow_state=req["workflow_state"],
                assessment_ids=req["assessment_ids"], conflicting=req["conflicting"],
                implementation_entities=req["implementation_entities"],
                evaluation_entities=req["evaluation_entities"],
                assessment_context_chunks=req["assessment_context_chunks"],
                retrieved_candidate_chunks=req["retrieved_candidate_chunks"],
                unresolved_references=req["unresolved_references"],
            )
            rows.append(row)
            bucket = status.value if status else "NOT_ASSESSED"
            distribution[bucket] = distribution.get(bucket, 0) + 1
            for source in set(row.source_documents):
                source_coverage[source] = source_coverage.get(source, 0) + 1
        rows.sort(key=lambda row: (row.requirement_id, row.requirement_key))
        result = TraceabilityMatrix(rows, {}, distribution, source_coverage,
                                    snapshot["run_id"], snapshot["snapshot_key"])
        result.summary = {
            "total_requirements": len(rows), "implemented_count": result.implemented_count,
            "coverage_percentage": result.coverage_percentage, "avg_confidence": result.avg_confidence,
            "assessed": sum(row.status is not None for row in rows),
            "unassessed": sum(row.status is None for row in rows),
            "deterministic_resolved": sum(row.resolution_method == "deterministic_rule" for row in rows),
            "llm_resolved": sum(row.resolution_method == "llm_stage2" for row in rows),
        }
        return result
