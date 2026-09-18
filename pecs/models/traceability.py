"""Typed, immutable context around a correlation decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pecs.models.correlation_result import CorrelationResult

TargetKind = Literal["evidence", "chunk", "unresolved"]
ReferenceRole = Literal[
    "explicit_implementation", "explicit_evaluation", "rule_selected_chunk",
    "stage2_context", "retrieved_candidate", "legacy_citation", "legacy_primary_target",
]


@dataclass(frozen=True)
class AssessmentReference:
    role: ReferenceRole
    target_kind: TargetKind
    evidence_row_id: int | None = None
    chunk_id: str | None = None
    unresolved_value: str | None = None
    candidates: tuple[int, ...] = ()
    rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    combined_score: float | None = None
    retrieval_method: str | None = None
    id_match: bool | None = None
    visible_text: str | None = None

    def __post_init__(self) -> None:
        if self.target_kind == "evidence" and self.evidence_row_id is None:
            raise ValueError("Evidence references need a row ID")
        if self.target_kind == "chunk" and not self.chunk_id:
            raise ValueError("Chunk references need a chunk ID")
        if self.target_kind == "unresolved" and not self.unresolved_value:
            raise ValueError("Unresolved references need their original value")


@dataclass
class AssessmentEnvelope:
    result: CorrelationResult
    requirement_row_id: int | None
    scope: Literal["requirement_assessment", "evaluation_review", "claim_review"]
    sequence: int
    references: list[AssessmentReference] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalOutcome:
    candidates: list
    error: str | None = None
    bm25_available: bool = False
