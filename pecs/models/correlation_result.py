"""
CorrelationResult — a single link in the traceability matrix.

Connects a requirement to evidence, classifies the relationship into one of
the 7 CorrelationStatus labels, and provides the confidence score and
supporting citations needed for auditing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CorrelationStatus(str, Enum):
    """
    The seven classification labels for requirement-evidence correlations.

    These are the only valid output labels for both the deterministic rule
    engine (Stage 1) and the LLM classifier (Stage 2).
    """
    IMPLEMENTED_AND_VALIDATED = "IMPLEMENTED_AND_VALIDATED"
    IMPLEMENTED_BUT_NEGATIVELY_EVALUATED = "IMPLEMENTED_BUT_NEGATIVELY_EVALUATED"
    IMPLEMENTED_WITHOUT_EVALUATION = "IMPLEMENTED_WITHOUT_EVALUATION"
    PARTIALLY_IMPLEMENTED = "PARTIALLY_IMPLEMENTED"
    CLAIMED_BUT_NO_EVIDENCE = "CLAIMED_BUT_NO_EVIDENCE"
    EVALUATION_WITHOUT_REQUIREMENT = "EVALUATION_WITHOUT_REQUIREMENT"
    REQUIREMENT_NOT_IMPLEMENTED = "REQUIREMENT_NOT_IMPLEMENTED"


class CorrelationResult(BaseModel):
    """
    A single correlation in the traceability matrix.

    Represents the relationship between a requirement and a piece of
    implementation or evaluation evidence.

    Attributes:
        correlation_id: Auto-generated UUID for this correlation record.
        requirement_entity_id: The entity_id of the requirement being correlated.
        evidence_entity_id: The entity_id of the implementation/evaluation evidence.
        status: One of the 7 CorrelationStatus labels.
        resolution_method: "deterministic_rule" or "llm_stage2". Tracks whether
                           the correlation was resolved without the LLM, which
                           makes it more trustworthy.
        rule_name: If resolved deterministically, the name of the rule that fired
                   (e.g., "exact_requirement_id_match"). Enables debugging.
        confidence: Computed confidence score (0.0–1.0). Never LLM-generated —
                    always computed by the formula in confidence.py.
        supporting_chunk_ids: List of chunk_ids that provide supporting evidence
                              for this correlation. Enables citation generation.
        reasoning: Brief justification for the classification, citing specific
                   text from the evidence (from LLM or rule description).
        created_at: Timestamp for temporal auditing and version tracking.
    """

    model_config = ConfigDict(extra="ignore")

    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requirement_entity_id: str
    evidence_entity_id: str
    status: CorrelationStatus
    resolution_method: str  # "deterministic_rule" or "llm_stage2"
    rule_name: str | None = None
    confidence: float
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    @field_validator("resolution_method")
    @classmethod
    def resolution_method_must_be_valid(cls, v: str) -> str:
        valid = {"deterministic_rule", "llm_stage2"}
        if v not in valid:
            raise ValueError(f"resolution_method must be one of {valid}, got {v!r}")
        return v

    def __repr__(self) -> str:
        return (
            f"CorrelationResult(req={self.requirement_entity_id!r}, "
            f"evidence={self.evidence_entity_id!r}, "
            f"status={self.status.value}, "
            f"confidence={self.confidence:.2f})"
        )
