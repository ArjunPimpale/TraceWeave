"""
ExtractionResult — the Pydantic-validated output of Stage 1 extraction.

Maps directly to a row in the SQLite Evidence table. Pydantic validation
ensures type safety, required fields, and value constraints before anything
touches the database.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class EntityType(str, Enum):
    """
    The three entity types that Stage 1 can extract.

    REQUIREMENT: A stated project requirement or goal.
    IMPLEMENTATION: Evidence that something was built, coded, or delivered.
    EVALUATION: A professor's or evaluator's comment, score, ranking, or feedback.
    """
    REQUIREMENT = "REQUIREMENT"
    IMPLEMENTATION = "IMPLEMENTATION"
    EVALUATION = "EVALUATION"


class ExtractionResult(BaseModel):
    """
    A single extracted entity from a source chunk.

    This is the validated output of Stage 1 (Phi-4-mini extraction).
    It maps directly to a row in the SQLite evidence table.

    Attributes:
        entity_type: One of REQUIREMENT, IMPLEMENTATION, EVALUATION.
        entity_id: Descriptive identifier extracted from or derived from the text
                   (e.g., "R3-gnn-training", "impl-train-gnn"). Not a DB key.
        text: The extracted factual statement from the source. Must not be empty.
        linked_requirement: The requirement ID this entity explicitly references.
                            Null if no explicit reference is made. Stage 1 only
                            sets this if the text explicitly mentions the requirement —
                            it never infers.
        source_document: Inherited from the EvidenceChunk (original filename).
        chunk_id: Inherited from the EvidenceChunk (deterministic SHA-256 ID).
        author: Extracted or inherited author if identifiable; null otherwise.
        timestamp: Extracted or inherited date/time string if identifiable; null otherwise.
        metadata: Additional extracted metadata (flexible — source-specific fields).
    """

    model_config = ConfigDict(extra="ignore")  # Silently ignore extra fields from LLM

    entity_type: EntityType
    entity_id: str
    text: str
    linked_requirement: str | None = None
    source_document: str
    chunk_id: str
    author: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = {}

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v.strip()

    @field_validator("entity_id")
    @classmethod
    def entity_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("entity_id must not be empty")
        return v.strip()

    @field_validator("source_document")
    @classmethod
    def source_document_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_document must not be empty")
        return v.strip()

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chunk_id must not be empty")
        return v.strip()

    @field_validator("linked_requirement")
    @classmethod
    def normalize_linked_requirement(cls, v: str | None) -> str | None:
        """Normalize empty strings to None."""
        if v is not None and not v.strip():
            return None
        return v

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(type={self.entity_type.value}, "
            f"id={self.entity_id!r}, "
            f"doc={self.source_document!r})"
        )
