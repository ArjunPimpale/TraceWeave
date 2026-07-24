"""
Universal internal object models for PECS.

These data classes and Pydantic models are the system's internal API.
Every module produces and consumes these objects — no module should pass
raw dictionaries or untyped data to another.
"""

from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.models.retrieved_evidence import RetrievedEvidence
from pecs.models.extraction_result import ExtractionResult, EntityType
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus

__all__ = [
    "EvidenceChunk",
    "SourceType",
    "RetrievedEvidence",
    "ExtractionResult",
    "EntityType",
    "CorrelationResult",
    "CorrelationStatus",
]
