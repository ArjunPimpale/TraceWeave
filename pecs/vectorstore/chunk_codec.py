"""Conversion between ChromaDB records and ``EvidenceChunk`` objects."""

from __future__ import annotations

from typing import Any

from pecs.models.evidence_chunk import EvidenceChunk, SourceType


_CORE_METADATA_FIELDS = frozenset(
    {
        "source_document",
        "source_type",
        "source_hash",
        "chunk_index",
        "source_locator",
        "char_count",
        "created_at",
    }
)


def reconstruct_evidence_chunk(
    chunk_id: str,
    document: str,
    metadata: dict[str, Any],
    *,
    include_extra_metadata: bool = True,
) -> EvidenceChunk:
    """Reconstruct a chunk from a ChromaDB document and metadata record.

    Retrieval retains extra Chroma metadata for callers that use it. The Stage 1
    extraction page historically omitted that metadata, so it requests the
    explicit ``False`` mode below rather than changing its observable inputs.
    """
    try:
        source_type = SourceType(metadata.get("source_type", "MARKDOWN"))
    except ValueError:
        source_type = SourceType.MARKDOWN

    extra_metadata = (
        {key: value for key, value in metadata.items() if key not in _CORE_METADATA_FIELDS}
        if include_extra_metadata
        else {}
    )
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_document=metadata.get("source_document", ""),
        source_hash=metadata.get("source_hash", ""),
        source_type=source_type,
        chunk_index=metadata.get("chunk_index", 0),
        source_locator=metadata.get("source_locator", ""),
        normalized_text=document,
        char_count=len(document),
        metadata=extra_metadata,
    )
