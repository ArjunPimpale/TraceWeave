"""
EvidenceChunk — the universal internal currency of the PECS pipeline.

Every module from normalization onward works with EvidenceChunk objects.
It encapsulates both the text and its full provenance, ensuring that no
downstream module can accidentally lose provenance information.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Supported source document types."""
    PDF = "PDF"
    DOCX = "DOCX"
    EMAIL = "EMAIL"
    WHATSAPP = "WHATSAPP"
    GIT = "GIT"
    MARKDOWN = "MARKDOWN"
    PYTHON = "PYTHON"


@dataclass
class EvidenceChunk:
    """
    A single chunk of evidence from a source document.

    This is the universal internal object that flows through the pipeline
    from normalization onward. It encapsulates both the text content and
    its full provenance metadata.

    Attributes:
        chunk_id: Deterministic SHA-256(doc_hash + chunk_index).
                  Ensures the same document always produces the same chunk IDs.
        source_document: Original filename (not the full path).
        source_hash: SHA-256 of the original file bytes for deduplication.
        source_type: Enum indicating the type of source document.
        chunk_index: 0-indexed position of this chunk within the document.
        source_locator: Human-readable source-specific location string
                        (e.g., "page 3", "2024-01-15 14:22", "line 45-67").
                        A string rather than a structured type because different
                        sources have fundamentally different locator formats.
        normalized_text: Clean Markdown text after normalization via MarkItDown.
        char_count: Length of normalized_text for size tracking.
        metadata: Flexible JSON-serializable dict for source-specific metadata
                  (e.g., author, subject, commit_sha, heading_text).
        created_at: Timestamp of when this chunk was created.
    """

    chunk_id: str
    source_document: str
    source_hash: str
    source_type: SourceType
    chunk_index: int
    source_locator: str
    normalized_text: str
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def make_chunk_id(cls, source_hash: str, chunk_index: int) -> str:
        """
        Generate a deterministic chunk_id from the source document hash and chunk index.

        Deterministic IDs ensure:
        - Same document always produces the same chunk IDs
        - Idempotent upserts into ChromaDB
        - No duplicate vectors even if a document is re-ingested

        Args:
            source_hash: SHA-256 hash of the source file bytes.
            chunk_index: 0-indexed position of the chunk within the document.

        Returns:
            A hex string SHA-256 digest.
        """
        raw = f"{source_hash}::{chunk_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_chroma_metadata(self) -> dict[str, Any]:
        """
        Serialize metadata for storage in ChromaDB.

        ChromaDB metadata values must be str, int, float, or bool.
        Nested dicts and datetimes must be flattened/serialized.

        Returns:
            A flat dict suitable for ChromaDB's metadata parameter.
        """
        base = {
            "source_document": self.source_document,
            "source_type": self.source_type.value,
            "source_hash": self.source_hash,
            "chunk_index": self.chunk_index,
            "source_locator": self.source_locator,
            "char_count": self.char_count,
            "created_at": self.created_at.isoformat(),
        }
        # Flatten top-level metadata entries that are scalar-typed
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                base[f"meta_{k}"] = v
            elif v is None:
                pass  # Omit None values — ChromaDB does not accept None
        return base

    def __repr__(self) -> str:
        return (
            f"EvidenceChunk(id={self.chunk_id[:8]}…, "
            f"doc={self.source_document!r}, "
            f"type={self.source_type.value}, "
            f"idx={self.chunk_index}, "
            f"chars={self.char_count})"
        )
