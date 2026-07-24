"""
Abstract base class for all source-specific parsers.

Each parser adapter extracts both the text content and the structural
metadata that the chunker will use to decide chunk boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pecs.models.evidence_chunk import SourceType


@dataclass
class ParsedDocument:
    """
    The output of a parser adapter.

    Attributes:
        source_type: Which type of source this document is.
        raw_text: Full extracted text (before normalization).
        structural_metadata: Source-specific structure for the chunker
                              (e.g., page tuples for PDF, heading tuples for DOCX).
        global_metadata: Cross-source metadata (author, date, filename, hash).
        parse_warnings: Non-fatal issues encountered during parsing.
    """
    source_type: SourceType
    raw_text: str
    structural_metadata: list[Any]
    global_metadata: dict[str, Any]
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class ParseRequest:
    """
    Input to a parser adapter from the ingestion module.

    Attributes:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used for display and as source_document).
        file_hash: SHA-256 of file_bytes (precomputed by ingestion module).
        mime_type: MIME type detected by python-magic.
        user_metadata: Any additional metadata supplied by the user at upload time.
        upload_timestamp: ISO 8601 timestamp of when the file was uploaded.
    """
    file_bytes: bytes
    filename: str
    file_hash: str
    mime_type: str
    user_metadata: dict[str, Any] = field(default_factory=dict)
    upload_timestamp: str = ""


class BaseParser(ABC):
    """
    Abstract base class for all source-specific parser adapters.

    Subclasses must implement `parse()` and declare which MIME types and
    file extensions they handle.
    """

    #: MIME types this parser can handle (checked by the ingestion dispatcher)
    SUPPORTED_MIME_TYPES: tuple[str, ...] = ()

    #: File extensions this parser can handle (fallback when MIME is ambiguous)
    SUPPORTED_EXTENSIONS: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, request: ParseRequest) -> ParsedDocument:
        """
        Parse the file and return a ParsedDocument.

        Args:
            request: ParseRequest containing file bytes and metadata.

        Returns:
            ParsedDocument with raw text, structural metadata, and any warnings.

        Raises:
            ValueError: If the file cannot be parsed.
        """
        ...

    def can_handle(self, mime_type: str, extension: str) -> bool:
        """
        Return True if this parser can handle the given MIME type or extension.

        Used by the ingestion dispatcher to select the correct parser.
        """
        return (
            mime_type in self.SUPPORTED_MIME_TYPES
            or extension.lower().lstrip(".") in self.SUPPORTED_EXTENSIONS
        )
