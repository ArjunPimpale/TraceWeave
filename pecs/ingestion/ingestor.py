"""
Ingestion orchestrator — the system's front door.

Accepts file bytes + metadata, validates them, detects the file type,
dispatches to the correct parser, chunks, normalizes, embeds, and stores.

This is the single entry point for all file ingestion. The Streamlit UI
calls ingest_file() for each uploaded file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pecs.chunking.docx_chunker import DOCXChunker
from pecs.chunking.email_chunker import EmailChunker
from pecs.chunking.git_chunker import GitChunker
from pecs.chunking.markdown_chunker import MarkdownChunker
from pecs.chunking.pdf_chunker import PDFChunker
from pecs.chunking.python_chunker import PythonChunker
from pecs.chunking.whatsapp_chunker import WhatsAppChunker
from pecs.ingestion.file_utils import (
    compute_sha256,
    detect_mime_type,
    get_extension,
    sniff_email,
    sniff_git_log,
    sniff_whatsapp,
    validate_file_size,
    validate_not_empty,
)
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.normalization.normalizer import Normalizer
from pecs.parsing.base_parser import ParseRequest
from pecs.parsing.docx_parser import DOCXParser
from pecs.parsing.email_parser import EmailParser
from pecs.parsing.git_parser import GitParser
from pecs.parsing.markdown_parser import MarkdownParser
from pecs.parsing.pdf_parser import PDFParser
from pecs.parsing.python_parser import PythonParser
from pecs.parsing.whatsapp_parser import WhatsAppParser
from pecs.store.evidence_repo import EvidenceRepo

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """
    Summary of a completed ingestion operation.

    Attributes:
        success: Whether ingestion completed without fatal errors.
        filename: The ingested file's name.
        file_hash: SHA-256 of the file bytes.
        source_type: Detected source type.
        chunk_count: Number of EvidenceChunks produced.
        skipped: Whether this file was skipped (already ingested).
        warnings: Non-fatal issues encountered.
        error: Fatal error message if success is False.
        evidence_chunks: The produced EvidenceChunk objects.
    """
    success: bool
    filename: str
    file_hash: str
    source_type: str
    chunk_count: int
    skipped: bool = False
    warnings: list[str] = None
    error: str | None = None
    evidence_chunks: list[EvidenceChunk] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.evidence_chunks is None:
            self.evidence_chunks = []


class Ingestor:
    """
    Orchestrates the full ingestion pipeline for a single file.

    Pipeline:
    1. Validate (size, empty)
    2. Compute hash → check for duplicates
    3. Detect MIME type → determine source type
    4. Dispatch to parser
    5. Dispatch to chunker
    6. Normalize chunks
    7. Return EvidenceChunks (embedding + storage happens in the embedder/vectorstore)
    """

    def __init__(self, evidence_repo: EvidenceRepo | None = None) -> None:
        self._repo = evidence_repo or EvidenceRepo()
        self._normalizer = Normalizer()

        # Parser registry
        self._parsers = {
            SourceType.PDF: PDFParser(),
            SourceType.DOCX: DOCXParser(),
            SourceType.EMAIL: EmailParser(),
            SourceType.WHATSAPP: WhatsAppParser(),
            SourceType.GIT: GitParser(),
            SourceType.MARKDOWN: MarkdownParser(),
            SourceType.PYTHON: PythonParser(),
        }

        # Chunker registry
        self._chunkers = {
            SourceType.PDF: PDFChunker(),
            SourceType.DOCX: DOCXChunker(),
            SourceType.EMAIL: EmailChunker(),
            SourceType.WHATSAPP: WhatsAppChunker(),
            SourceType.GIT: GitChunker(),
            SourceType.MARKDOWN: MarkdownChunker(),
            SourceType.PYTHON: PythonChunker(),
        }

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        user_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """
        Ingest a single file through the full pipeline.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original filename.
            user_metadata: Optional metadata from the user (project, description, etc.).

        Returns:
            IngestionResult with status, chunk count, and produced chunks.
        """
        user_metadata = user_metadata or {}
        start_time = datetime.utcnow()

        logger.info(
            "File upload started",
            extra={"context": {
                "filename": filename,
                "file_size": len(file_bytes),
            }},
        )

        # ── Step 1: Validate ──────────────────────────────────────────────
        try:
            validate_not_empty(file_bytes, filename)
            validate_file_size(file_bytes, filename)
        except ValueError as exc:
            logger.warning(
                "File validation failed",
                extra={"context": {"filename": filename, "error": str(exc)}},
            )
            return IngestionResult(
                success=False,
                filename=filename,
                file_hash="",
                source_type="UNKNOWN",
                chunk_count=0,
                error=str(exc),
            )

        # ── Step 2: Deduplication check ───────────────────────────────────
        file_hash = compute_sha256(file_bytes)
        if self._repo.is_file_ingested(file_hash):
            logger.warning(
                "Duplicate file detected — skipping",
                extra={"context": {"filename": filename, "hash": file_hash[:16]}},
            )
            return IngestionResult(
                success=True,
                filename=filename,
                file_hash=file_hash,
                source_type="UNKNOWN",
                chunk_count=0,
                skipped=True,
                warnings=[f"File already ingested (SHA-256: {file_hash[:16]}…)"],
            )

        # ── Step 3: MIME detection + source type resolution ───────────────
        mime_type = detect_mime_type(file_bytes, filename)
        source_type = self._resolve_source_type(mime_type, filename, file_bytes)

        if source_type is None:
            msg = (
                f"Unsupported file type: MIME={mime_type!r}, "
                f"extension={filename.rsplit('.', 1)[-1]!r}. "
                f"Supported: PDF, DOCX, Email, WhatsApp, Git log, Markdown, Python."
            )
            logger.warning(
                "File type unsupported",
                extra={"context": {"filename": filename, "mime": mime_type}},
            )
            return IngestionResult(
                success=False,
                filename=filename,
                file_hash=file_hash,
                source_type="UNKNOWN",
                chunk_count=0,
                error=msg,
            )

        # ── Step 4: Parse ─────────────────────────────────────────────────
        try:
            parse_request = ParseRequest(
                file_bytes=file_bytes,
                filename=filename,
                file_hash=file_hash,
                mime_type=mime_type,
                user_metadata=user_metadata,
                upload_timestamp=start_time.isoformat(),
            )
            parser = self._parsers[source_type]
            parsed_doc = parser.parse(parse_request)
        except Exception as exc:
            logger.error(
                "Parsing failed",
                extra={"context": {"filename": filename, "error": str(exc)}},
            )
            return self._processing_failure_result(
                filename, file_hash, file_bytes, source_type, "Parsing", exc
            )

        warnings = parsed_doc.parse_warnings.copy()

        # ── Step 5: Chunk ─────────────────────────────────────────────────
        try:
            chunker = self._chunkers[source_type]
            raw_chunks = chunker.chunk(parsed_doc)
        except Exception as exc:
            logger.error(
                "Chunking failed",
                extra={"context": {"filename": filename, "error": str(exc)}},
            )
            return self._processing_failure_result(
                filename, file_hash, file_bytes, source_type, "Chunking", exc, warnings
            )

        # ── Step 6: Normalize ─────────────────────────────────────────────
        evidence_chunks = self._normalizer.normalize_batch(raw_chunks)

        # ── Step 7: Log ingestion ─────────────────────────────────────────
        elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        self._repo.log_ingestion(
            filename=filename,
            file_hash=file_hash,
            file_size=len(file_bytes),
            source_type=source_type.value,
            chunk_count=len(evidence_chunks),
            status="success",
        )

        logger.info(
            "Ingestion successful",
            extra={"context": {
                "filename": filename,
                "hash": file_hash[:16],
                "source_type": source_type.value,
                "chunk_count": len(evidence_chunks),
                "parse_time_ms": elapsed_ms,
                "warnings": len(warnings),
            }},
        )

        return IngestionResult(
            success=True,
            filename=filename,
            file_hash=file_hash,
            source_type=source_type.value,
            chunk_count=len(evidence_chunks),
            warnings=warnings,
            evidence_chunks=evidence_chunks,
        )

    def _processing_failure_result(
        self,
        filename: str,
        file_hash: str,
        file_bytes: bytes,
        source_type: SourceType,
        stage: str,
        error: Exception,
        warnings: list[str] | None = None,
    ) -> IngestionResult:
        """Log a parse or chunk failure and return its existing result shape."""
        self._repo.log_ingestion(
            filename,
            file_hash,
            len(file_bytes),
            source_type.value,
            0,
            "failed",
            str(error),
        )
        return IngestionResult(
            success=False,
            filename=filename,
            file_hash=file_hash,
            source_type=source_type.value,
            chunk_count=0,
            error=f"{stage} failed: {error}",
            warnings=warnings,
        )

    def _resolve_source_type(
        self, mime_type: str, filename: str, file_bytes: bytes
    ) -> SourceType | None:
        """
        Map MIME type + extension + content heuristics to a SourceType.

        For ambiguous text/plain files, uses content sniffing to distinguish
        between WhatsApp exports, email exports, and Git log exports.
        """
        # Direct MIME mapping
        mime_to_type: dict[str, SourceType] = {
            "application/pdf": SourceType.PDF,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": SourceType.DOCX,
            "application/msword": SourceType.DOCX,
            "message/rfc822": SourceType.EMAIL,
            "text/markdown": SourceType.MARKDOWN,
            "text/x-markdown": SourceType.MARKDOWN,
            "text/x-python": SourceType.PYTHON,
            "application/x-python-code": SourceType.PYTHON,
        }

        if mime_type in mime_to_type:
            return mime_to_type[mime_type]

        # Extension-based resolution
        ext = get_extension(filename)
        ext_to_type: dict[str, SourceType] = {
            "pdf": SourceType.PDF,
            "docx": SourceType.DOCX,
            "doc": SourceType.DOCX,
            "eml": SourceType.EMAIL,
            "mbox": SourceType.EMAIL,
            "md": SourceType.MARKDOWN,
            "markdown": SourceType.MARKDOWN,
            "mdx": SourceType.MARKDOWN,
            "py": SourceType.PYTHON,
            "pyw": SourceType.PYTHON,
        }

        if ext in ext_to_type:
            return ext_to_type[ext]

        # For text/plain, use content sniffing
        if mime_type in ("text/plain", "application/octet-stream") or ext in ("txt", "log"):
            if sniff_whatsapp(file_bytes):
                return SourceType.WHATSAPP
            if sniff_email(file_bytes):
                return SourceType.EMAIL
            if sniff_git_log(file_bytes):
                return SourceType.GIT
            # Plain .txt could be Markdown
            if ext in ("txt",):
                return SourceType.MARKDOWN

        return None
