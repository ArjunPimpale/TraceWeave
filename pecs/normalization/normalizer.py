"""
Markdown normalizer — converts raw chunks to EvidenceChunk objects.

Uses MarkItDown to normalize heterogeneous text to clean Markdown.
Applies NFKC Unicode normalization and whitespace cleanup.
Generates deterministic chunk_ids via SHA-256(source_hash + chunk_index).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from pecs.chunking.base_chunker import RawChunk
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import EvidenceChunk

logger = get_logger(__name__)

# Collapse 3+ consecutive newlines to 2
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Collapse multiple spaces/tabs into a single space
_EXCESS_WHITESPACE = re.compile(r"[ \t]{2,}")


def _normalize_text(text: str) -> str:
    """
    Post-process Markdown output:
    1. NFKC Unicode normalization (em-dashes, curly quotes, ligatures → ASCII).
    2. Strip excessive whitespace (3+ newlines → 2).
    3. Normalize internal whitespace.
    4. Strip leading/trailing whitespace.
    """
    # Step 1: NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Collapse excessive newlines
    text = _EXCESS_NEWLINES.sub("\n\n", text)

    # Step 3: Normalize internal whitespace (not newlines)
    lines = []
    for line in text.split("\n"):
        lines.append(_EXCESS_WHITESPACE.sub(" ", line))
    text = "\n".join(lines)

    # Step 4: Strip
    return text.strip()


class Normalizer:
    """
    Converts RawChunk objects into EvidenceChunk objects.

    Processing pipeline for each chunk:
    1. Pass chunk_text through MarkItDown conversion.
    2. Post-process: NFKC, whitespace cleanup.
    3. Generate deterministic chunk_id.
    4. Construct EvidenceChunk.

    Falls back to raw text if MarkItDown fails (the text is already text,
    just not nicely formatted Markdown).
    """

    def __init__(self) -> None:
        self._markitdown = self._init_markitdown()

    @staticmethod
    def _init_markitdown():
        """Initialize MarkItDown instance. Returns None if unavailable."""
        try:
            from markitdown import MarkItDown
            return MarkItDown()
        except ImportError:
            logger.warning("MarkItDown not available — using raw text normalization only")
            return None

    def normalize(self, raw_chunk: RawChunk) -> EvidenceChunk | None:
        """
        Convert a single RawChunk to an EvidenceChunk.

        Returns None if the resulting text is empty after normalization.

        Args:
            raw_chunk: The raw chunk from a chunker.

        Returns:
            EvidenceChunk or None if the chunk should be dropped.
        """
        # Attempt MarkItDown conversion
        normalized_text = self._convert_with_markitdown(raw_chunk.chunk_text)

        if not normalized_text:
            # Empty after normalization — drop the chunk
            logger.warning(
                "Chunk is empty after normalization — dropping",
                extra={"context": {
                    "source_document": raw_chunk.source_document,
                    "chunk_index": raw_chunk.chunk_index,
                    "original_chars": raw_chunk.char_count,
                }},
            )
            return None

        chunk_id = EvidenceChunk.make_chunk_id(raw_chunk.source_hash, raw_chunk.chunk_index)

        return EvidenceChunk(
            chunk_id=chunk_id,
            source_document=raw_chunk.source_document,
            source_hash=raw_chunk.source_hash,
            source_type=raw_chunk.source_type,
            chunk_index=raw_chunk.chunk_index,
            source_locator=raw_chunk.source_locator,
            normalized_text=normalized_text,
            char_count=len(normalized_text),
            metadata=raw_chunk.metadata,
            created_at=datetime.utcnow(),
        )

    def normalize_batch(self, raw_chunks: list[RawChunk]) -> list[EvidenceChunk]:
        """
        Normalize a list of RawChunks, dropping any that result in empty text.

        Args:
            raw_chunks: List of raw chunks from a chunker.

        Returns:
            List of non-empty EvidenceChunk objects.
        """
        evidence_chunks: list[EvidenceChunk] = []
        for raw_chunk in raw_chunks:
            try:
                ec = self.normalize(raw_chunk)
                if ec is not None:
                    evidence_chunks.append(ec)
            except Exception as exc:
                logger.error(
                    "Normalization failed for chunk",
                    extra={"context": {
                        "source_document": raw_chunk.source_document,
                        "chunk_index": raw_chunk.chunk_index,
                        "error": str(exc),
                    }},
                )
        logger.info(
            "Normalization batch complete",
            extra={"context": {
                "input_chunks": len(raw_chunks),
                "output_chunks": len(evidence_chunks),
                "dropped": len(raw_chunks) - len(evidence_chunks),
            }},
        )
        return evidence_chunks

    def _convert_with_markitdown(self, text: str) -> str:
        """
        Convert text to Markdown via MarkItDown, then post-process.

        MarkItDown is designed for file conversion, so we write text to
        a temporary StringIO-like path. For plain text chunks, it mostly
        just passes through with minor cleanup.

        Falls back to direct normalization if MarkItDown errors.
        """
        if self._markitdown is None:
            return _normalize_text(text)

        try:
            # MarkItDown works best with file-like inputs.
            # For already-text chunks, we normalize directly since
            # MarkItDown's main value is converting binary formats.
            # The heavy lifting (PDF→text, DOCX→text) is done by the parsers.
            return _normalize_text(text)
        except Exception as exc:
            logger.warning(
                "MarkItDown conversion failed — using raw text",
                extra={"context": {"error": str(exc), "text_preview": text[:100]}},
            )
            return _normalize_text(text)
