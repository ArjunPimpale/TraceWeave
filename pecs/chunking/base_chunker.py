"""
Abstract base class for all source-specific chunkers.

Chunking determines retrieval granularity. Each chunker takes a ParsedDocument
and produces RawChunk objects with provenance metadata. The normalizer then
converts RawChunks into EvidenceChunks.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pecs.config import settings
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


@dataclass
class RawChunk:
    """
    A single raw text chunk before normalization.

    Produced by chunkers; consumed by the normalizer to create EvidenceChunks.

    Attributes:
        chunk_text: The raw text of the chunk (not yet normalized).
        chunk_index: 0-indexed position in the original document.
        source_locator: Source-specific location string (e.g., "page 3",
                        "2024-01-15 14:22:30", "lines 45-67").
        source_document: Original filename (inherited from ParsedDocument).
        source_hash: SHA-256 of the source file (inherited).
        source_type: SourceType enum value (inherited).
        metadata: Additional source-specific metadata for this chunk.
        char_count: Length of chunk_text.
    """
    chunk_text: str
    chunk_index: int
    source_locator: str
    source_document: str
    source_hash: str
    source_type: SourceType
    metadata: dict[str, Any] = field(default_factory=dict)
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.chunk_text)


# Sentence boundary regex (matches ., !, ? followed by whitespace + uppercase)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class BaseChunker(ABC):
    """
    Abstract base class for all source-specific chunkers.

    Subclasses implement the source-specific splitting strategy.
    BaseChunker provides common utilities: sentence splitting, overlap,
    size enforcement.
    """

    def __init__(
        self,
        max_chunk_chars: int | None = None,
        min_chunk_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars or settings.MAX_CHUNK_CHARS
        self.min_chunk_chars = min_chunk_chars or settings.MIN_CHUNK_CHARS
        self.overlap_chars = overlap_chars or settings.OVERLAP_CHARS

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        """
        Split a ParsedDocument into RawChunk objects.

        Args:
            document: The parsed document from a parser adapter.

        Returns:
            List of RawChunk objects with provenance metadata.
        """
        ...

    # ── Common utilities ──────────────────────────────────────────────────────

    def _split_at_sentences(self, text: str) -> list[str]:
        """Split text at sentence boundaries."""
        sentences = _SENTENCE_BOUNDARY.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_at_paragraphs(self, text: str) -> list[str]:
        """Split text at paragraph boundaries (double newline)."""
        paras = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paras if p.strip()]

    def _enforce_size_limits(self, chunks: list[RawChunk]) -> list[RawChunk]:
        """
        Ensure no chunk exceeds max_chunk_chars and no chunk is below
        min_chunk_chars (except possibly the last).

        Large chunks are split at sentence boundaries.
        Small chunks are merged with the adjacent chunk.
        """
        # Split oversized chunks
        result: list[RawChunk] = []
        for chunk in chunks:
            if chunk.char_count > self.max_chunk_chars:
                sub_chunks = self._split_large_chunk(chunk)
                result.extend(sub_chunks)
            else:
                result.append(chunk)

        # Merge undersized chunks (except the last)
        merged: list[RawChunk] = []
        i = 0
        while i < len(result):
            chunk = result[i]
            if chunk.char_count < self.min_chunk_chars and i < len(result) - 1:
                # Merge with next chunk
                next_chunk = result[i + 1]
                merged_text = chunk.chunk_text + "\n\n" + next_chunk.chunk_text
                result[i + 1] = RawChunk(
                    chunk_text=merged_text,
                    chunk_index=chunk.chunk_index,
                    source_locator=chunk.source_locator,
                    source_document=chunk.source_document,
                    source_hash=chunk.source_hash,
                    source_type=chunk.source_type,
                    metadata={**chunk.metadata, **next_chunk.metadata},
                )
                i += 1  # Skip (consumed into next)
            else:
                merged.append(chunk)
                i += 1

        # Re-index
        for idx, chunk in enumerate(merged):
            chunk.chunk_index = idx

        return merged

    def _split_large_chunk(self, chunk: RawChunk) -> list[RawChunk]:
        """Split a single oversized chunk into smaller pieces."""
        # Try paragraph splits first
        paragraphs = self._split_at_paragraphs(chunk.chunk_text)
        if len(paragraphs) <= 1:
            # Fall back to sentence splits
            paragraphs = self._split_at_sentences(chunk.chunk_text)
        if len(paragraphs) <= 1:
            # Can't split further — return as-is even if oversized
            return [chunk]

        sub_chunks: list[RawChunk] = []
        current_text = ""
        sub_idx = 0

        for para in paragraphs:
            if current_text and len(current_text) + len(para) + 2 > self.max_chunk_chars:
                sub_chunks.append(RawChunk(
                    chunk_text=current_text.strip(),
                    chunk_index=chunk.chunk_index + sub_idx,
                    source_locator=chunk.source_locator,
                    source_document=chunk.source_document,
                    source_hash=chunk.source_hash,
                    source_type=chunk.source_type,
                    metadata=chunk.metadata,
                ))
                sub_idx += 1
                current_text = para
            else:
                current_text = (current_text + "\n\n" + para).strip() if current_text else para

        if current_text:
            sub_chunks.append(RawChunk(
                chunk_text=current_text.strip(),
                chunk_index=chunk.chunk_index + sub_idx,
                source_locator=chunk.source_locator,
                source_document=chunk.source_document,
                source_hash=chunk.source_hash,
                source_type=chunk.source_type,
                metadata=chunk.metadata,
            ))

        return sub_chunks

    def _apply_overlap(self, chunks: list[RawChunk]) -> list[RawChunk]:
        """
        Apply character overlap between adjacent chunks.

        The last `overlap_chars` characters of chunk N are prepended
        to chunk N+1 as context.
        """
        if len(chunks) <= 1 or self.overlap_chars <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].chunk_text
            overlap = prev_text[-self.overlap_chars:] if len(prev_text) > self.overlap_chars else prev_text
            chunks[i] = RawChunk(
                chunk_text=(overlap + "\n\n" + chunks[i].chunk_text).strip(),
                chunk_index=chunks[i].chunk_index,
                source_locator=chunks[i].source_locator,
                source_document=chunks[i].source_document,
                source_hash=chunks[i].source_hash,
                source_type=chunks[i].source_type,
                metadata=chunks[i].metadata,
            )
            result.append(chunks[i])

        return result

    def _base_metadata(self, document: ParsedDocument) -> dict[str, Any]:
        """Extract common metadata fields from a ParsedDocument."""
        return {
            "source_type": document.source_type.value,
            **{k: v for k, v in document.global_metadata.items()
               if isinstance(v, (str, int, float, bool))},
        }
