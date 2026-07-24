"""Markdown chunker — heading-first, code-block-aware."""

from __future__ import annotations

import re

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument

_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)


class MarkdownChunker(BaseChunker):
    """
    Section-heading-first chunking for Markdown files.

    Strategy:
    1. Each section (heading + body until next same/higher heading) = one chunk.
    2. Code blocks are kept intact — never split mid-code-block.
    3. If section > max_chunk_chars, split at paragraph boundaries.
    4. Apply overlap_chars between chunks.

    Metadata: heading_text, heading_level, section_path.
    """

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        chunks: list[RawChunk] = []

        for idx, section in enumerate(document.structural_metadata):
            section_text = section.get("section_text", "").strip()
            if not section_text:
                continue

            heading_text = section.get("heading_text", "")
            heading_level = section.get("heading_level", 0)
            section_path = section.get("section_path", "")

            chunk_meta = {
                **base_meta,
                "heading_text": heading_text,
                "heading_level": heading_level,
                "section_path": section_path,
            }
            source_locator = section_path or heading_text or f"section {idx + 1}"

            chunks.append(RawChunk(
                chunk_text=section_text,
                chunk_index=idx,
                source_locator=source_locator,
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.MARKDOWN,
                metadata=chunk_meta,
            ))

        chunks = self._enforce_size_limits(chunks)
        chunks = self._apply_overlap(chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _split_at_paragraphs(self, text: str) -> list[str]:
        """
        Split at paragraph boundaries, but keep code blocks intact.
        """
        # Find code block positions
        code_blocks = list(_CODE_BLOCK_PATTERN.finditer(text))
        code_ranges = [(m.start(), m.end()) for m in code_blocks]

        def is_in_code_block(pos: int) -> bool:
            return any(start <= pos < end for start, end in code_ranges)

        # Split at double newlines that are NOT inside code blocks
        parts: list[str] = []
        current_start = 0
        i = 0
        while i < len(text) - 1:
            if text[i] == "\n" and text[i + 1] == "\n" and not is_in_code_block(i):
                part = text[current_start:i].strip()
                if part:
                    parts.append(part)
                current_start = i + 2
            i += 1

        remaining = text[current_start:].strip()
        if remaining:
            parts.append(remaining)

        return parts if parts else [text]
