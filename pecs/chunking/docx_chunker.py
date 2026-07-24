"""DOCX chunker — section-heading-first chunking strategy."""

from __future__ import annotations

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


class DOCXChunker(BaseChunker):
    """
    Section-heading-first chunking for DOCX documents.

    Strategy:
    1. Each section (heading + body until next same/higher heading) = one chunk.
    2. Sections exceeding max_chunk_chars are split at paragraph boundaries.
    3. Apply overlap_chars between adjacent chunks.

    Metadata: heading_text, heading_level, section_path.
    """

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        chunks: list[RawChunk] = []
        idx = 0

        # Group consecutive body paragraphs under the same heading
        current_heading_level = 0
        current_heading_text = ""
        current_section_path = ""
        current_body_parts: list[str] = []

        def flush_section():
            nonlocal idx
            if not current_body_parts:
                return
            section_text = "\n\n".join(current_body_parts).strip()
            if not section_text:
                return
            locator = current_section_path or current_heading_text or "(body)"
            chunk_meta = {
                **base_meta,
                "heading_text": current_heading_text,
                "heading_level": current_heading_level,
                "section_path": current_section_path,
            }
            chunks.append(RawChunk(
                chunk_text=section_text,
                chunk_index=idx,
                source_locator=locator,
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.DOCX,
                metadata=chunk_meta,
            ))
            idx += 1

        for element in document.structural_metadata:
            heading_level = element.get("heading_level", 0)
            section_text = element.get("section_text", "").strip()

            if heading_level > 0:
                # New heading — flush accumulated body
                flush_section()
                current_heading_level = heading_level
                current_heading_text = element.get("heading_text", "")
                current_section_path = element.get("section_path", "")
                current_body_parts = [section_text]
            else:
                current_body_parts.append(section_text)

        flush_section()  # Flush the last section

        chunks = self._enforce_size_limits(chunks)
        chunks = self._apply_overlap(chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks
