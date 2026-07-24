"""PDF chunker — page-first chunking strategy."""

from __future__ import annotations

from typing import Any

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


class PDFChunker(BaseChunker):
    """
    Page-first chunking for PDF documents.

    Strategy:
    1. Start with page boundaries from the PDF parser.
    2. Within each page, if text > max_chunk_chars, split at paragraph boundaries.
    3. If a paragraph > max_chunk_chars, split at sentence boundaries.
    4. Apply overlap_chars between adjacent chunks.

    Page numbers are the universal citation unit for PDFs.
    """

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        chunks: list[RawChunk] = []
        global_idx = 0

        for page_data in document.structural_metadata:
            page_num: int = page_data.get("page_number", 0)
            page_text: str = page_data.get("page_text", "").strip()

            if not page_text:
                continue

            page_meta = {**base_meta, "page_number": page_num}
            source_locator = f"page {page_num}"

            if len(page_text) <= self.max_chunk_chars:
                # Entire page fits in one chunk
                chunks.append(RawChunk(
                    chunk_text=page_text,
                    chunk_index=global_idx,
                    source_locator=source_locator,
                    source_document=document.global_metadata.get("filename", ""),
                    source_hash=document.global_metadata.get("file_hash", ""),
                    source_type=SourceType.PDF,
                    metadata=page_meta,
                ))
                global_idx += 1
            else:
                # Split within the page
                sub_chunks = self._split_page(page_text, page_num, document, base_meta)
                for sc in sub_chunks:
                    sc.chunk_index = global_idx
                    global_idx += 1
                chunks.extend(sub_chunks)

        chunks = self._enforce_size_limits(chunks)
        chunks = self._apply_overlap(chunks)

        # Re-index after all transformations
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _split_page(
        self,
        page_text: str,
        page_num: int,
        document: ParsedDocument,
        base_meta: dict,
    ) -> list[RawChunk]:
        paragraphs = self._split_at_paragraphs(page_text)
        chunks: list[RawChunk] = []
        current = ""
        sub_idx = 0

        page_meta = {**base_meta, "page_number": page_num}
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        for para in paragraphs:
            if current and len(current) + len(para) + 2 > self.max_chunk_chars:
                chunks.append(RawChunk(
                    chunk_text=current.strip(),
                    chunk_index=sub_idx,
                    source_locator=f"page {page_num}",
                    source_document=source_doc,
                    source_hash=source_hash,
                    source_type=SourceType.PDF,
                    metadata=page_meta,
                ))
                sub_idx += 1
                current = para
            else:
                current = (current + "\n\n" + para).strip() if current else para

        if current:
            chunks.append(RawChunk(
                chunk_text=current.strip(),
                chunk_index=sub_idx,
                source_locator=f"page {page_num}",
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.PDF,
                metadata=page_meta,
            ))

        return chunks
