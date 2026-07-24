"""
PDF parser using PyMuPDF (fitz).

Extracts text page-by-page, preserving page numbers.
Detects and flags image-only pages (no extractable text — skipped per
the no-OCR constraint).
"""

from __future__ import annotations

import io
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """
    Parser for PDF documents using PyMuPDF.

    Structural metadata: list of dicts:
        {"page_number": int, "page_text": str}

    Image-only pages (where no text can be extracted) are skipped and
    logged as warnings. No OCR is performed.
    """

    SUPPORTED_MIME_TYPES = ("application/pdf",)
    SUPPORTED_EXTENSIONS = ("pdf",)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        import fitz  # PyMuPDF

        logger.debug(
            "PDF parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        pages: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        try:
            doc = fitz.open(stream=request.file_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"PyMuPDF could not open {request.filename!r}: {exc}") from exc

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text").strip()

            if not page_text:
                # Image-only page — no extractable text
                msg = f"Page {page_num + 1} has no extractable text (image-only?) — skipped"
                warnings.append(msg)
                logger.warning(
                    "Image-only PDF page skipped",
                    extra={"context": {
                        "filename": request.filename,
                        "page_number": page_num + 1,
                    }},
                )
                continue

            pages.append({"page_number": page_num + 1, "page_text": page_text})
            full_text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        total_pages = len(doc)
        doc.close()

        raw_text = "\n\n".join(full_text_parts)
        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.PDF.value,
            "page_count": total_pages,
            "text_page_count": len(pages),
            **request.user_metadata,
        }

        logger.info(
            "PDF parse completed",
            extra={"context": {
                "filename": request.filename,
                "text_pages": len(pages),
                "skipped_pages": len(warnings),
            }},
        )

        return ParsedDocument(
            source_type=SourceType.PDF,
            raw_text=raw_text,
            structural_metadata=pages,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )
