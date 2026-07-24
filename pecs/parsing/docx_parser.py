"""
DOCX parser using python-docx.

Walks the paragraph tree, tracking heading levels to build a section hierarchy.
Extracts tables as Markdown-style tables.
"""

from __future__ import annotations

import io
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)

# python-docx heading style names follow the pattern "Heading 1", "Heading 2", etc.
_HEADING_STYLES = {f"Heading {i}": i for i in range(1, 10)}


class DOCXParser(BaseParser):
    """
    Parser for .docx documents using python-docx.

    Structural metadata: list of dicts:
        {
            "heading_level": int,       # 0 = body text (no heading)
            "heading_text": str,        # "" for body paragraphs
            "section_text": str,        # Text content of this element
            "section_path": str,        # Breadcrumb: "2 > 2.1 > 2.1.3"
        }
    """

    SUPPORTED_MIME_TYPES = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    )
    SUPPORTED_EXTENSIONS = ("docx", "doc")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        from docx import Document
        from docx.oxml.ns import qn

        logger.debug(
            "DOCX parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        sections: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        try:
            doc = Document(io.BytesIO(request.file_bytes))
        except Exception as exc:
            raise ValueError(f"python-docx could not open {request.filename!r}: {exc}") from exc

        # Track the current heading hierarchy for section_path generation
        heading_stack: list[tuple[int, str]] = []  # [(level, text), ...]
        has_headings = False

        for para in doc.paragraphs:
            style_name = para.style.name if para.style else ""
            heading_level = _HEADING_STYLES.get(style_name, 0)
            text = para.text.strip()

            if not text:
                continue

            if heading_level > 0:
                has_headings = True
                # Pop the stack until we find a lower-level heading
                while heading_stack and heading_stack[-1][0] >= heading_level:
                    heading_stack.pop()
                heading_stack.append((heading_level, text))

                section_path = " > ".join(h for _, h in heading_stack)
                sections.append({
                    "heading_level": heading_level,
                    "heading_text": text,
                    "section_text": text,
                    "section_path": section_path,
                })
                full_text_parts.append(f"{'#' * heading_level} {text}")
            else:
                section_path = " > ".join(h for _, h in heading_stack)
                sections.append({
                    "heading_level": 0,
                    "heading_text": "",
                    "section_text": text,
                    "section_path": section_path,
                })
                full_text_parts.append(text)

        # Extract tables as Markdown tables
        for table in doc.tables:
            table_md = self._table_to_markdown(table)
            if table_md:
                section_path = " > ".join(h for _, h in heading_stack)
                sections.append({
                    "heading_level": 0,
                    "heading_text": "",
                    "section_text": table_md,
                    "section_path": section_path,
                    "is_table": True,
                })
                full_text_parts.append(table_md)

        if not has_headings:
            warnings.append(
                "Document has no heading structure — applying paragraph-boundary splitting"
            )
            logger.warning(
                "DOCX has no headings",
                extra={"context": {"filename": request.filename}},
            )

        raw_text = "\n\n".join(full_text_parts)
        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.DOCX.value,
            "has_headings": has_headings,
            "paragraph_count": len([s for s in sections if s["heading_level"] == 0]),
            **request.user_metadata,
        }

        logger.info(
            "DOCX parse completed",
            extra={"context": {
                "filename": request.filename,
                "sections": len(sections),
                "has_headings": has_headings,
            }},
        )

        return ParsedDocument(
            source_type=SourceType.DOCX,
            raw_text=raw_text,
            structural_metadata=sections,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )

    @staticmethod
    def _table_to_markdown(table: Any) -> str:
        """Convert a python-docx Table object to a Markdown table string."""
        try:
            rows = []
            for i, row in enumerate(table.rows):
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    # Add separator row after header
                    rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            return "\n".join(rows)
        except Exception:
            return ""
