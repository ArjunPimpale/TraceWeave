"""
Markdown file parser.

Parses ATX headings (#, ##, etc.) to build a section hierarchy.
Handles front-matter (YAML between --- delimiters) as metadata.
Preserves code blocks, lists, and emphasis.
"""

from __future__ import annotations

import re
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)

# Match ATX headings
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Match YAML front-matter between --- delimiters
_FRONT_MATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MarkdownParser(BaseParser):
    """
    Parser for Markdown (.md, .markdown) files.

    Structural metadata: list of dicts:
        {
            "heading_level": int,     # 0 = body text
            "heading_text": str,
            "section_text": str,
            "section_path": str,      # Breadcrumb heading path
        }

    Front-matter (YAML between --- delimiters) is extracted as metadata
    and not included in chunk text.
    """

    SUPPORTED_MIME_TYPES = ("text/markdown", "text/x-markdown", "text/plain")
    SUPPORTED_EXTENSIONS = ("md", "markdown", "mdx")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        logger.debug(
            "Markdown parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        text = self._decode(request.file_bytes)

        # Extract and remove front-matter
        front_matter: dict[str, Any] = {}
        fm_match = _FRONT_MATTER_PATTERN.match(text)
        if fm_match:
            front_matter = self._parse_front_matter(fm_match.group(1))
            text = text[fm_match.end():]

        sections, has_headings = self._extract_sections(text)

        if not has_headings:
            warnings.append("Markdown file has no headings — treating as plain text")

        raw_text = text.strip()
        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.MARKDOWN.value,
            "has_headings": has_headings,
            "front_matter": front_matter,
            **request.user_metadata,
        }

        logger.info(
            "Markdown parse completed",
            extra={"context": {
                "filename": request.filename,
                "sections": len(sections),
                "has_headings": has_headings,
            }},
        )

        return ParsedDocument(
            source_type=SourceType.MARKDOWN,
            raw_text=raw_text,
            structural_metadata=sections,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )

    @staticmethod
    def _decode(data: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_sections(text: str) -> tuple[list[dict[str, Any]], bool]:
        """
        Split Markdown into sections based on ATX headings.

        Returns:
            (sections list, has_headings bool)
        """
        sections: list[dict[str, Any]] = []
        has_headings = False

        # Find all heading positions
        headings = list(_HEADING_PATTERN.finditer(text))
        if not headings:
            # No headings — entire text is one section
            return [{"heading_level": 0, "heading_text": "", "section_text": text.strip(), "section_path": ""}], False

        has_headings = True
        heading_stack: list[tuple[int, str]] = []

        # Text before the first heading
        if headings[0].start() > 0:
            preamble = text[:headings[0].start()].strip()
            if preamble:
                sections.append({
                    "heading_level": 0,
                    "heading_text": "",
                    "section_text": preamble,
                    "section_path": "",
                })

        for i, match in enumerate(headings):
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Update heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            section_path = " > ".join(h for _, h in heading_stack)

            # Body is text until next heading (or end of file)
            body_start = match.end()
            body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[body_start:body_end].strip()

            sections.append({
                "heading_level": level,
                "heading_text": heading_text,
                "section_text": f"{'#' * level} {heading_text}\n\n{body}".strip(),
                "section_path": section_path,
            })

        return sections, has_headings

    @staticmethod
    def _parse_front_matter(yaml_text: str) -> dict[str, Any]:
        """Parse YAML front-matter into a dict. Fails gracefully."""
        try:
            import yaml
            result = yaml.safe_load(yaml_text)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
