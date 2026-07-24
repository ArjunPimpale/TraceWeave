"""
Python source file parser using the built-in ast module.

Extracts module docstring, class definitions (with methods), standalone
functions (signature + docstring + body), and import statements.
Preserves line numbers for provenance.
Falls back to plain text if the file has syntax errors.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)


class PythonParser(BaseParser):
    """
    Parser for Python source files.

    Uses Python's built-in ast module to extract structured elements.
    Falls back to treating the file as plain text if ast.parse fails.

    Structural metadata: list of dicts:
        {
            "element_type": str,       # "module", "class", "function", "import_block"
            "element_name": str,
            "docstring": str,
            "body": str,               # Source code text of this element
            "line_start": int,
            "line_end": int,
        }
    """

    SUPPORTED_MIME_TYPES = ("text/x-python", "text/plain", "application/x-python-code")
    SUPPORTED_EXTENSIONS = ("py", "pyw")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        logger.debug(
            "Python parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        source_text = self._decode(request.file_bytes)

        try:
            tree = ast.parse(source_text, filename=request.filename)
        except SyntaxError as exc:
            warnings.append(
                f"SyntaxError in {request.filename!r}: {exc} — treating as plain text"
            )
            logger.warning(
                "Python file has syntax errors — plain text fallback",
                extra={"context": {"filename": request.filename, "error": str(exc)}},
            )
            return self._plain_text_fallback(request, source_text, warnings)

        source_lines = source_text.splitlines()
        elements: list[dict[str, Any]] = []

        # ── Module docstring ───────────────────────────────────────────────
        module_docstring = ast.get_docstring(tree) or ""
        if module_docstring:
            elements.append({
                "element_type": "module",
                "element_name": request.filename,
                "docstring": module_docstring,
                "body": module_docstring,
                "line_start": 1,
                "line_end": module_docstring.count("\n") + 1,
            })

        # ── Import block ───────────────────────────────────────────────────
        import_lines: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and hasattr(node, "lineno"):
                line = source_lines[node.lineno - 1].strip()
                import_lines.append(line)

        if import_lines:
            elements.append({
                "element_type": "import_block",
                "element_name": f"{request.filename}:imports",
                "docstring": "",
                "body": "\n".join(import_lines),
                "line_start": 1,
                "line_end": len(import_lines),
            })

        # ── Top-level classes and functions ───────────────────────────────
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                elements.append(self._extract_class(node, source_lines, request.filename))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                elements.append(self._extract_function(node, source_lines, request.filename))

        raw_text = source_text
        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.PYTHON.value,
            "element_count": len(elements),
            "line_count": len(source_lines),
            **request.user_metadata,
        }

        logger.info(
            "Python parse completed",
            extra={"context": {
                "filename": request.filename,
                "elements": len(elements),
                "lines": len(source_lines),
            }},
        )

        return ParsedDocument(
            source_type=SourceType.PYTHON,
            raw_text=raw_text,
            structural_metadata=elements,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        source_lines: list[str],
        filename: str,
    ) -> dict[str, Any]:
        """Extract a class definition with its methods."""
        docstring = ast.get_docstring(node) or ""
        body = self._get_source_segment(source_lines, node.lineno, node.end_lineno or node.lineno)

        return {
            "element_type": "class",
            "element_name": node.name,
            "docstring": docstring,
            "body": f"# File: {filename}\n{body}",
            "line_start": node.lineno,
            "line_end": node.end_lineno or node.lineno,
        }

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        filename: str,
    ) -> dict[str, Any]:
        """Extract a standalone function definition."""
        docstring = ast.get_docstring(node) or ""
        body = self._get_source_segment(source_lines, node.lineno, node.end_lineno or node.lineno)

        return {
            "element_type": "function",
            "element_name": node.name,
            "docstring": docstring,
            "body": f"# File: {filename}\n{body}",
            "line_start": node.lineno,
            "line_end": node.end_lineno or node.lineno,
        }

    @staticmethod
    def _get_source_segment(
        source_lines: list[str], start: int, end: int
    ) -> str:
        """Extract source lines from start to end (1-indexed, inclusive)."""
        lines = source_lines[start - 1 : end]
        return textwrap.dedent("\n".join(lines)).strip()

    @staticmethod
    def _decode(data: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _plain_text_fallback(
        request: ParseRequest, text: str, warnings: list[str]
    ) -> ParsedDocument:
        """Return the file as a single plain-text element when AST parsing fails."""
        return ParsedDocument(
            source_type=SourceType.PYTHON,
            raw_text=text,
            structural_metadata=[{
                "element_type": "plain_text",
                "element_name": request.filename,
                "docstring": "",
                "body": text,
                "line_start": 1,
                "line_end": text.count("\n") + 1,
            }],
            global_metadata={
                "filename": request.filename,
                "file_hash": request.file_hash,
                "source_type": SourceType.PYTHON.value,
                "parse_failed": True,
            },
            parse_warnings=warnings,
        )
