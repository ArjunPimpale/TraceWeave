"""Parsing module — source-specific parser adapters."""

from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument
from pecs.parsing.pdf_parser import PDFParser
from pecs.parsing.docx_parser import DOCXParser
from pecs.parsing.email_parser import EmailParser
from pecs.parsing.whatsapp_parser import WhatsAppParser
from pecs.parsing.git_parser import GitParser
from pecs.parsing.markdown_parser import MarkdownParser
from pecs.parsing.python_parser import PythonParser

__all__ = [
    "BaseParser",
    "ParseRequest",
    "ParsedDocument",
    "PDFParser",
    "DOCXParser",
    "EmailParser",
    "WhatsAppParser",
    "GitParser",
    "MarkdownParser",
    "PythonParser",
]
