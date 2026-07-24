"""Chunking module — source-aware chunkers."""

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.chunking.pdf_chunker import PDFChunker
from pecs.chunking.docx_chunker import DOCXChunker
from pecs.chunking.email_chunker import EmailChunker
from pecs.chunking.whatsapp_chunker import WhatsAppChunker
from pecs.chunking.git_chunker import GitChunker
from pecs.chunking.markdown_chunker import MarkdownChunker
from pecs.chunking.python_chunker import PythonChunker

__all__ = [
    "BaseChunker",
    "RawChunk",
    "PDFChunker",
    "DOCXChunker",
    "EmailChunker",
    "WhatsAppChunker",
    "GitChunker",
    "MarkdownChunker",
    "PythonChunker",
]
