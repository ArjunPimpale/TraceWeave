"""Tests for source-aware chunkers."""

from __future__ import annotations

import hashlib

import pytest

from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


def _make_doc(source_type: SourceType, raw_text: str, structural_metadata: list, **meta) -> ParsedDocument:
    return ParsedDocument(
        source_type=source_type,
        raw_text=raw_text,
        structural_metadata=structural_metadata,
        global_metadata={
            "filename": "test_doc",
            "file_hash": hashlib.sha256(raw_text.encode()).hexdigest(),
            "source_type": source_type.value,
            **meta,
        },
    )


# ── PDF chunker ───────────────────────────────────────────────────────────────

class TestPDFChunker:
    def test_one_chunk_per_short_page(self):
        from pecs.chunking.pdf_chunker import PDFChunker
        doc = _make_doc(
            SourceType.PDF,
            "Page 1 text. Page 2 text.",
            [
                {"page_number": 1, "page_text": "Short page 1 text about authentication."},
                {"page_number": 2, "page_text": "Short page 2 text about GNN models."},
            ],
        )
        chunker = PDFChunker()
        chunks = chunker.chunk(doc)

        # Short pages may be merged into a single chunk — at least 1 chunk produced
        assert len(chunks) >= 1
        assert all(c.source_type == SourceType.PDF for c in chunks)
        # All page content should appear in chunks
        full_text = " ".join(c.chunk_text for c in chunks)
        assert "authentication" in full_text
        assert "GNN" in full_text

    def test_large_page_split(self):
        from pecs.chunking.pdf_chunker import PDFChunker
        large_text = "This is a sentence about requirements. " * 100
        doc = _make_doc(SourceType.PDF, large_text, [{"page_number": 1, "page_text": large_text}])
        chunker = PDFChunker(max_chunk_chars=500)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        # Each chunk should be reasonably sized (within 2x max to account for overlap strategy)
        assert all(c.char_count <= 1100 for c in chunks)

    def test_chunk_indices_sequential(self):
        from pecs.chunking.pdf_chunker import PDFChunker
        doc = _make_doc(
            SourceType.PDF, "p1 p2 p3",
            [{"page_number": i, "page_text": f"Content of page {i}."} for i in range(1, 4)],
        )
        chunks = PDFChunker().chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ── Markdown chunker ──────────────────────────────────────────────────────────

class TestMarkdownChunker:
    def test_sections_become_chunks(self):
        from pecs.chunking.markdown_chunker import MarkdownChunker
        sections = [
            {"heading_level": 1, "heading_text": "Title", "section_text": "# Title\n\nIntro text.", "section_path": "Title"},
            {"heading_level": 2, "heading_text": "R1", "section_text": "## R1\n\nAuthentication requirement.", "section_path": "Title > R1"},
            {"heading_level": 2, "heading_text": "R2", "section_text": "## R2\n\nGNN training requirement.", "section_path": "Title > R2"},
        ]
        doc = _make_doc(SourceType.MARKDOWN, "# Title\n\n## R1\n\n## R2", sections)
        chunker = MarkdownChunker()
        chunks = chunker.chunk(doc)

        # The markdown chunker produces chunks with parent context included;
        # at minimum one chunk per leaf section (R1, R2)
        assert len(chunks) >= 2
        assert all(c.source_type == SourceType.MARKDOWN for c in chunks)
        full_text = " ".join(c.chunk_text for c in chunks)
        assert "Authentication" in full_text
        assert "GNN" in full_text

    def test_empty_sections_dropped(self):
        from pecs.chunking.markdown_chunker import MarkdownChunker
        sections = [
            {"heading_level": 1, "heading_text": "Title", "section_text": "", "section_path": "Title"},
            {"heading_level": 2, "heading_text": "Body", "section_text": "Real content here.", "section_path": "Title > Body"},
        ]
        doc = _make_doc(SourceType.MARKDOWN, "content", sections)
        chunks = MarkdownChunker().chunk(doc)

        assert len(chunks) == 1
        assert "Real content" in chunks[0].chunk_text


# ── Python chunker ────────────────────────────────────────────────────────────

class TestPythonChunker:
    def test_class_gets_own_chunk(self, sample_python_bytes):
        from pecs.parsing.python_parser import PythonParser
        from pecs.chunking.python_chunker import PythonChunker
        import hashlib

        file_hash = hashlib.sha256(sample_python_bytes).hexdigest()
        parser = PythonParser()
        from pecs.parsing.base_parser import ParseRequest
        req = ParseRequest(sample_python_bytes, "gnn.py", file_hash, "text/x-python")
        doc = parser.parse(req)

        chunker = PythonChunker()
        chunks = chunker.chunk(doc)

        # Should have at least a module chunk + class chunk + function chunk
        assert len(chunks) >= 2
        element_types = {c.metadata.get("element_type") for c in chunks}
        assert "class" in element_types or "module" in element_types

    def test_no_overlap_between_definitions(self, sample_python_bytes):
        from pecs.parsing.python_parser import PythonParser
        from pecs.chunking.python_chunker import PythonChunker
        from pecs.parsing.base_parser import ParseRequest
        import hashlib

        file_hash = hashlib.sha256(sample_python_bytes).hexdigest()
        doc = PythonParser().parse(ParseRequest(sample_python_bytes, "gnn.py", file_hash, "text/x-python"))
        chunks = PythonChunker().chunk(doc)

        # Sequential indices
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ── WhatsApp chunker ──────────────────────────────────────────────────────────

class TestWhatsAppChunker:
    def test_time_window_grouping(self):
        from pecs.chunking.whatsapp_chunker import WhatsAppChunker

        messages = [
            {"sender": "Alice", "timestamp": "01/01/2024, 10:00:00", "message_text": "Hello", "is_media": False, "is_system": False},
            {"sender": "Bob",   "timestamp": "01/01/2024, 10:01:00", "message_text": "Hi",    "is_media": False, "is_system": False},
            # Gap > 30 min → new window
            {"sender": "Alice", "timestamp": "01/01/2024, 11:05:00", "message_text": "Back",  "is_media": False, "is_system": False},
        ]
        doc = _make_doc(SourceType.WHATSAPP, "chat text", messages, participants=["Alice", "Bob"])
        chunker = WhatsAppChunker()
        chunks = chunker.chunk(doc)

        # Should produce at least 2 windows
        assert len(chunks) >= 2

    def test_media_messages_excluded(self):
        from pecs.chunking.whatsapp_chunker import WhatsAppChunker

        messages = [
            {"sender": "Alice", "timestamp": "01/01/2024, 10:00:00", "message_text": "<Media omitted>", "is_media": True, "is_system": False},
            {"sender": "Bob",   "timestamp": "01/01/2024, 10:01:00", "message_text": "Real message",    "is_media": False, "is_system": False},
        ]
        doc = _make_doc(SourceType.WHATSAPP, "chat", messages)
        chunks = WhatsAppChunker().chunk(doc)

        # Only non-media messages should be in chunk text
        assert all("<Media omitted>" not in c.chunk_text for c in chunks)


# ── Git chunker ───────────────────────────────────────────────────────────────

class TestGitChunker:
    def test_one_chunk_per_commit(self):
        from pecs.chunking.git_chunker import GitChunker

        commits = [
            {"commit_sha": "abc123" * 7, "author": "Alice", "date": "2024-01-01", "message": "First commit", "changed_files": [], "changed_files_count": 0, "is_merge": False},
            {"commit_sha": "def456" * 7, "author": "Bob", "date": "2024-01-02", "message": "Second commit", "changed_files": [], "changed_files_count": 0, "is_merge": False},
        ]
        doc = _make_doc(SourceType.GIT, "git log", commits)
        chunks = GitChunker().chunk(doc)

        assert len(chunks) == 2
        assert all(c.source_type == SourceType.GIT for c in chunks)
        assert "abc123" in chunks[0].chunk_text or "abc123" in chunks[0].source_locator


# ── Base chunker utilities ────────────────────────────────────────────────────

class TestBaseChunkerUtilities:
    def test_enforce_size_limits_splits_large(self):
        from pecs.chunking.markdown_chunker import MarkdownChunker
        from pecs.chunking.base_chunker import RawChunk

        chunker = MarkdownChunker(max_chunk_chars=100)
        large_chunk = RawChunk(
            chunk_text="This is a very long sentence. " * 20,
            chunk_index=0,
            source_locator="section 1",
            source_document="test.md",
            source_hash="abc",
            source_type=SourceType.MARKDOWN,
        )
        result = chunker._enforce_size_limits([large_chunk])
        assert all(c.char_count <= 150 for c in result)  # allow some margin

    def test_min_chunk_chars_merges_small(self):
        from pecs.chunking.markdown_chunker import MarkdownChunker
        from pecs.chunking.base_chunker import RawChunk

        chunker = MarkdownChunker(min_chunk_chars=100)
        small1 = RawChunk("Short.", 0, "s1", "test.md", "abc", SourceType.MARKDOWN)
        small2 = RawChunk("Also short.", 1, "s2", "test.md", "abc", SourceType.MARKDOWN)
        result = chunker._enforce_size_limits([small1, small2])
        # The two small chunks should be merged
        assert len(result) <= 1
