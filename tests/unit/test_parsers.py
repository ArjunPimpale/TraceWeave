"""Tests for all source-specific parsers."""

from __future__ import annotations

import pytest
from pecs.parsing.base_parser import ParseRequest
from pecs.models.evidence_chunk import SourceType


def _make_request(data: bytes, filename: str, mime: str = "text/plain") -> ParseRequest:
    import hashlib
    return ParseRequest(
        file_bytes=data,
        filename=filename,
        file_hash=hashlib.sha256(data).hexdigest(),
        mime_type=mime,
    )


# ── Markdown parser ───────────────────────────────────────────────────────────

class TestMarkdownParser:
    def test_parses_headings(self, sample_md_bytes):
        from pecs.parsing.markdown_parser import MarkdownParser
        parser = MarkdownParser()
        req = _make_request(sample_md_bytes, "req.md", "text/markdown")
        doc = parser.parse(req)

        assert doc.source_type == SourceType.MARKDOWN
        assert len(doc.structural_metadata) > 0
        assert doc.global_metadata["has_headings"] is True

    def test_front_matter_extracted(self):
        from pecs.parsing.markdown_parser import MarkdownParser
        md = b"---\ntitle: Test\nauthor: Alice\n---\n# Body\nHello world.\n"
        parser = MarkdownParser()
        req = _make_request(md, "doc.md")
        doc = parser.parse(req)

        assert doc.global_metadata["front_matter"].get("title") == "Test"

    def test_no_headings_warning(self):
        from pecs.parsing.markdown_parser import MarkdownParser
        md = b"Just plain text without any headings here."
        parser = MarkdownParser()
        req = _make_request(md, "plain.md")
        doc = parser.parse(req)

        assert len(doc.parse_warnings) > 0
        assert any("heading" in w.lower() for w in doc.parse_warnings)


# ── Python parser ─────────────────────────────────────────────────────────────

class TestPythonParser:
    def test_parses_classes_and_functions(self, sample_python_bytes):
        from pecs.parsing.python_parser import PythonParser
        parser = PythonParser()
        req = _make_request(sample_python_bytes, "gnn.py", "text/x-python")
        doc = parser.parse(req)

        assert doc.source_type == SourceType.PYTHON
        element_types = {e["element_type"] for e in doc.structural_metadata}
        assert "class" in element_types
        assert "function" in element_types

    def test_syntax_error_fallback(self):
        from pecs.parsing.python_parser import PythonParser
        bad_py = b"def broken(\n  # unclosed"
        parser = PythonParser()
        req = _make_request(bad_py, "broken.py")
        doc = parser.parse(req)

        assert doc.source_type == SourceType.PYTHON
        assert len(doc.parse_warnings) > 0
        assert any("Syntax" in w for w in doc.parse_warnings)

    def test_module_docstring_extracted(self, sample_python_bytes):
        from pecs.parsing.python_parser import PythonParser
        parser = PythonParser()
        req = _make_request(sample_python_bytes, "gnn.py")
        doc = parser.parse(req)

        module_elements = [e for e in doc.structural_metadata if e["element_type"] == "module"]
        assert module_elements
        assert "GNN training" in module_elements[0]["docstring"]


# ── Email parser ──────────────────────────────────────────────────────────────

class TestEmailParser:
    def test_parses_basic_email(self):
        from pecs.parsing.email_parser import EmailParser
        email = b"""From: alice@example.com
Date: Mon, 1 Jan 2024 10:00:00 +0000
Subject: Project update

Hi team, we have completed the GNN implementation.
"""
        parser = EmailParser()
        req = _make_request(email, "update.eml", "message/rfc822")
        doc = parser.parse(req)

        assert doc.source_type == SourceType.EMAIL
        assert len(doc.structural_metadata) >= 1
        msg = doc.structural_metadata[0]
        assert "alice@example.com" in msg["sender"]
        assert "Project update" in msg["subject"]

    def test_body_extracted_without_headers(self):
        from pecs.parsing.email_parser import EmailParser
        email = b"""From: bob@example.com
Date: 2024-01-02
Subject: Test

This is the body text.
It spans multiple lines.
"""
        parser = EmailParser()
        req = _make_request(email, "test.eml")
        doc = parser.parse(req)

        body = doc.structural_metadata[0]["body"]
        assert "body text" in body
        assert "From:" not in body


# ── WhatsApp parser ───────────────────────────────────────────────────────────

class TestWhatsAppParser:
    def test_parses_whatsapp_export(self):
        from pecs.parsing.whatsapp_parser import WhatsAppParser
        chat = b"""[01/01/2024, 10:00:00] Alice: We need to implement the GNN module.
[01/01/2024, 10:01:00] Bob: Agreed, I'll start on the training pipeline.
[01/01/2024, 10:02:00] Alice: Make sure it handles both supervised and unsupervised modes.
"""
        parser = WhatsAppParser()
        req = _make_request(chat, "chat.txt")
        doc = parser.parse(req)

        assert doc.source_type == SourceType.WHATSAPP
        assert len(doc.structural_metadata) == 3
        assert doc.structural_metadata[0]["sender"] == "Alice"

    def test_media_messages_flagged(self):
        from pecs.parsing.whatsapp_parser import WhatsAppParser
        chat = b"[01/01/2024, 10:00:00] Alice: <Media omitted>\n"
        parser = WhatsAppParser()
        req = _make_request(chat, "chat.txt")
        doc = parser.parse(req)

        assert doc.structural_metadata[0]["is_media"] is True


# ── Git parser ────────────────────────────────────────────────────────────────

class TestGitParser:
    def test_parses_standard_git_log(self):
        from pecs.parsing.git_parser import GitParser
        log = b"""commit abc123def456abc123def456abc123def456abc1
Author: Alice <alice@example.com>
Date:   Mon Jan 1 10:00:00 2024 +0000

    Implement GNN training pipeline (R2)

commit def456abc123def456abc123def456abc123def4
Author: Bob <bob@example.com>
Date:   Tue Jan 2 11:00:00 2024 +0000

    Add REST API endpoints for inference
"""
        parser = GitParser()
        req = _make_request(log, "commits.log")
        doc = parser.parse(req)

        assert doc.source_type == SourceType.GIT
        assert len(doc.structural_metadata) == 2
        commit = doc.structural_metadata[0]
        assert "Alice" in commit["author"]
        assert "GNN training" in commit["message"]
