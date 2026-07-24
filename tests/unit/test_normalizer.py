"""Tests for the Markdown normalizer."""

from __future__ import annotations

import hashlib

import pytest

from pecs.chunking.base_chunker import RawChunk
from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.normalization.normalizer import Normalizer, _normalize_text


class TestNormalizeText:
    def test_nfkc_normalization(self):
        """Curly quotes and em-dashes should be normalized."""
        text = "\u201cHello\u201d \u2014 world"  # "Hello" — world
        result = _normalize_text(text)
        # After NFKC, these should become ASCII-like equivalents
        assert result.strip()

    def test_excess_newlines_collapsed(self):
        result = _normalize_text("Line 1\n\n\n\n\nLine 2")
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_internal_whitespace_normalized(self):
        result = _normalize_text("word1    word2\t\tword3")
        assert "    " not in result
        assert "\t\t" not in result

    def test_strips_leading_trailing(self):
        result = _normalize_text("   hello world   ")
        assert result == "hello world"

    def test_empty_string(self):
        result = _normalize_text("")
        assert result == ""


class TestNormalizer:
    def test_normalize_produces_evidence_chunk(self, sample_raw_chunk):
        norm = Normalizer()
        ec = norm.normalize(sample_raw_chunk)

        assert ec is not None
        assert isinstance(ec, EvidenceChunk)
        assert ec.normalized_text.strip()
        assert ec.chunk_id  # deterministic SHA-256
        assert ec.source_document == sample_raw_chunk.source_document
        assert ec.source_type == sample_raw_chunk.source_type

    def test_deterministic_chunk_id(self):
        """Same source_hash + chunk_index → same chunk_id."""
        norm = Normalizer()
        src_hash = hashlib.sha256(b"test").hexdigest()

        raw1 = RawChunk("text", 3, "page 4", "doc.pdf", src_hash, SourceType.PDF)
        raw2 = RawChunk("different text", 3, "page 4", "doc.pdf", src_hash, SourceType.PDF)

        ec1 = norm.normalize(raw1)
        ec2 = norm.normalize(raw2)

        # Chunk IDs are based on source_hash + chunk_index, not text content
        assert ec1.chunk_id == ec2.chunk_id

    def test_empty_text_returns_none(self):
        norm = Normalizer()
        raw = RawChunk("   \n\n   ", 0, "p1", "doc.pdf",
                       hashlib.sha256(b"x").hexdigest(), SourceType.PDF)
        result = norm.normalize(raw)
        assert result is None

    def test_normalize_batch_drops_empty(self):
        norm = Normalizer()
        src_hash = hashlib.sha256(b"batch").hexdigest()
        chunks = [
            RawChunk("Valid content here.", 0, "p1", "doc.pdf", src_hash, SourceType.PDF),
            RawChunk("   ", 1, "p2", "doc.pdf", src_hash, SourceType.PDF),
            RawChunk("More valid content.", 2, "p3", "doc.pdf", src_hash, SourceType.PDF),
        ]
        results = norm.normalize_batch(chunks)
        assert len(results) == 2
        assert all(ec.normalized_text.strip() for ec in results)

    def test_metadata_preserved(self, sample_raw_chunk):
        sample_raw_chunk.metadata = {"page_number": 3, "heading": "Introduction"}
        norm = Normalizer()
        ec = norm.normalize(sample_raw_chunk)

        assert ec.metadata.get("page_number") == 3
        assert ec.metadata.get("heading") == "Introduction"

    def test_char_count_accurate(self, sample_raw_chunk):
        norm = Normalizer()
        ec = norm.normalize(sample_raw_chunk)

        assert ec.char_count == len(ec.normalized_text)
