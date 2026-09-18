"""Characterization tests for Chroma record access and reconstruction."""

from __future__ import annotations

from unittest.mock import MagicMock

from pecs.models.evidence_chunk import SourceType
from pecs.vectorstore.chroma_store import ChromaStore
from pecs.vectorstore.chunk_codec import reconstruct_evidence_chunk


def test_get_all_records_returns_raw_documents_and_metadata(monkeypatch) -> None:
    collection = MagicMock()
    expected = {
        "ids": ["chunk-1"],
        "documents": ["Requirement R1"],
        "metadatas": [{"source_document": "requirements.md"}],
    }
    collection.get.return_value = expected
    store = ChromaStore(persist_directory="unused")
    monkeypatch.setattr(store, "_get_collection", lambda: collection)

    assert store.get_all_records() == expected
    collection.get.assert_called_once_with(include=["documents", "metadatas"])


def test_reconstruction_preserves_extra_metadata_only_when_requested() -> None:
    metadata = {
        "source_document": "requirements.md",
        "source_hash": "abc",
        "source_type": "PDF",
        "chunk_index": 3,
        "source_locator": "page 4",
        "char_count": 12,
        "created_at": "2026-01-01T00:00:00",
        "meta_heading": "Security",
    }

    retrieved = reconstruct_evidence_chunk("chunk-1", "Requirement R1", metadata)
    extracted = reconstruct_evidence_chunk(
        "chunk-1", "Requirement R1", metadata, include_extra_metadata=False
    )

    assert retrieved.source_type is SourceType.PDF
    assert retrieved.metadata == {"meta_heading": "Security"}
    assert extracted.metadata == {}


def test_reconstruction_defaults_invalid_source_type_to_markdown() -> None:
    chunk = reconstruct_evidence_chunk(
        "chunk-1", "text", {"source_type": "UNKNOWN"}
    )

    assert chunk.source_type is SourceType.MARKDOWN
