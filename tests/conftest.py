"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from pathlib import Path

import pytest

from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.models.extraction_result import ExtractionResult, EntityType
from pecs.chunking.base_chunker import RawChunk


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    from pecs.store.database import Database
    db = Database(db_path=tmp_path / "test.db")
    db.initialize_schema()
    yield db
    db.close()


@pytest.fixture
def evidence_repo(tmp_db):
    """EvidenceRepo backed by a temporary test database."""
    from pecs.store.evidence_repo import EvidenceRepo
    return EvidenceRepo(db=tmp_db)


@pytest.fixture
def correlation_repo(tmp_db):
    """CorrelationRepo backed by a temporary test database."""
    from pecs.store.correlation_repo import CorrelationRepo
    return CorrelationRepo(db=tmp_db)


@pytest.fixture
def sample_md_bytes():
    """Sample Markdown document bytes."""
    return b"""# Requirements

## R1 - Authentication
The system must implement JWT-based user authentication.

## R2 - GNN Training
The system must implement a GNN training pipeline for anomaly detection.
"""


@pytest.fixture
def sample_python_bytes():
    """Sample Python source file bytes."""
    return b'''"""GNN training module."""

import torch

class GNNModel:
    """Graph Neural Network model for anomaly detection."""

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def forward(self, x):
        """Forward pass."""
        return x


def train_model(model, data, epochs=100):
    """Train the GNN model. Implements Requirement R2."""
    for epoch in range(epochs):
        pass
'''


@pytest.fixture
def sample_chunk():
    """A sample EvidenceChunk for testing."""
    source_hash = hashlib.sha256(b"test").hexdigest()
    chunk_id = EvidenceChunk.make_chunk_id(source_hash, 0)
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_document="test.md",
        source_hash=source_hash,
        source_type=SourceType.MARKDOWN,
        chunk_index=0,
        source_locator="section Introduction",
        normalized_text="The system must implement JWT-based user authentication for Requirement R1.",
        char_count=73,
        metadata={"heading_text": "Introduction"},
    )


@pytest.fixture
def sample_extraction_result(sample_chunk):
    """A sample ExtractionResult for testing."""
    return ExtractionResult(
        entity_type=EntityType.REQUIREMENT,
        entity_id="R1-auth",
        text="The system must implement JWT-based user authentication.",
        linked_requirement=None,
        source_document=sample_chunk.source_document,
        chunk_id=sample_chunk.chunk_id,
        author=None,
        timestamp=None,
    )


@pytest.fixture
def sample_raw_chunk():
    """A sample RawChunk before normalization."""
    return RawChunk(
        chunk_text="The system must implement **JWT** authentication for all users.",
        chunk_index=0,
        source_locator="page 1",
        source_document="requirements.pdf",
        source_hash=hashlib.sha256(b"pdf_bytes").hexdigest(),
        source_type=SourceType.PDF,
    )
