"""Characterization tests for ingestion orchestration failure handling."""

from __future__ import annotations

from unittest.mock import MagicMock

from pecs.ingestion.ingestor import Ingestor
from pecs.models.evidence_chunk import SourceType


def test_parse_failure_is_logged_and_returned(monkeypatch, evidence_repo) -> None:
    ingestor = Ingestor(evidence_repo=evidence_repo)
    parser = MagicMock()
    parser.parse.side_effect = RuntimeError("parser failed")
    ingestor._parsers[SourceType.MARKDOWN] = parser
    monkeypatch.setattr(
        "pecs.ingestion.ingestor.detect_mime_type", lambda _bytes, _filename: "text/markdown"
    )

    result = ingestor.ingest_file(b"# Requirement R1", "requirements.md")

    assert result.success is False
    assert result.error == "Parsing failed: parser failed"
    log = evidence_repo.get_ingestion_log()
    assert len(log) == 1
    assert log[0]["status"] == "failed"
    assert log[0]["error_message"] == "parser failed"
