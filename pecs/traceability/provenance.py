"""Immutable source snapshots for traceability citations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pecs.models.evidence_chunk import EvidenceChunk


@dataclass(frozen=True)
class ChunkSnapshot:
    snapshot_id: str
    chunk_id: str
    source_hash: str
    source_document: str
    source_type: str
    source_locator: str
    chunk_index: int
    normalized_text: str
    metadata: dict[str, Any]
    origin: str = "pipeline"


def from_chunk(chunk: EvidenceChunk, origin: str = "pipeline") -> ChunkSnapshot:
    metadata = {str(k).removeprefix("meta_"): v for k, v in chunk.metadata.items()}
    body = {
        "chunk_id": chunk.chunk_id, "source_hash": chunk.source_hash,
        "source_document": chunk.source_document, "source_type": chunk.source_type.value,
        "source_locator": chunk.source_locator, "chunk_index": chunk.chunk_index,
        "normalized_text": chunk.normalized_text, "metadata": metadata,
    }
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)
    return ChunkSnapshot(snapshot_id=hashlib.sha256(canonical.encode()).hexdigest(), origin=origin, **body)
