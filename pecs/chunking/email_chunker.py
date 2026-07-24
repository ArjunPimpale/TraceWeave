"""Email chunker — one message per chunk, with overlap_messages context."""

from __future__ import annotations

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.config import settings
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


class EmailChunker(BaseChunker):
    """
    Message-first chunking for email exports.

    Strategy:
    1. Each individual email message = one chunk.
    2. Each chunk is prefixed with "From: ... | Date: ... | Subject: ..."
    3. If a message exceeds max_chunk_chars, split at paragraph boundaries.
    4. Apply overlap_messages (default: 2 previous messages as context).

    Metadata: sender, timestamp, subject.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.overlap_messages = settings.OVERLAP_MESSAGES

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        messages = [
            m for m in document.structural_metadata
            if isinstance(m, dict) and m.get("body", "").strip()
        ]

        if not messages:
            return []

        raw_chunks: list[RawChunk] = []
        for idx, msg in enumerate(messages):
            header = (
                f"From: {msg.get('sender', 'Unknown')} | "
                f"Date: {msg.get('timestamp', '')} | "
                f"Subject: {msg.get('subject', '(no subject)')}"
            )
            body = msg.get("body", "").strip()
            chunk_text = f"{header}\n\n{body}"

            chunk_meta = {
                **base_meta,
                "sender": msg.get("sender", ""),
                "timestamp": msg.get("timestamp", ""),
                "subject": msg.get("subject", ""),
                "message_index": idx,
            }
            source_locator = f"Message {idx + 1} from {msg.get('sender', 'Unknown')}"

            raw_chunks.append(RawChunk(
                chunk_text=chunk_text,
                chunk_index=idx,
                source_locator=source_locator,
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.EMAIL,
                metadata=chunk_meta,
            ))

        # Apply message-level overlap: prepend the last N messages as context
        chunks_with_overlap: list[RawChunk] = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and self.overlap_messages > 0:
                overlap_start = max(0, i - self.overlap_messages)
                overlap_texts = [
                    raw_chunks[j].chunk_text for j in range(overlap_start, i)
                ]
                prefix = "\n\n[Previous context:]\n" + "\n---\n".join(overlap_texts) + "\n[End context]\n\n"
                chunk = RawChunk(
                    chunk_text=(prefix + chunk.chunk_text).strip(),
                    chunk_index=chunk.chunk_index,
                    source_locator=chunk.source_locator,
                    source_document=chunk.source_document,
                    source_hash=chunk.source_hash,
                    source_type=chunk.source_type,
                    metadata=chunk.metadata,
                )
            chunks_with_overlap.append(chunk)

        chunks_with_overlap = self._enforce_size_limits(chunks_with_overlap)

        for i, chunk in enumerate(chunks_with_overlap):
            chunk.chunk_index = i

        return chunks_with_overlap
