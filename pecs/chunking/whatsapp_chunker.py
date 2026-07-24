"""
WhatsApp chunker — time-window-based chunking.

Groups messages that are temporally close (gap < WHATSAPP_TIME_GAP_MINUTES)
into conversation windows. This captures topic boundaries more naturally
than fixed message counts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.config import settings
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument

# Supported datetime formats for parsing WhatsApp timestamps
_TIMESTAMP_FORMATS = [
    "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%Y, %H:%M",
    "%m/%d/%Y, %H:%M:%S",
    "%m/%d/%Y, %I:%M %p",
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%y, %H:%M",
    "%m/%d/%y, %I:%M %p",
    "%d/%m/%y, %I:%M %p",
]


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Try to parse a WhatsApp timestamp string into a datetime."""
    ts_str = ts_str.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


class WhatsAppChunker(BaseChunker):
    """
    Time-window chunking for WhatsApp exports.

    Strategy:
    1. Group messages into conversation windows using time-gap heuristic.
    2. If gap between consecutive messages > time_gap_minutes → new window.
    3. Each window = one chunk, concatenating all messages with attribution.
    4. If window > max_chunk_chars, split at message boundaries.
    5. Apply overlap_messages between consecutive windows.

    Metadata: window_start_time, window_end_time, participants, message_count.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.time_gap_minutes = settings.WHATSAPP_TIME_GAP_MINUTES
        self.overlap_messages = settings.OVERLAP_MESSAGES

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        # Filter: only textual, non-system messages
        messages = [
            m for m in document.structural_metadata
            if not m.get("is_system", False) and not m.get("is_media", False)
            and m.get("message_text", "").strip()
        ]

        if not messages:
            return []

        # Group messages into time windows
        windows = self._group_into_windows(messages)

        raw_chunks: list[RawChunk] = []
        for idx, window in enumerate(windows):
            text = self._format_window(window["messages"])
            participants = sorted({m["sender"] for m in window["messages"]})
            chunk_meta = {
                **base_meta,
                "window_start_time": window["start_time"],
                "window_end_time": window["end_time"],
                "message_count": len(window["messages"]),
                "participants": ", ".join(participants),
            }
            source_locator = f"{window['start_time']} to {window['end_time']}"
            raw_chunks.append(RawChunk(
                chunk_text=text,
                chunk_index=idx,
                source_locator=source_locator,
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.WHATSAPP,
                metadata=chunk_meta,
            ))

        # Apply message-level overlap
        chunks_with_overlap: list[RawChunk] = []
        all_msgs_flat: list[dict] = []
        for window in windows:
            all_msgs_flat.extend(window["messages"])

        window_msg_counts = [len(w["messages"]) for w in windows]
        msg_offset = 0

        for i, chunk in enumerate(raw_chunks):
            if i > 0 and self.overlap_messages > 0:
                # Prepend last N messages from the previous window
                start = max(0, msg_offset - self.overlap_messages)
                overlap_msgs = all_msgs_flat[start:msg_offset]
                overlap_text = self._format_window(overlap_msgs)
                if overlap_text:
                    chunk = RawChunk(
                        chunk_text=f"[Context:]\n{overlap_text}\n\n[Current window:]\n{chunk.chunk_text}",
                        chunk_index=chunk.chunk_index,
                        source_locator=chunk.source_locator,
                        source_document=chunk.source_document,
                        source_hash=chunk.source_hash,
                        source_type=chunk.source_type,
                        metadata=chunk.metadata,
                    )
            msg_offset += window_msg_counts[i]
            chunks_with_overlap.append(chunk)

        chunks_with_overlap = self._enforce_size_limits(chunks_with_overlap)

        for i, chunk in enumerate(chunks_with_overlap):
            chunk.chunk_index = i

        return chunks_with_overlap

    def _group_into_windows(
        self, messages: list[dict]
    ) -> list[dict[str, Any]]:
        """Group messages into conversation windows based on time gaps."""
        windows: list[dict[str, Any]] = []
        if not messages:
            return windows

        current_window_msgs = [messages[0]]
        current_dt = _parse_timestamp(messages[0].get("timestamp", ""))

        for msg in messages[1:]:
            msg_dt = _parse_timestamp(msg.get("timestamp", ""))

            if msg_dt and current_dt:
                gap = (msg_dt - current_dt).total_seconds() / 60
                if gap > self.time_gap_minutes:
                    # New window
                    windows.append(self._make_window(current_window_msgs))
                    current_window_msgs = []
                current_dt = msg_dt

            current_window_msgs.append(msg)

        if current_window_msgs:
            windows.append(self._make_window(current_window_msgs))

        return windows

    @staticmethod
    def _make_window(messages: list[dict]) -> dict[str, Any]:
        start_time = messages[0].get("timestamp", "") if messages else ""
        end_time = messages[-1].get("timestamp", "") if messages else ""
        return {
            "messages": messages,
            "start_time": start_time,
            "end_time": end_time,
        }

    @staticmethod
    def _format_window(messages: list[dict]) -> str:
        """Format a list of messages into a single text block."""
        lines = [
            f"[{m.get('timestamp', '')}] {m.get('sender', 'Unknown')}: {m.get('message_text', '')}"
            for m in messages
        ]
        return "\n".join(lines).strip()
