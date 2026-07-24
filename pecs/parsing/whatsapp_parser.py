"""
WhatsApp chat export parser.

WhatsApp exports follow the format: [DD/MM/YY, HH:MM:SS] Sender: Message
or variations thereof. This parser handles multi-line messages and common
WhatsApp-specific content (media placeholders, system messages).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)

# Supported WhatsApp timestamp formats (ordered most → least specific)
_TIMESTAMP_PATTERNS = [
    # [DD/MM/YYYY, HH:MM:SS] (24h or 12h with AM/PM)
    re.compile(
        r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s+(.+?):\s+(.*)$"
    ),
    # DD/MM/YYYY, HH:MM - Sender: Message (no brackets)
    re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\s+-\s+(.+?):\s+(.*)$"
    ),
    # MM/DD/YY, HH:MM AM/PM - Sender: Message
    re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M)\s+-\s+(.+?):\s+(.*)$"
    ),
]

# System messages (no sender colon) that should be excluded
_SYSTEM_MSG_PATTERNS = [
    re.compile(r"(?:added|removed|left|joined|created|changed|Messages and calls)", re.IGNORECASE),
    re.compile(r"<Media omitted>", re.IGNORECASE),
    re.compile(r"This message was deleted", re.IGNORECASE),
    re.compile(r"You deleted this message", re.IGNORECASE),
    re.compile(r"Security code changed", re.IGNORECASE),
]


class WhatsAppParser(BaseParser):
    """
    Parser for WhatsApp exported chat text files.

    Structural metadata: list of dicts:
        {
            "sender": str,
            "timestamp": str,       # Raw timestamp string from export
            "message_text": str,    # Cleaned message text
            "is_media": bool,       # True if message is "<Media omitted>"
            "is_system": bool,      # True if a system message
        }
    """

    SUPPORTED_MIME_TYPES = ("text/plain",)
    SUPPORTED_EXTENSIONS = ("txt",)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        logger.debug(
            "WhatsApp parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        text = self._decode(request.file_bytes)
        lines = text.splitlines()

        messages: list[dict[str, Any]] = []
        current_msg: dict[str, Any] | None = None

        for line_num, line in enumerate(lines, start=1):
            parsed = self._try_parse_line(line)

            if parsed is not None:
                # Save the previous message
                if current_msg is not None:
                    messages.append(current_msg)
                current_msg = parsed
            elif current_msg is not None:
                # Continuation of the previous message (multi-line)
                current_msg["message_text"] += "\n" + line
            else:
                # Line before any timestamp match (e.g., WhatsApp header)
                if line.strip() and not line.startswith("\ufeff"):
                    warnings.append(f"Line {line_num} before first message: {line[:80]!r}")

        if current_msg is not None:
            messages.append(current_msg)

        if not messages:
            warnings.append(
                "No WhatsApp messages were parsed — file may not be a WhatsApp export"
            )
            logger.warning(
                "WhatsApp format unrecognized",
                extra={"context": {"filename": request.filename, "line_count": len(lines)}},
            )

        # Build full text (excluding system/media messages from raw_text)
        text_messages = [m for m in messages if not m["is_system"]]
        full_text_parts = [
            f"[{m['timestamp']}] {m['sender']}: {m['message_text']}"
            for m in text_messages
            if not m["is_media"]
        ]
        raw_text = "\n".join(full_text_parts)

        participants = sorted({m["sender"] for m in messages if not m["is_system"] and m["sender"] != "Unknown"})

        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.WHATSAPP.value,
            "message_count": len(messages),
            "text_message_count": len(text_messages),
            "participants": participants,
            **request.user_metadata,
        }

        logger.info(
            "WhatsApp parse completed",
            extra={"context": {
                "filename": request.filename,
                "messages": len(messages),
                "participants": len(participants),
            }},
        )

        return ParsedDocument(
            source_type=SourceType.WHATSAPP,
            raw_text=raw_text,
            structural_metadata=messages,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )

    @staticmethod
    def _decode(data: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _try_parse_line(line: str) -> dict[str, Any] | None:
        """
        Try all known timestamp patterns to parse a single line.

        Returns a message dict or None if no pattern matches.
        """
        for pattern in _TIMESTAMP_PATTERNS:
            match = pattern.match(line.strip())
            if match:
                timestamp_str, sender, text = match.group(1), match.group(2), match.group(3)
                is_media = "<Media omitted>" in text
                is_system = any(p.search(text) for p in _SYSTEM_MSG_PATTERNS)
                return {
                    "sender": sender.strip(),
                    "timestamp": timestamp_str.strip(),
                    "message_text": text.strip(),
                    "is_media": is_media,
                    "is_system": is_system,
                }
        return None
