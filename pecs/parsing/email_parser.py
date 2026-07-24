"""
Email parser for pre-exported text files.

Uses regex-based parsing to split email exports into individual messages.
Handles both single-email exports and thread/conversation exports.
Preserves sender attribution and timestamp per message.
"""

from __future__ import annotations

import re
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)

# Regex patterns for email header fields
_FROM_PATTERN = re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_TO_PATTERN = re.compile(r"^To:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_DATE_PATTERN = re.compile(r"^Date:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_SUBJECT_PATTERN = re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE)

# Pattern to detect the start of a new email message in a thread export
_EMAIL_BOUNDARY = re.compile(
    r"(?:^|\n)(?:From:\s*.+\n(?:To:\s*.+\n)?(?:Date:\s*.+\n)?(?:Subject:\s*.+\n)?)",
    re.MULTILINE | re.IGNORECASE,
)

# Forwarded message detection
_FORWARDED_PATTERN = re.compile(
    r"-{5,}\s*Forwarded message\s*-{5,}",
    re.IGNORECASE,
)


class EmailParser(BaseParser):
    """
    Parser for exported email text files.

    Structural metadata: list of dicts:
        {
            "sender": str,
            "timestamp": str,
            "subject": str,
            "body": str,
            "is_forwarded": bool,
        }
    """

    SUPPORTED_MIME_TYPES = ("message/rfc822", "text/plain")
    SUPPORTED_EXTENSIONS = ("eml", "mbox", "txt")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        logger.debug(
            "Email parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        text = self._decode(request.file_bytes)
        messages = self._split_into_messages(text, warnings)

        full_text_parts: list[str] = []
        for msg in messages:
            header = f"From: {msg['sender']} | Date: {msg['timestamp']} | Subject: {msg['subject']}"
            full_text_parts.append(f"{header}\n\n{msg['body']}")

        raw_text = "\n\n---\n\n".join(full_text_parts)
        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.EMAIL.value,
            "message_count": len(messages),
            **request.user_metadata,
        }

        logger.info(
            "Email parse completed",
            extra={"context": {
                "filename": request.filename,
                "messages": len(messages),
            }},
        )

        return ParsedDocument(
            source_type=SourceType.EMAIL,
            raw_text=raw_text,
            structural_metadata=messages,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _decode(data: bytes) -> str:
        """Attempt to decode bytes with common encodings."""
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _split_into_messages(
        self, text: str, warnings: list[str]
    ) -> list[dict[str, Any]]:
        """
        Split an email export into individual message dicts.

        Strategy:
        1. Detect message boundaries by looking for 'From:' headers.
        2. For each segment, extract sender, date, subject, body.
        3. Handle forwarded messages as nested (attributed to original sender).
        """
        messages: list[dict[str, Any]] = []

        # Split by email boundaries (From: header blocks)
        segments = re.split(r"\n(?=From:\s)", text.strip())
        if len(segments) == 1:
            # Maybe a single message without thread structure
            segments = [text.strip()]

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            sender = self._extract_field(seg, _FROM_PATTERN)
            timestamp = self._extract_field(seg, _DATE_PATTERN)
            subject = self._extract_field(seg, _SUBJECT_PATTERN)

            # Body is everything after the last header line
            body = self._extract_body(seg)
            is_forwarded = bool(_FORWARDED_PATTERN.search(body))

            if not sender and not timestamp:
                # Unrecognized format segment
                warnings.append(f"Could not parse email headers in segment (length={len(seg)})")
                continue

            messages.append({
                "sender": sender or "Unknown",
                "timestamp": timestamp or "Unknown",
                "subject": subject or "(no subject)",
                "body": body,
                "is_forwarded": is_forwarded,
            })

        return messages

    @staticmethod
    def _extract_field(text: str, pattern: re.Pattern) -> str:
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_body(text: str) -> str:
        """
        Extract the email body by skipping recognized header lines.

        Header lines are lines starting with: From:, To:, Date:, Subject:,
        Cc:, Bcc:, Reply-To:, Message-ID:, Content-Type:, etc.
        """
        header_pattern = re.compile(
            r"^(?:From|To|Date|Subject|Cc|Bcc|Reply-To|Message-ID|"
            r"Content-Type|Content-Transfer-Encoding|MIME-Version|"
            r"Received|Return-Path|X-\w+):\s*.*$",
            re.MULTILINE | re.IGNORECASE,
        )
        # Remove header lines
        body = header_pattern.sub("", text).strip()
        # Remove leading separator lines (e.g., "---")
        body = re.sub(r"^\s*[-_=]{3,}\s*$", "", body, flags=re.MULTILINE).strip()
        return body
