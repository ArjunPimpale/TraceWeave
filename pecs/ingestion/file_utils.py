"""
File utilities for ingestion: MIME detection, SHA-256 hashing, size validation.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pecs.config import settings
from pecs.logging_config import get_logger

logger = get_logger(__name__)

# MIME type → source type mapping used by the dispatcher
MIME_TO_SOURCE_TYPE: dict[str, str] = {
    "application/pdf": "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/msword": "DOCX",
    "message/rfc822": "EMAIL",
    "text/markdown": "MARKDOWN",
    "text/x-markdown": "MARKDOWN",
    "text/x-python": "PYTHON",
    "application/x-python-code": "PYTHON",
}

# Extension → source type fallback when MIME is ambiguous
EXTENSION_TO_SOURCE_TYPE: dict[str, str] = {
    "pdf": "PDF",
    "docx": "DOCX",
    "doc": "DOCX",
    "eml": "EMAIL",
    "mbox": "EMAIL",
    "md": "MARKDOWN",
    "markdown": "MARKDOWN",
    "mdx": "MARKDOWN",
    "py": "PYTHON",
    "pyw": "PYTHON",
    "log": "GIT",
    "txt": "TEXT",  # Ambiguous — further sniffing needed
}


def compute_sha256(data: bytes) -> str:
    """
    Compute the SHA-256 hex digest of a byte sequence.

    Used for:
    - Deduplication checks (same hash = same file)
    - Deterministic chunk_id generation
    """
    return hashlib.sha256(data).hexdigest()


def detect_mime_type(data: bytes, filename: str) -> str:
    """
    Detect the MIME type of a file.

    Primary: python-magic (libmagic bindings) — more reliable than extensions.
    Fallback: extension-based detection.

    Args:
        data: Raw file bytes.
        filename: Original filename (used for extension fallback).

    Returns:
        MIME type string (e.g., "application/pdf").
    """
    try:
        import magic
        mime = magic.from_buffer(data, mime=True)
        return mime
    except Exception as exc:
        logger.debug(
            "python-magic unavailable or failed — falling back to extension",
            extra={"context": {"error": str(exc), "filename": filename}},
        )
        return _mime_from_extension(filename)


def _mime_from_extension(filename: str) -> str:
    """Fallback MIME detection using file extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    extension_mime_map: dict[str, str] = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "eml": "message/rfc822",
        "md": "text/markdown",
        "markdown": "text/markdown",
        "py": "text/x-python",
        "txt": "text/plain",
        "log": "text/plain",
    }
    return extension_mime_map.get(ext, "application/octet-stream")


def get_extension(filename: str) -> str:
    """Return the lowercase file extension without the dot."""
    return Path(filename).suffix.lower().lstrip(".")


def validate_file_size(data: bytes, filename: str) -> None:
    """
    Validate that the file size is within the configured limit.

    Raises:
        ValueError: If the file exceeds the maximum allowed size.
    """
    size = len(data)
    max_bytes = settings.max_file_size_bytes
    if size > max_bytes:
        raise ValueError(
            f"File {filename!r} is too large ({size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {settings.MAX_FILE_SIZE_MB} MB."
        )


def validate_not_empty(data: bytes, filename: str) -> None:
    """
    Validate that the file is not empty.

    Raises:
        ValueError: If the file has no bytes.
    """
    if not data:
        raise ValueError(f"File {filename!r} is empty.")


def sniff_whatsapp(data: bytes) -> bool:
    """
    Heuristic: detect if a text/plain file is a WhatsApp export.

    WhatsApp exports typically start with a BOM or contain the
    characteristic timestamp pattern in the first few lines.
    """
    try:
        text = data[:2000].decode("utf-8", errors="replace")
        import re
        # WhatsApp timestamp pattern
        wa_pattern = re.compile(
            r"\[?\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}"
        )
        lines = text.splitlines()[:10]
        matches = sum(1 for line in lines if wa_pattern.search(line))
        return matches >= 2
    except Exception:
        return False


def sniff_email(data: bytes) -> bool:
    """
    Heuristic: detect if a text/plain file is an email export.

    Email exports typically start with 'From:' or 'From ' header lines.
    """
    try:
        text = data[:1000].decode("utf-8", errors="replace")
        return text.strip().startswith("From:") or text.strip().startswith("From ")
    except Exception:
        return False


def sniff_git_log(data: bytes) -> bool:
    """
    Heuristic: detect if a text/plain file is a git log export.
    """
    try:
        text = data[:1000].decode("utf-8", errors="replace")
        import re
        # 'commit <sha>' pattern
        return bool(re.search(r"^commit [0-9a-f]{40}", text, re.MULTILINE))
    except Exception:
        return False
