"""
Centralized logging configuration for PECS.

Sets up structured JSON logging to both file and console.
Provides log rotation (10 MB, keep 5 files) and dual output.

Usage:
    from pecs.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("File ingested", extra={"context": {"filename": "foo.pdf", "hash": "abc"}})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pecs.config import settings


class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Each log entry has:
        timestamp, level, module, message, context (if provided)
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Attach structured context if passed via extra={"context": {...}}
        if hasattr(record, "context") and record.context:
            entry["context"] = record.context

        # Attach exception info if present
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


def setup_logging() -> None:
    """
    Configure root logger with:
    - JSON formatter
    - File handler with rotation (10 MB, keep 5 files)
    - Stream handler (stderr) for interactive use
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if called multiple times (e.g., Streamlit hot-reload)
    if root_logger.handlers:
        return

    formatter = JsonFormatter()

    # ── File handler with rotation ──────────────────────────────────────────
    log_path = settings.log_file_path
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # ── Stream handler (stderr) ─────────────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "chromadb", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Ensures logging is set up before returning the logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Ingestion complete", extra={"context": {"chunks": 15}})
    """
    setup_logging()
    return logging.getLogger(name)


# Initialize on import so any module that imports get_logger gets logging set up
setup_logging()
