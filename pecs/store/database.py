"""
SQLite database connection management and schema creation.

Design principles:
- One immutable Evidence table + one Correlation table
- Rows are never updated (append-only for auditability)
- All insertions within a single document are wrapped in a transaction
- WAL mode for better concurrent read performance
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pecs.config import settings
from pecs.logging_config import get_logger

logger = get_logger(__name__)

# ── Schema DDL ───────────────────────────────────────────────────────────────

EVIDENCE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('REQUIREMENT', 'IMPLEMENTATION', 'EVALUATION')),
    entity_id       TEXT NOT NULL,
    text            TEXT NOT NULL,
    linked_requirement TEXT,
    source_document TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    author          TEXT,
    timestamp       TEXT,
    metadata        TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),

    CHECK(length(source_document) > 0),
    CHECK(length(chunk_id) > 0)
);
"""

CORRELATION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS correlation (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id        TEXT NOT NULL UNIQUE,
    requirement_entity_id TEXT NOT NULL,
    evidence_entity_id    TEXT NOT NULL,
    status                TEXT NOT NULL CHECK(status IN (
                              'IMPLEMENTED_AND_VALIDATED',
                              'IMPLEMENTED_BUT_NEGATIVELY_EVALUATED',
                              'IMPLEMENTED_WITHOUT_EVALUATION',
                              'PARTIALLY_IMPLEMENTED',
                              'CLAIMED_BUT_NO_EVIDENCE',
                              'EVALUATION_WITHOUT_REQUIREMENT',
                              'REQUIREMENT_NOT_IMPLEMENTED'
                          )),
    resolution_method     TEXT NOT NULL CHECK(resolution_method IN ('deterministic_rule', 'llm_stage2')),
    rule_name             TEXT,
    confidence            REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    supporting_chunk_ids  TEXT,
    reasoning             TEXT,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);
"""

INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_evidence_entity_type ON evidence(entity_type);",
    "CREATE INDEX IF NOT EXISTS idx_evidence_entity_id ON evidence(entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_evidence_linked_req ON evidence(linked_requirement);",
    "CREATE INDEX IF NOT EXISTS idx_evidence_source_doc ON evidence(source_document);",
    "CREATE INDEX IF NOT EXISTS idx_evidence_chunk_id ON evidence(chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_correlation_req ON correlation(requirement_entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_correlation_evidence ON correlation(evidence_entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_correlation_status ON correlation(status);",
]

# ── Ingestion tracking table ──────────────────────────────────────────────────

INGESTION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    file_hash       TEXT NOT NULL UNIQUE,
    file_size       INTEGER NOT NULL,
    source_type     TEXT NOT NULL,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'success',
    error_message   TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);
"""

INGESTION_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_ingestion_hash ON ingestion_log(file_hash);",
]


class Database:
    """
    Manages SQLite database connections and schema initialization.

    Provides thread-safe connections using SQLite's check_same_thread=False
    (appropriate for the single-process Streamlit use case).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.sqlite_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """
        Open a connection (or return the existing one).

        Enables:
        - Row factory for dict-like row access
        - WAL mode for better read concurrency
        - Foreign keys enforcement
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            logger.debug(
                "SQLite connection opened",
                extra={"context": {"path": str(self._db_path)}},
            )
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize_schema(self) -> None:
        """
        Create all tables and indexes if they don't already exist.
        Idempotent — safe to call on every startup.
        """
        conn = self.connect()
        with conn:
            conn.execute(EVIDENCE_TABLE_DDL)
            conn.execute(CORRELATION_TABLE_DDL)
            conn.execute(INGESTION_TABLE_DDL)
            for idx_ddl in INDEXES_DDL + INGESTION_INDEX_DDL:
                conn.execute(idx_ddl)
        logger.info(
            "SQLite schema initialized",
            extra={"context": {"path": str(self._db_path)}},
        )

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# Module-level singleton
_db_instance: Database | None = None


def get_database() -> Database:
    """
    Get the module-level Database singleton.

    Initializes the schema on first access.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        _db_instance.initialize_schema()
    return _db_instance
