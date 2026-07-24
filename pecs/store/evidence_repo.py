"""
Evidence repository — CRUD operations for the evidence table.

Design constraints:
- INSERT only (no UPDATE). Evidence rows are immutable.
- Idempotency: same entity_id + chunk_id + entity_type → skip (don't insert twice).
- All insertions for a single document are wrapped in a transaction.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.extraction_result import ExtractionResult, EntityType
from pecs.store.database import Database, get_database

logger = get_logger(__name__)


class EvidenceRepo:
    """
    Repository for the evidence table.

    Provides insert-only access patterns aligned with the immutability constraint.
    All reads are read-only; no update methods exist by design.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_database()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._db.connect()

    # ── Write ────────────────────────────────────────────────────────────────

    def insert(self, result: ExtractionResult) -> int | None:
        """
        Insert a single ExtractionResult into the evidence table.

        Skips (idempotent) if an identical row already exists
        (same entity_id + chunk_id + entity_type).

        Args:
            result: Validated ExtractionResult to insert.

        Returns:
            The auto-generated row id, or None if the row was skipped.
        """
        if self._row_exists(result.entity_id, result.chunk_id, result.entity_type.value):
            logger.debug(
                "Evidence row already exists — skipping",
                extra={"context": {
                    "entity_id": result.entity_id,
                    "chunk_id": result.chunk_id[:8],
                }},
            )
            return None

        sql = """
            INSERT INTO evidence
                (entity_type, entity_id, text, linked_requirement,
                 source_document, chunk_id, author, timestamp, metadata)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            result.entity_type.value,
            result.entity_id,
            result.text,
            result.linked_requirement,
            result.source_document,
            result.chunk_id,
            result.author,
            result.timestamp,
            json.dumps(result.metadata) if result.metadata else None,
        )

        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        row_id = cursor.lastrowid
        logger.debug(
            "Evidence row inserted",
            extra={"context": {
                "row_id": row_id,
                "entity_type": result.entity_type.value,
                "entity_id": result.entity_id,
            }},
        )
        return row_id

    def insert_batch(self, results: list[ExtractionResult]) -> list[int]:
        """
        Insert multiple ExtractionResult objects in a single transaction.

        If any insertion fails, the entire batch is rolled back.

        Args:
            results: List of validated ExtractionResult objects.

        Returns:
            List of successfully inserted row ids (skipped rows are excluded).
        """
        inserted_ids: list[int] = []
        with self._conn:
            for result in results:
                if self._row_exists(
                    result.entity_id, result.chunk_id, result.entity_type.value
                ):
                    continue
                sql = """
                    INSERT INTO evidence
                        (entity_type, entity_id, text, linked_requirement,
                         source_document, chunk_id, author, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor = self._conn.execute(
                    sql,
                    (
                        result.entity_type.value,
                        result.entity_id,
                        result.text,
                        result.linked_requirement,
                        result.source_document,
                        result.chunk_id,
                        result.author,
                        result.timestamp,
                        json.dumps(result.metadata) if result.metadata else None,
                    ),
                )
                if cursor.lastrowid:
                    inserted_ids.append(cursor.lastrowid)

        logger.info(
            "Batch evidence insert complete",
            extra={"context": {"inserted": len(inserted_ids), "total": len(results)}},
        )
        return inserted_ids

    def log_ingestion(
        self,
        filename: str,
        file_hash: str,
        file_size: int,
        source_type: str,
        chunk_count: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        """
        Log a file ingestion event to the ingestion_log table.

        Uses INSERT OR REPLACE so a successful retry can overwrite a prior
        failure row for the same file_hash. This ensures failed files can
        always be re-uploaded and re-processed.
        """
        sql = """
            INSERT OR REPLACE INTO ingestion_log
                (filename, file_hash, file_size, source_type, chunk_count, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(
            sql,
            (filename, file_hash, file_size, source_type, chunk_count, status, error_message),
        )
        self._conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_all_requirements(self) -> list[dict[str, Any]]:
        """Return all rows of type REQUIREMENT."""
        return self._fetch_by_entity_type("REQUIREMENT")

    def get_all_implementations(self) -> list[dict[str, Any]]:
        """Return all rows of type IMPLEMENTATION."""
        return self._fetch_by_entity_type("IMPLEMENTATION")

    def get_all_evaluations(self) -> list[dict[str, Any]]:
        """Return all rows of type EVALUATION."""
        return self._fetch_by_entity_type("EVALUATION")

    def get_by_entity_id(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all evidence rows matching the given entity_id."""
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE entity_id = ?", (entity_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_linked_requirement(self, requirement_id: str) -> list[dict[str, Any]]:
        """Return all evidence rows explicitly linked to a requirement."""
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE linked_requirement = ?", (requirement_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_chunk_id(self, chunk_id: str) -> list[dict[str, Any]]:
        """Return all evidence rows associated with a given chunk_id."""
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE chunk_id = ?", (chunk_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_source_document(self, source_document: str) -> list[dict[str, Any]]:
        """Return all evidence rows from a specific source document."""
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE source_document = ?", (source_document,)
        ).fetchall()
        return [dict(r) for r in rows]

    def is_file_ingested(self, file_hash: str) -> bool:
        """
        Return True only if this file hash was previously ingested SUCCESSFULLY.

        Files with a 'failed' status in the log are NOT blocked — the user
        must be able to fix the issue and re-upload the same file.
        """
        row = self._conn.execute(
            "SELECT 1 FROM ingestion_log WHERE file_hash = ? AND status = 'success'",
            (file_hash,),
        ).fetchone()
        return row is not None

    def get_ingestion_log(self) -> list[dict[str, Any]]:
        """Return all ingestion log entries, most recent first."""
        rows = self._conn.execute(
            "SELECT * FROM ingestion_log ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all evidence rows."""
        rows = self._conn.execute(
            "SELECT * FROM evidence ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _row_exists(self, entity_id: str, chunk_id: str, entity_type: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM evidence
            WHERE entity_id = ? AND chunk_id = ? AND entity_type = ?
            """,
            (entity_id, chunk_id, entity_type),
        ).fetchone()
        return row is not None

    def _fetch_by_entity_type(self, entity_type: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE entity_type = ? ORDER BY created_at DESC",
            (entity_type,),
        ).fetchall()
        return [dict(r) for r in rows]
