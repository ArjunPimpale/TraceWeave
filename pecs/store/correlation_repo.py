"""
Correlation repository — CRUD for the correlation table.

Design constraints:
- Append-only (no UPDATE). Correlations are immutable once written.
- Re-correlation runs insert new rows; old rows remain for audit.
- The UI shows the latest correlation for each requirement by default.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationResult
from pecs.store.database import Database, get_database

logger = get_logger(__name__)


_INSERT_CORRELATION_SQL = """
    INSERT INTO correlation
        (correlation_id, requirement_entity_id, evidence_entity_id,
         status, resolution_method, rule_name, confidence,
         supporting_chunk_ids, reasoning)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _insert_params(result: CorrelationResult) -> tuple[Any, ...]:
    """Return the correlation-table values in the schema's insertion order."""
    return (
        result.correlation_id,
        result.requirement_entity_id,
        result.evidence_entity_id,
        result.status.value,
        result.resolution_method,
        result.rule_name,
        result.confidence,
        json.dumps(result.supporting_chunk_ids),
        result.reasoning,
    )


class CorrelationRepo:
    """
    Repository for the correlation table.

    All writes are inserts only; the table is append-only for auditability.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_database()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._db.connect()

    # ── Write ────────────────────────────────────────────────────────────────

    def insert(self, result: CorrelationResult) -> int:
        """
        Insert a single CorrelationResult into the correlation table.

        Args:
            result: Validated CorrelationResult to insert.

        Returns:
            The auto-generated row id.
        """
        cursor = self._conn.execute(_INSERT_CORRELATION_SQL, _insert_params(result))
        self._conn.commit()
        row_id = cursor.lastrowid
        logger.debug(
            "Correlation row inserted",
            extra={"context": {
                "correlation_id": result.correlation_id,
                "req": result.requirement_entity_id,
                "status": result.status.value,
            }},
        )
        return row_id

    def insert_batch(self, results: list[CorrelationResult]) -> int:
        """
        Insert multiple CorrelationResult objects in a single transaction.

        Returns:
            Number of rows inserted.
        """
        with self._conn:
            self._conn.executemany(
                _INSERT_CORRELATION_SQL, [_insert_params(result) for result in results]
            )
        logger.info(
            "Batch correlation insert complete",
            extra={"context": {"inserted": len(results)}},
        )
        return len(results)

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_latest_for_requirement(
        self, requirement_entity_id: str
    ) -> dict[str, Any] | None:
        """
        Return the most recent correlation for a given requirement.

        'Most recent' is determined by created_at DESC.
        """
        row = self._conn.execute(
            """
            SELECT * FROM correlation
            WHERE requirement_entity_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (requirement_entity_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_all_for_requirement(self, requirement_entity_id: str) -> list[dict[str, Any]]:
        """Return all correlations for a requirement, most recent first."""
        rows = self._conn.execute(
            """
            SELECT * FROM correlation
            WHERE requirement_entity_id = ?
            ORDER BY created_at DESC
            """,
            (requirement_entity_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_latest(self) -> list[dict[str, Any]]:
        """
        Return the most recent correlation for each requirement.

        Used to build the traceability matrix view.
        """
        rows = self._conn.execute(
            """
            SELECT c.*
            FROM correlation c
            INNER JOIN (
                SELECT requirement_entity_id, MAX(created_at) AS max_created
                FROM correlation
                GROUP BY requirement_entity_id
            ) latest
            ON c.requirement_entity_id = latest.requirement_entity_id
               AND c.created_at = latest.max_created
            ORDER BY c.requirement_entity_id
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_status(self, status: str) -> list[dict[str, Any]]:
        """Return all correlations with a given status."""
        rows = self._conn.execute(
            "SELECT * FROM correlation WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all correlation rows."""
        rows = self._conn.execute(
            "SELECT * FROM correlation ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
