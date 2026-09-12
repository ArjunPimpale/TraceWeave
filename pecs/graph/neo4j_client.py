"""
Neo4j connection management for the PECS graph layer.

Design mirrors pecs/store/database.py:
- Module-level singleton accessed via get_neo4j_client()
- Graceful degradation: all public methods are no-ops when NEO4J_ENABLED=False
- No authentication (local Community Edition)

EXTENSION POINT (future auth): when NEO4J_USER / NEO4J_PASSWORD are added to
PecsSettings, replace `auth=None` with `auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)`.
"""

from __future__ import annotations

from typing import Any

from pecs.config import settings
from pecs.logging_config import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    Thin wrapper around the official neo4j Python driver.

    Provides:
    - Lazy connection (driver opened on first use)
    - Health check
    - Write and read session factories
    - Graceful degradation when NEO4J_ENABLED is False

    Usage:
        client = get_neo4j_client()
        if client.check_health():
            with client.write_session() as session:
                session.run("MERGE (n:Test {id: $id})", id="x")
    """

    def __init__(self, uri: str | None = None, database: str | None = None) -> None:
        self._uri = uri or settings.NEO4J_URI
        self._database = database or settings.NEO4J_DATABASE
        self._driver = None

    # ── Driver lifecycle ──────────────────────────────────────────────────────

    def _get_driver(self):
        """Open the driver on first access (lazy init)."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase  # type: ignore
                # No auth for local Community Edition.
                # EXTENSION POINT: replace auth=None with auth=(user, pw) when needed.
                self._driver = GraphDatabase.driver(self._uri, auth=None)
                logger.info(
                    "Neo4j driver initialised",
                    extra={"context": {"uri": self._uri, "database": self._database}},
                )
            except Exception as exc:
                logger.error(
                    "Neo4j driver init failed",
                    extra={"context": {"uri": self._uri, "error": str(exc)}},
                )
                raise
        return self._driver

    def close(self) -> None:
        """Close the driver and release the connection pool."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    # ── Health check ──────────────────────────────────────────────────────────

    def check_health(self) -> bool:
        """
        Return True if Neo4j is reachable and the database responds.

        Safe to call even when NEO4J_ENABLED is False (returns False immediately).
        """
        if not settings.NEO4J_ENABLED:
            return False
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.warning(
                "Neo4j health check failed",
                extra={"context": {"error": str(exc)}},
            )
            return False

    # ── Session factories ─────────────────────────────────────────────────────

    def write_session(self):
        """
        Return a Neo4j Session configured for write operations.

        Usage:
            with client.write_session() as session:
                session.run("MERGE ...")
        """
        return self._get_driver().session(database=self._database)

    def read_session(self):
        """
        Return a Neo4j Session configured for read operations.

        Usage:
            with client.read_session() as session:
                result = session.run("MATCH ...")
        """
        return self._get_driver().session(database=self._database)

    # ── Convenience query helpers ─────────────────────────────────────────────

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """
        Execute a read query and return results as a list of dicts.

        Args:
            cypher: Cypher query string.
            **params: Query parameters.

        Returns:
            List of result records as dicts.
        """
        with self.read_session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def get_node_count(self) -> int:
        """Return total number of nodes in the graph (for status display)."""
        try:
            rows = self.run_read("MATCH (n) RETURN count(n) AS total")
            return rows[0]["total"] if rows else 0
        except Exception:
            return 0

    def get_graph_stats(self) -> dict[str, int]:
        """
        Return counts of nodes by label and relationships by type.

        Returns:
            Dict with keys like 'Requirement', 'Implementation', 'CORRELATES_TO', etc.
        """
        stats: dict[str, int] = {}
        try:
            # Node counts by label
            node_rows = self.run_read(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
            )
            for row in node_rows:
                if row.get("label"):
                    stats[row["label"]] = row["cnt"]

            # Relationship counts by type
            rel_rows = self.run_read(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
            )
            for row in rel_rows:
                if row.get("rel_type"):
                    stats[row["rel_type"]] = row["cnt"]
        except Exception as exc:
            logger.warning(
                "Graph stats query failed",
                extra={"context": {"error": str(exc)}},
            )
        return stats


# ── Module-level singleton ────────────────────────────────────────────────────

_client_instance: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """
    Get the module-level Neo4jClient singleton.

    Thread-safe for the single-process Streamlit use case.
    Raises RuntimeError if NEO4J_ENABLED is False.
    """
    global _client_instance
    if not settings.NEO4J_ENABLED:
        raise RuntimeError(
            "Neo4j features are disabled (NEO4J_ENABLED=false). "
            "Set NEO4J_ENABLED=true in your .env file to enable graph features."
        )
    if _client_instance is None:
        _client_instance = Neo4jClient()
    return _client_instance
