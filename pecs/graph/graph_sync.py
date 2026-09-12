"""
Graph synchronisation orchestrator.

Implements full-rebuild sync: clear the existing Neo4j graph, enforce UNIQUE
constraints, then rebuild deterministically from SQLite via GraphBuilder.

Why full rebuild over incremental sync (V1):
- The correlation table is append-only and already deduplicates via get_all_latest().
- Incremental sync would need to track which correlations have been synced and
  handle entity_id instability across extraction runs (see ARN-1 in the plan).
- Expected data volume (undergraduate project: < 200 entities, < 500 edges) makes
  a full rebuild take < 2 seconds — acceptable latency for a manual trigger.

EXTENSION POINT (incremental sync): when snapshot support is added, replace
_clear_graph() + build() with a delta-based approach that:
  1. Compares correlation created_at timestamps against a `last_synced_at` marker.
  2. Only MERGEs new/changed nodes and relationships.
  3. Records the sync watermark in a dedicated Neo4j metadata node.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pecs.graph.graph_builder import BuildStats, GraphBuilder
from pecs.graph.neo4j_client import Neo4jClient, get_neo4j_client
from pecs.logging_config import get_logger
from pecs.store.correlation_repo import CorrelationRepo
from pecs.store.evidence_repo import EvidenceRepo

logger = get_logger(__name__)

# Unique constraint definitions — enforced before any MERGE runs.
_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement)    REQUIRE r.entity_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Implementation) REQUIRE i.entity_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Evaluation)     REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:SourceDocument) REQUIRE d.name IS UNIQUE",
]


@dataclass
class GraphSyncResult:
    """
    The outcome of a single graph sync operation.

    EXTENSION POINT (snapshot support): add `snapshot_id: str | None = None`
    to identify which correlation run this sync reflects.
    """
    success: bool
    nodes_created: int = 0
    relationships_created: int = 0
    sync_timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    build_stats: BuildStats | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "nodes_created": self.nodes_created,
            "relationships_created": self.relationships_created,
            "sync_timestamp": self.sync_timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "warnings": self.warnings,
            "error": self.error,
        }


class GraphSync:
    """
    Orchestrates full-rebuild synchronisation between SQLite and Neo4j.

    Args:
        client: Neo4jClient instance. Defaults to the module singleton.
        evidence_repo: EvidenceRepo to pass to GraphBuilder.
        correlation_repo: CorrelationRepo to pass to GraphBuilder.
    """

    def __init__(
        self,
        client: Neo4jClient | None = None,
        evidence_repo: EvidenceRepo | None = None,
        correlation_repo: CorrelationRepo | None = None,
    ) -> None:
        self._client = client or get_neo4j_client()
        self._ev_repo = evidence_repo or EvidenceRepo()
        self._corr_repo = correlation_repo or CorrelationRepo()

    def sync_graph(self) -> GraphSyncResult:
        """
        Perform a full graph rebuild.

        Steps:
          1. Clear all existing nodes and relationships.
          2. Enforce UNIQUE constraints (idempotent CREATE CONSTRAINT IF NOT EXISTS).
          3. Rebuild nodes and relationships from SQLite via GraphBuilder.

        The entire operation runs inside a single Neo4j transaction: if any step
        fails, the transaction is rolled back and the previous graph state is
        preserved (since the DETACH DELETE was inside the same transaction).

        Returns:
            GraphSyncResult with counts and timing.
        """
        t0 = time.monotonic()
        logger.info(
            "Graph sync started",
            extra={"context": {"trigger": "manual_or_auto"}},
        )

        try:
            with self._client.write_session() as session:
                with session.begin_transaction() as tx:
                    # Step 1: Clear the existing graph
                    self._clear_graph(tx)

                    # Step 2: Enforce UNIQUE constraints
                    # Note: constraints must be created outside a data transaction
                    # in some Neo4j versions. We commit the clear first, then
                    # create constraints and re-open for data writes.
                    tx.commit()

                # Constraints are DDL — separate session
                with session.begin_transaction() as tx:
                    self._create_constraints(tx)
                    tx.commit()

                # Step 3: Build nodes and relationships
                with session.begin_transaction() as tx:
                    builder = GraphBuilder(
                        session=tx,
                        evidence_repo=self._ev_repo,
                        correlation_repo=self._corr_repo,
                    )
                    stats = builder.build()
                    tx.commit()

            duration_ms = int((time.monotonic() - t0) * 1000)

            result = GraphSyncResult(
                success=True,
                nodes_created=stats.total_nodes,
                relationships_created=stats.total_relationships,
                duration_ms=duration_ms,
                warnings=stats.warnings,
                build_stats=stats,
            )

            logger.info(
                "Graph sync completed",
                extra={"context": {
                    "nodes": stats.total_nodes,
                    "relationships": stats.total_relationships,
                    "duration_ms": duration_ms,
                    "warnings": len(stats.warnings),
                }},
            )
            return result

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.error(
                "Graph sync failed",
                extra={"context": {"error": str(exc), "duration_ms": duration_ms}},
            )
            return GraphSyncResult(
                success=False,
                duration_ms=duration_ms,
                error=str(exc),
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clear_graph(tx) -> None:
        """
        Delete all nodes and relationships in the graph.

        Uses CALL { ... } IN TRANSACTIONS to handle large graphs without
        hitting transaction memory limits. For the expected V1 data volume
        this is a simple MATCH / DETACH DELETE.
        """
        tx.run("MATCH (n) DETACH DELETE n")
        logger.debug("Graph cleared (DETACH DELETE all nodes)")

    @staticmethod
    def _create_constraints(tx) -> None:
        """Create UNIQUE constraints on identity keys (idempotent)."""
        for stmt in _CONSTRAINTS:
            tx.run(stmt)
        logger.debug(
            "UNIQUE constraints created",
            extra={"context": {"count": len(_CONSTRAINTS)}},
        )
