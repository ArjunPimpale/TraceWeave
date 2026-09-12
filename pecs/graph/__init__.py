"""
Graph interpretability layer for PECS.

This package deterministically transforms existing validated entities and
correlation results from SQLite into a Neo4j graph for interactive exploration.

The graph is a *derived view* of the SQLite truth — it never produces new
correlations or invents relationships. All edges originate from CorrelationResult
rows written by the rule engine or Stage 2 classifier.

Modules:
    neo4j_client  -- Connection management and health checks.
    graph_builder -- Deterministic node/relationship construction from SQLite.
    graph_queries -- Parameterised Cypher query templates for the UI.
    graph_sync    -- Full-rebuild orchestration and sync result reporting.
"""

from pecs.graph.neo4j_client import Neo4jClient, get_neo4j_client
from pecs.graph.graph_sync import GraphSync, GraphSyncResult

__all__ = [
    "Neo4jClient",
    "get_neo4j_client",
    "GraphSync",
    "GraphSyncResult",
]
