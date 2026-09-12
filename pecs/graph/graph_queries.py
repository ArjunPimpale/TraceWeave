"""
Cypher query templates for the PECS graph UI layer.

All queries are read-only (no MERGE/CREATE/DELETE). They return data structures
that the UI layer converts into pyvis nodes and edges.

Query naming convention:
  get_*  -- returns data for the UI
  count_* -- returns integer counts for status panels

Each query is a module-level string constant plus a companion function that
executes it via the Neo4jClient and returns typed results.

EXTENSION POINT (snapshot support): add a `created_after: str | None = None`
parameter to any query that filters on created_at timestamps when per-run
snapshots are implemented.
"""

from __future__ import annotations

from typing import Any

from pecs.graph.neo4j_client import Neo4jClient
from pecs.logging_config import get_logger

logger = get_logger(__name__)


# ── Raw Cypher templates ──────────────────────────────────────────────────────

_REQUIREMENT_OVERVIEW = """
MATCH (r:Requirement)
OPTIONAL MATCH (r)-[c:CORRELATES_TO]->(ev)
OPTIONAL MATCH (eval:Evaluation)-[:EVALUATES]->(r)
RETURN
    r.entity_id          AS entity_id,
    r.text               AS text,
    r.source_document    AS source_document,
    count(DISTINCT ev)   AS implementation_count,
    count(DISTINCT eval) AS evaluation_count,
    collect(DISTINCT c.status)[0] AS status,
    collect(DISTINCT c.confidence)[0] AS confidence
ORDER BY r.entity_id
"""

_REQUIREMENT_EGO_NETWORK = """
MATCH (r:Requirement {entity_id: $entity_id})
OPTIONAL MATCH (r)-[c:CORRELATES_TO]->(ev)
OPTIONAL MATCH (eval:Evaluation)-[:EVALUATES]->(r)
OPTIONAL MATCH (impl:Implementation)-[:EXPLICITLY_LINKS]->(r)
OPTIONAL MATCH (ev)-[:EXTRACTED_FROM]->(doc:SourceDocument)
OPTIONAL MATCH (r)-[:EXTRACTED_FROM]->(rdoc:SourceDocument)
RETURN
    r,
    collect(DISTINCT {rel: c, node: ev})   AS correlations,
    collect(DISTINCT eval)                  AS evaluations,
    collect(DISTINCT impl)                  AS explicit_impls,
    collect(DISTINCT doc)                   AS evidence_docs,
    collect(DISTINCT rdoc)                  AS req_docs
"""

_SHARED_ARTIFACTS = """
MATCH (r1:Requirement)-[:CORRELATES_TO]->(shared)<-[:CORRELATES_TO]-(r2:Requirement)
WHERE r1.entity_id < r2.entity_id
RETURN
    r1.entity_id  AS req1_id,
    r1.text       AS req1_text,
    r2.entity_id  AS req2_id,
    r2.text       AS req2_text,
    labels(shared)[0]     AS shared_type,
    shared.entity_id      AS shared_id,
    shared.text           AS shared_text,
    shared.source_document AS shared_source
ORDER BY r1.entity_id, r2.entity_id
"""

_ORPHAN_ENTITIES = """
MATCH (n)
WHERE NOT (n)--()
  AND NOT n:SourceDocument
RETURN
    labels(n)[0]  AS entity_type,
    n.entity_id   AS entity_id,
    n.text        AS text
ORDER BY entity_type, entity_id
"""

_SOURCE_DOCUMENT_COVERAGE = """
MATCH (n)-[:EXTRACTED_FROM]->(doc:SourceDocument {name: $doc_name})
OPTIONAL MATCH (req:Requirement)-[:CORRELATES_TO]->(n)
RETURN
    labels(n)[0]  AS entity_type,
    n.entity_id   AS entity_id,
    n.text        AS text,
    collect(DISTINCT req.entity_id) AS linked_requirements
ORDER BY entity_type, entity_id
"""

_GRAPH_STATS = """
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS cnt
UNION ALL
MATCH ()-[r]->()
RETURN type(r) AS label, count(r) AS cnt
"""

_FULL_GRAPH_NODES = """
MATCH (n)
WHERE NOT n:SourceDocument OR $include_source_docs = true
RETURN
    labels(n)[0]            AS label,
    n.entity_id             AS entity_id,
    n.name                  AS name,
    n.text                  AS text,
    n.source_document       AS source_document,
    n.source_type           AS source_type,
    n.author                AS author,
    n.timestamp             AS timestamp,
    n.linked_requirement    AS linked_requirement,
    n.chunk_id              AS chunk_id
"""


_FULL_GRAPH_EDGES = """
MATCH (a)-[r]->(b)
RETURN
    labels(a)[0]    AS from_label,
    a.entity_id     AS from_id,
    a.name          AS from_name,
    type(r)         AS rel_type,
    labels(b)[0]    AS to_label,
    b.entity_id     AS to_id,
    b.name          AS to_name,
    r.status        AS status,
    r.confidence    AS confidence,
    r.resolution_method AS resolution_method,
    r.rule_name     AS rule_name,
    r.reasoning     AS reasoning,
    r.correlation_id AS correlation_id
"""


# ── Query execution functions ─────────────────────────────────────────────────

def get_requirement_overview(client: Neo4jClient) -> list[dict[str, Any]]:
    """
    Return a summary of all requirements with their connection counts.

    Used to populate the requirement selector in the graph page.
    Each row: entity_id, text, implementation_count, evaluation_count, status, confidence.
    """
    try:
        rows = client.run_read(_REQUIREMENT_OVERVIEW)
        logger.debug(
            "get_requirement_overview",
            extra={"context": {"count": len(rows)}},
        )
        return rows
    except Exception as exc:
        logger.warning(
            "get_requirement_overview failed",
            extra={"context": {"error": str(exc)}},
        )
        return []


def get_requirement_ego_network(
    client: Neo4jClient, entity_id: str
) -> dict[str, Any]:
    """
    Return all nodes and relationships within 1–2 hops of a requirement.

    Used to build the focused ego-network view when a requirement is selected.

    Returns a dict with keys:
        requirement: the central Requirement node properties
        correlations: list of {rel_props, evidence_node_props}
        evaluations: list of Evaluation node properties
        explicit_impls: list of Implementation nodes with EXPLICITLY_LINKS
        evidence_docs: list of SourceDocument nodes connected to evidence
        req_docs: list of SourceDocument nodes connected to the requirement
    """
    try:
        rows = client.run_read(_REQUIREMENT_EGO_NETWORK, entity_id=entity_id)
        if not rows:
            return {}
        logger.debug(
            "get_requirement_ego_network",
            extra={"context": {"entity_id": entity_id}},
        )
        return rows[0]
    except Exception as exc:
        logger.warning(
            "get_requirement_ego_network failed",
            extra={"context": {"entity_id": entity_id, "error": str(exc)}},
        )
        return {}


def get_shared_artifacts(client: Neo4jClient) -> list[dict[str, Any]]:
    """
    Return artifacts that are connected to more than one requirement.

    These represent cross-cutting implementation pieces — a file or piece
    of evidence that satisfies multiple requirements simultaneously.
    """
    try:
        rows = client.run_read(_SHARED_ARTIFACTS)
        logger.debug(
            "get_shared_artifacts",
            extra={"context": {"count": len(rows)}},
        )
        return rows
    except Exception as exc:
        logger.warning(
            "get_shared_artifacts failed",
            extra={"context": {"error": str(exc)}},
        )
        return []


def get_orphan_entities(client: Neo4jClient) -> list[dict[str, Any]]:
    """
    Return entities with no relationships.

    Typically: orphan evaluations, or requirements with no correlated evidence
    that were stored as isolated nodes.
    """
    try:
        return client.run_read(_ORPHAN_ENTITIES)
    except Exception as exc:
        logger.warning(
            "get_orphan_entities failed",
            extra={"context": {"error": str(exc)}},
        )
        return []


def get_source_document_coverage(
    client: Neo4jClient, doc_name: str
) -> list[dict[str, Any]]:
    """
    Return all entities extracted from a given source document.

    Useful for understanding which requirements a document contributes to.
    """
    try:
        return client.run_read(_SOURCE_DOCUMENT_COVERAGE, doc_name=doc_name)
    except Exception as exc:
        logger.warning(
            "get_source_document_coverage failed",
            extra={"context": {"doc_name": doc_name, "error": str(exc)}},
        )
        return []


def get_full_graph(
    client: Neo4jClient,
    include_source_docs: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return all nodes and all relationships in the graph.

    Used for the full-graph overview (all requirements at once).
    Returns (nodes, edges) where each is a list of dicts.

    Args:
        include_source_docs: If False, SourceDocument nodes are excluded
                             (reduces visual clutter).
    """
    try:
        nodes = client.run_read(
            _FULL_GRAPH_NODES,
            include_source_docs=include_source_docs,
        )
        edges = client.run_read(_FULL_GRAPH_EDGES)
        logger.debug(
            "get_full_graph",
            extra={"context": {"nodes": len(nodes), "edges": len(edges)}},
        )
        return nodes, edges
    except Exception as exc:
        logger.warning(
            "get_full_graph failed",
            extra={"context": {"error": str(exc)}},
        )
        return [], []


def get_graph_stats(client: Neo4jClient) -> dict[str, int]:
    """Return node and relationship counts by type."""
    try:
        return client.get_graph_stats()
    except Exception as exc:
        logger.warning(
            "get_graph_stats failed",
            extra={"context": {"error": str(exc)}},
        )
        return {}
