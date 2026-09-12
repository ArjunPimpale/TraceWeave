"""
Deterministic graph builder — transforms SQLite evidence into Neo4j nodes and edges.

This module is the core of the graph layer. It reads from the SQLite evidence and
correlation tables, then writes to Neo4j using idempotent MERGE operations.

Key design decisions:
- No LLM, no additional reasoning. Every node and relationship is derived directly
  from ExtractionResult / CorrelationResult data already in SQLite.
- MERGE, not CREATE. Every write is idempotent. Running the builder twice produces
  the same graph.
- Batch writes via UNWIND to reduce network round trips.
- evidence_entity_id dual-mode: CorrelationResult.evidence_entity_id can hold either
  an entity_id (R1, R2 prefix) or a chunk_id (64-char hex from Rule 2/Stage 2).
  The builder resolves chunk_id → entity_id via evidence_repo.get_by_chunk_id().

Node types created:
  :Requirement, :Implementation, :Evaluation, :SourceDocument

Relationship types created:
  :CORRELATES_TO    -- Requirement → Implementation/Evaluation (from correlation table)
  :EXPLICITLY_LINKS -- Implementation → Requirement (from linked_requirement field)
  :EVALUATES        -- Evaluation → Requirement (from linked_requirement field)
  :EXTRACTED_FROM   -- Any entity → SourceDocument (provenance)

EXTENSION POINT (snapshot support): GraphBuildOptions currently defaults to
latest_only=True. When snapshot support is added, pass a `correlation_run_id`
or `snapshot_timestamp` to filter correlations to a specific run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pecs.logging_config import get_logger
from pecs.store.correlation_repo import CorrelationRepo
from pecs.store.evidence_repo import EvidenceRepo

logger = get_logger(__name__)

# ── Chunk-id detection ────────────────────────────────────────────────────────
# EvidenceChunk IDs are SHA-256 hex digests (64 lowercase hex chars).
_CHUNK_ID_RE = re.compile(r"^[0-9a-f]{64}$")

# Batch size for UNWIND operations
_BATCH_SIZE = 100


def _is_chunk_id(value: str) -> bool:
    """Return True if the value looks like a chunk_id (SHA-256 hex), not an entity_id."""
    return bool(_CHUNK_ID_RE.match(value))


def _infer_source_type(filename: str) -> str:
    """
    Infer SourceType from file extension when not stored in the evidence table.

    The evidence table does not store source_type directly (it lives in the
    EvidenceChunk / ChromaDB metadata). We infer it from the filename extension
    as a best-effort fallback.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mapping = {
        "pdf": "PDF",
        "docx": "DOCX",
        "doc": "DOCX",
        "py": "PYTHON",
        "md": "MARKDOWN",
        "txt": "TEXT",
        "eml": "EMAIL",
        "msg": "EMAIL",
    }
    source_type = mapping.get(ext, "UNKNOWN")
    # GIT repositories are usually named with "git" or have no extension
    if "git" in filename.lower() or "commit" in filename.lower():
        source_type = "GIT"
    if "whatsapp" in filename.lower() or "chat" in filename.lower():
        source_type = "WHATSAPP"
    return source_type


@dataclass
class BuildStats:
    """Counts of nodes and relationships written during a build."""
    requirement_nodes: int = 0
    implementation_nodes: int = 0
    evaluation_nodes: int = 0
    source_document_nodes: int = 0
    correlates_to_rels: int = 0
    explicitly_links_rels: int = 0
    evaluates_rels: int = 0
    extracted_from_rels: int = 0
    chunk_id_resolutions: int = 0
    skipped_none_evidence: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return (
            self.requirement_nodes
            + self.implementation_nodes
            + self.evaluation_nodes
            + self.source_document_nodes
        )

    @property
    def total_relationships(self) -> int:
        return (
            self.correlates_to_rels
            + self.explicitly_links_rels
            + self.evaluates_rels
            + self.extracted_from_rels
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {
                "Requirement": self.requirement_nodes,
                "Implementation": self.implementation_nodes,
                "Evaluation": self.evaluation_nodes,
                "SourceDocument": self.source_document_nodes,
                "total": self.total_nodes,
            },
            "relationships": {
                "CORRELATES_TO": self.correlates_to_rels,
                "EXPLICITLY_LINKS": self.explicitly_links_rels,
                "EVALUATES": self.evaluates_rels,
                "EXTRACTED_FROM": self.extracted_from_rels,
                "total": self.total_relationships,
            },
            "chunk_id_resolutions": self.chunk_id_resolutions,
            "skipped_none_evidence": self.skipped_none_evidence,
            "warnings": self.warnings,
        }


class GraphBuilder:
    """
    Deterministic transformation from SQLite evidence → Neo4j graph.

    Operates within a single Neo4j session. The caller (GraphSync) is
    responsible for session lifecycle and constraint creation.

    Args:
        session: An open neo4j.Session for write operations.
        evidence_repo: EvidenceRepo to read entities from SQLite.
        correlation_repo: CorrelationRepo to read correlations from SQLite.
    """

    def __init__(
        self,
        session,
        evidence_repo: EvidenceRepo | None = None,
        correlation_repo: CorrelationRepo | None = None,
    ) -> None:
        self._session = session
        self._ev_repo = evidence_repo or EvidenceRepo()
        self._corr_repo = correlation_repo or CorrelationRepo()

    def build(self) -> BuildStats:
        """
        Execute the full deterministic build sequence.

        EXTENSION POINT (snapshot support): accept a `snapshot_timestamp: str | None`
        parameter and pass it to `_load_correlations()` to filter by run time.

        Returns:
            BuildStats with counts of nodes and relationships created.
        """
        stats = BuildStats()

        # ── 1. Load all entities from SQLite ──────────────────────────────────
        requirements = self._ev_repo.get_all_requirements()
        implementations = self._ev_repo.get_all_implementations()
        evaluations = self._ev_repo.get_all_evaluations()
        # Latest correlation per requirement (append-only table, most recent wins)
        correlations = self._corr_repo.get_all_latest()

        logger.info(
            "Graph build: evidence loaded from SQLite",
            extra={"context": {
                "requirements": len(requirements),
                "implementations": len(implementations),
                "evaluations": len(evaluations),
                "correlations": len(correlations),
            }},
        )

        # ── 2. Build chunk_id → entity lookup (for evidence_entity_id resolution) ──
        chunk_to_entity: dict[str, dict[str, Any]] = {}
        for entity in requirements + implementations + evaluations:
            cid = entity.get("chunk_id")
            if cid:
                chunk_to_entity[cid] = entity

        # ── 3. Merge entity nodes ──────────────────────────────────────────────
        stats.requirement_nodes = self._merge_nodes(
            requirements, "Requirement"
        )
        stats.implementation_nodes = self._merge_nodes(
            implementations, "Implementation"
        )
        stats.evaluation_nodes = self._merge_nodes(
            evaluations, "Evaluation"
        )

        # ── 4. Merge SourceDocument nodes ──────────────────────────────────────
        all_entities = requirements + implementations + evaluations
        source_docs: dict[str, str] = {}  # name → source_type
        for entity in all_entities:
            name = entity.get("source_document", "")
            if name and name not in source_docs:
                source_docs[name] = _infer_source_type(name)

        stats.source_document_nodes = self._merge_source_documents(source_docs)

        # ── 5. EXTRACTED_FROM relationships ────────────────────────────────────
        stats.extracted_from_rels = self._merge_extracted_from(all_entities)

        # ── 6. CORRELATES_TO relationships (from correlation table) ────────────
        corr_stats = self._merge_correlates_to(correlations, chunk_to_entity, stats)
        stats.correlates_to_rels = corr_stats

        # ── 7. EXPLICITLY_LINKS relationships (from linked_requirement field) ───
        stats.explicitly_links_rels = self._merge_explicitly_links(implementations)

        # ── 8. EVALUATES relationships ─────────────────────────────────────────
        stats.evaluates_rels = self._merge_evaluates(evaluations)

        logger.info(
            "Graph build complete",
            extra={"context": stats.to_dict()},
        )
        return stats

    # ── Node MERGE helpers ────────────────────────────────────────────────────

    def _merge_nodes(self, entities: list[dict[str, Any]], label: str) -> int:
        """
        MERGE a batch of entity nodes with the given label.

        Uses UNWIND for O(1) network round trips regardless of entity count.
        SET overwrites all mutable properties on each run (idempotent update).
        """
        if not entities:
            return 0

        cypher = f"""
        UNWIND $batch AS row
        MERGE (n:{label} {{entity_id: row.entity_id}})
        SET
            n.text             = row.text,
            n.linked_requirement = row.linked_requirement,
            n.source_document  = row.source_document,
            n.chunk_id         = row.chunk_id,
            n.author           = row.author,
            n.timestamp        = row.timestamp,
            n.created_at       = row.created_at
        """

        batch = [
            {
                "entity_id":          e.get("entity_id", ""),
                "text":               e.get("text", ""),
                "linked_requirement": e.get("linked_requirement"),
                "source_document":    e.get("source_document", ""),
                "chunk_id":           e.get("chunk_id", ""),
                "author":             e.get("author"),
                "timestamp":          e.get("timestamp"),
                "created_at":         str(e.get("created_at", "")),
            }
            for e in entities
            if e.get("entity_id")  # Guard against empty entity_ids
        ]

        self._run_batched(cypher, batch)
        logger.debug(
            "Node MERGE batch complete",
            extra={"context": {"label": label, "count": len(batch)}},
        )
        return len(batch)

    def _merge_source_documents(self, source_docs: dict[str, str]) -> int:
        """MERGE SourceDocument nodes (one per unique source filename)."""
        if not source_docs:
            return 0

        cypher = """
        UNWIND $batch AS row
        MERGE (d:SourceDocument {name: row.name})
        SET d.source_type = row.source_type
        """

        batch = [
            {"name": name, "source_type": stype}
            for name, stype in source_docs.items()
            if name
        ]
        self._run_batched(cypher, batch)
        logger.debug(
            "SourceDocument MERGE complete",
            extra={"context": {"count": len(batch)}},
        )
        return len(batch)

    # ── Relationship MERGE helpers ────────────────────────────────────────────

    def _merge_extracted_from(self, entities: list[dict[str, Any]]) -> int:
        """
        MERGE EXTRACTED_FROM relationships: every entity → its source document.

        This provides the provenance trail: entity → source file.
        """
        cypher = """
        UNWIND $batch AS row
        MATCH (n {entity_id: row.entity_id})
        MATCH (d:SourceDocument {name: row.source_document})
        MERGE (n)-[:EXTRACTED_FROM]->(d)
        """

        batch = [
            {
                "entity_id":       e.get("entity_id", ""),
                "source_document": e.get("source_document", ""),
            }
            for e in entities
            if e.get("entity_id") and e.get("source_document")
        ]
        self._run_batched(cypher, batch)
        logger.debug(
            "EXTRACTED_FROM MERGE complete",
            extra={"context": {"count": len(batch)}},
        )
        return len(batch)

    def _merge_correlates_to(
        self,
        correlations: list[dict[str, Any]],
        chunk_to_entity: dict[str, dict[str, Any]],
        stats: BuildStats,
    ) -> int:
        """
        MERGE CORRELATES_TO relationships from the correlation table.

        Handles the dual-mode evidence_entity_id:
        - If it looks like a chunk_id (64 hex chars), resolve to the owning entity.
        - If it is "(none)", skip (REQUIREMENT_NOT_IMPLEMENTED has no target node).
        """
        batch = []

        for corr in correlations:
            req_id = corr.get("requirement_entity_id", "")
            raw_ev_id = corr.get("evidence_entity_id", "")

            # Skip sentinel "(none)" — no implementation exists for this requirement
            if not raw_ev_id or raw_ev_id in ("(none)", "(unlinked)", "(unknown)"):
                stats.skipped_none_evidence += 1
                continue

            # Resolve chunk_id → entity_id if needed
            ev_id = raw_ev_id
            if _is_chunk_id(raw_ev_id):
                owning_entity = chunk_to_entity.get(raw_ev_id)
                if owning_entity:
                    ev_id = owning_entity["entity_id"]
                    stats.chunk_id_resolutions += 1
                    logger.debug(
                        "evidence_entity_id resolved from chunk_id to entity_id",
                        extra={"context": {
                            "chunk_id": raw_ev_id[:16] + "…",
                            "resolved_to": ev_id,
                        }},
                    )
                else:
                    # Chunk exists in ChromaDB but was never extracted into an entity.
                    # Skip this edge — we cannot link to a non-existent node.
                    msg = (
                        f"chunk_id '{raw_ev_id[:16]}…' in correlation "
                        f"'{corr.get('correlation_id', '')[:8]}' has no owning entity — "
                        f"edge skipped"
                    )
                    stats.warnings.append(msg)
                    logger.warning(msg)
                    continue

            import json
            chunk_ids_raw = corr.get("supporting_chunk_ids", "[]") or "[]"
            try:
                supporting_chunks = json.loads(chunk_ids_raw)
            except (json.JSONDecodeError, TypeError):
                supporting_chunks = []

            batch.append({
                "req_id":           req_id,
                "ev_id":            ev_id,
                "correlation_id":   corr.get("correlation_id", ""),
                "status":           corr.get("status", ""),
                "confidence":       float(corr.get("confidence", 0.0)),
                "resolution_method": corr.get("resolution_method", ""),
                "rule_name":        corr.get("rule_name"),
                "reasoning":        corr.get("reasoning", ""),
                "created_at":       str(corr.get("created_at", "")),
                "supporting_chunk_ids": supporting_chunks,
            })

        if not batch:
            return 0

        # MATCH both endpoints — if either node is missing, skip (MERGE would create
        # orphan nodes, which would pollute the graph).
        cypher = """
        UNWIND $batch AS row
        MATCH (req:Requirement {entity_id: row.req_id})
        MATCH (ev {entity_id: row.ev_id})
        MERGE (req)-[r:CORRELATES_TO {correlation_id: row.correlation_id}]->(ev)
        SET
            r.status             = row.status,
            r.confidence         = row.confidence,
            r.resolution_method  = row.resolution_method,
            r.rule_name          = row.rule_name,
            r.reasoning          = row.reasoning,
            r.created_at         = row.created_at
        """

        self._run_batched(cypher, batch)
        logger.debug(
            "CORRELATES_TO MERGE complete",
            extra={"context": {"count": len(batch)}},
        )
        return len(batch)

    def _merge_explicitly_links(self, implementations: list[dict[str, Any]]) -> int:
        """
        MERGE EXPLICITLY_LINKS: Implementation → Requirement.

        Created only when Stage 1 extraction set linked_requirement on an
        IMPLEMENTATION row — i.e., the LLM found an explicit mention of the
        requirement ID in the source text.
        """
        batch = [
            {
                "impl_id": impl["entity_id"],
                "req_id":  impl["linked_requirement"],
            }
            for impl in implementations
            if impl.get("entity_id") and impl.get("linked_requirement")
        ]

        if not batch:
            return 0

        cypher = """
        UNWIND $batch AS row
        MATCH (impl:Implementation {entity_id: row.impl_id})
        MATCH (req:Requirement {entity_id: row.req_id})
        MERGE (impl)-[:EXPLICITLY_LINKS {source: "stage1_extraction"}]->(req)
        """

        self._run_batched(cypher, batch)
        logger.debug(
            "EXPLICITLY_LINKS MERGE complete",
            extra={"context": {"count": len(batch)}},
        )
        return len(batch)

    def _merge_evaluates(self, evaluations: list[dict[str, Any]]) -> int:
        """
        MERGE EVALUATES: Evaluation → Requirement.

        Created only for evaluations that have a linked_requirement.
        Orphan evaluations (no linked_requirement) are left as isolated nodes;
        the UI hides them by default.
        """
        batch = [
            {
                "eval_id": ev["entity_id"],
                "req_id":  ev["linked_requirement"],
            }
            for ev in evaluations
            if ev.get("entity_id") and ev.get("linked_requirement")
        ]

        if not batch:
            return 0

        cypher = """
        UNWIND $batch AS row
        MATCH (ev:Evaluation {entity_id: row.eval_id})
        MATCH (req:Requirement {entity_id: row.req_id})
        MERGE (ev)-[:EVALUATES {source: "stage1_extraction"}]->(req)
        """

        self._run_batched(cypher, batch)
        logger.debug(
            "EVALUATES MERGE complete",
            extra={"context": {"count": len(batch)}},
        )
        return len(batch)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_batched(self, cypher: str, batch: list[dict[str, Any]]) -> None:
        """
        Execute a Cypher UNWIND query in chunks to stay within Neo4j's
        default transaction memory limit.

        The UNWIND pattern means each call is a single network round trip
        per chunk of _BATCH_SIZE rows.
        """
        for i in range(0, len(batch), _BATCH_SIZE):
            chunk = batch[i: i + _BATCH_SIZE]
            self._session.run(cypher, batch=chunk)
