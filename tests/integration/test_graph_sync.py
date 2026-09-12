"""
Integration tests for GraphSync — require a running Neo4j instance.

These tests are skipped automatically if Neo4j is unavailable, so they
do not block CI/CD pipelines or development environments without Neo4j.

To run locally:
    1. Start Neo4j: docker run -p 7474:7474 -p 7687:7687 neo4j:community
    2. Run: pytest tests/integration/test_graph_sync.py -v -m integration

The tests use an in-memory SQLite database (via a real Database instance
pointing to a temp file) but a real Neo4j connection.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

# ── Neo4j availability guard ──────────────────────────────────────────────────


def _neo4j_available() -> bool:
    """Return True if a local Neo4j instance is reachable."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=None)
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


_SKIP_REASON = "Neo4j not running at localhost:7687"
pytestmark = pytest.mark.skipif(not _neo4j_available(), reason=_SKIP_REASON)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def neo4j_client():
    """Real Neo4jClient for the integration test session."""
    from pecs.graph.neo4j_client import Neo4jClient
    client = Neo4jClient(uri="neo4j://localhost:7687", database="neo4j")
    yield client
    client.close()


@pytest.fixture(scope="module")
def sqlite_repos():
    """Real SQLite repos backed by a temp file, pre-populated with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    from pecs.store.database import Database
    from pecs.store.evidence_repo import EvidenceRepo
    from pecs.store.correlation_repo import CorrelationRepo
    from pecs.models.extraction_result import ExtractionResult, EntityType
    from pecs.models.correlation_result import CorrelationResult, CorrelationStatus

    db = Database(db_path=db_path)
    db.initialize_schema()

    ev_repo = EvidenceRepo(db=db)
    corr_repo = CorrelationRepo(db=db)

    # Insert sample requirements
    for i in range(1, 4):
        ev_repo.insert(ExtractionResult(
            entity_type=EntityType.REQUIREMENT,
            entity_id=f"R{i}-test-req",
            text=f"Requirement {i}: the system must do thing {i}",
            source_document="spec.pdf",
            chunk_id=f"{'a' * 63}{i}",
        ))

    # Insert sample implementations
    ev_repo.insert(ExtractionResult(
        entity_type=EntityType.IMPLEMENTATION,
        entity_id="impl-thing-1",
        text="Implementation of thing 1",
        linked_requirement="R1-test-req",
        source_document="train.py",
        chunk_id="b" * 64,
    ))
    ev_repo.insert(ExtractionResult(
        entity_type=EntityType.IMPLEMENTATION,
        entity_id="impl-thing-2",
        text="Implementation of thing 2 and 3",
        source_document="model.py",
        chunk_id="c" * 64,
    ))

    # Insert sample evaluation
    ev_repo.insert(ExtractionResult(
        entity_type=EntityType.EVALUATION,
        entity_id="eval-prof-feedback",
        text="Excellent implementation. Score: 9/10",
        linked_requirement="R1-test-req",
        source_document="feedback.pdf",
        chunk_id="d" * 64,
        author="Professor Smith",
    ))

    # Insert correlations
    corr_repo.insert(CorrelationResult(
        correlation_id=str(uuid.uuid4()),
        requirement_entity_id="R1-test-req",
        evidence_entity_id="impl-thing-1",
        status=CorrelationStatus.IMPLEMENTED_AND_VALIDATED,
        resolution_method="deterministic_rule",
        rule_name="exact_requirement_id_match",
        confidence=0.95,
        supporting_chunk_ids=["b" * 64, "d" * 64],
        reasoning="Explicit link found.",
    ))
    corr_repo.insert(CorrelationResult(
        correlation_id=str(uuid.uuid4()),
        requirement_entity_id="R2-test-req",
        evidence_entity_id="impl-thing-2",
        status=CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION,
        resolution_method="llm_stage2",
        confidence=0.72,
        supporting_chunk_ids=["c" * 64],
        reasoning="Stage 2 classified as implemented.",
    ))
    corr_repo.insert(CorrelationResult(
        correlation_id=str(uuid.uuid4()),
        requirement_entity_id="R3-test-req",
        evidence_entity_id="(none)",
        status=CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED,
        resolution_method="deterministic_rule",
        rule_name="requirement_with_no_evidence",
        confidence=0.80,
        supporting_chunk_ids=[],
        reasoning="No evidence found.",
    ))

    yield ev_repo, corr_repo

    db.close()
    db_path.unlink(missing_ok=True)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGraphSyncIntegration:
    @pytest.fixture(autouse=True)
    def clear_graph_before_each(self, neo4j_client):
        """Wipe the graph before every test to ensure a clean slate."""
        with neo4j_client.write_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        yield

    def test_sync_creates_requirement_nodes(self, neo4j_client, sqlite_repos):
        """After sync, Neo4j should contain 3 Requirement nodes."""
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read("MATCH (r:Requirement) RETURN count(r) AS cnt")
        assert rows[0]["cnt"] == 3

    def test_sync_creates_implementation_nodes(self, neo4j_client, sqlite_repos):
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read("MATCH (i:Implementation) RETURN count(i) AS cnt")
        assert rows[0]["cnt"] == 2

    def test_sync_creates_evaluation_nodes(self, neo4j_client, sqlite_repos):
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read("MATCH (e:Evaluation) RETURN count(e) AS cnt")
        assert rows[0]["cnt"] == 1

    def test_sync_creates_source_document_nodes(self, neo4j_client, sqlite_repos):
        """3 source files: spec.pdf, train.py, model.py, feedback.pdf → 4 SourceDocument nodes."""
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read("MATCH (d:SourceDocument) RETURN count(d) AS cnt")
        assert rows[0]["cnt"] == 4

    def test_sync_creates_correlates_to_edges(self, neo4j_client, sqlite_repos):
        """
        2 valid correlations (R3 has (none) evidence → skipped) → 2 CORRELATES_TO edges.
        """
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read("MATCH ()-[r:CORRELATES_TO]->() RETURN count(r) AS cnt")
        assert rows[0]["cnt"] == 2

    def test_none_evidence_id_not_in_graph(self, neo4j_client, sqlite_repos):
        """Requirement R3 with '(none)' evidence should have no CORRELATES_TO edge."""
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read(
            "MATCH (r:Requirement {entity_id: 'R3-test-req'})-[:CORRELATES_TO]->(x) RETURN count(x) AS cnt"
        )
        assert rows[0]["cnt"] == 0

    def test_sync_creates_explicitly_links_edge(self, neo4j_client, sqlite_repos):
        """impl-thing-1 has linked_requirement='R1-test-req' → EXPLICITLY_LINKS edge."""
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read(
            "MATCH (i:Implementation {entity_id: 'impl-thing-1'})"
            "-[:EXPLICITLY_LINKS]->(r:Requirement {entity_id: 'R1-test-req'}) "
            "RETURN count(*) AS cnt"
        )
        assert rows[0]["cnt"] == 1

    def test_sync_creates_evaluates_edge(self, neo4j_client, sqlite_repos):
        """eval-prof-feedback is linked to R1-test-req → EVALUATES edge."""
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read(
            "MATCH (e:Evaluation {entity_id: 'eval-prof-feedback'})"
            "-[:EVALUATES]->(r:Requirement) RETURN count(*) AS cnt"
        )
        assert rows[0]["cnt"] == 1

    def test_sync_is_idempotent(self, neo4j_client, sqlite_repos):
        """Running sync twice produces the same node/relationship counts."""
        ev_repo, corr_repo = sqlite_repos
        stats1 = _run_sync(neo4j_client, ev_repo, corr_repo)

        # Clear and rebuild again
        with neo4j_client.write_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        stats2 = _run_sync(neo4j_client, ev_repo, corr_repo)

        assert stats1.total_nodes == stats2.total_nodes
        assert stats1.total_relationships == stats2.total_relationships

    def test_correlates_to_edge_has_confidence_property(self, neo4j_client, sqlite_repos):
        """CORRELATES_TO edges carry the confidence score from CorrelationResult."""
        ev_repo, corr_repo = sqlite_repos
        _run_sync(neo4j_client, ev_repo, corr_repo)

        rows = neo4j_client.run_read(
            "MATCH (:Requirement {entity_id: 'R1-test-req'})"
            "-[r:CORRELATES_TO]->()"
            "RETURN r.confidence AS confidence, r.status AS status"
        )
        assert len(rows) == 1
        assert abs(rows[0]["confidence"] - 0.95) < 0.001
        assert rows[0]["status"] == "IMPLEMENTED_AND_VALIDATED"


# ── Helper ────────────────────────────────────────────────────────────────────

def _run_sync(neo4j_client, ev_repo, corr_repo):
    """Execute a graph sync and return BuildStats."""
    from pecs.graph.graph_builder import GraphBuilder
    from pecs.graph.graph_sync import GraphSync, _CONSTRAINTS

    with neo4j_client.write_session() as session:
        with session.begin_transaction() as tx:
            tx.run("MATCH (n) DETACH DELETE n")
            tx.commit()

        with session.begin_transaction() as tx:
            for stmt in _CONSTRAINTS:
                tx.run(stmt)
            tx.commit()

        with session.begin_transaction() as tx:
            builder = GraphBuilder(
                session=tx,
                evidence_repo=ev_repo,
                correlation_repo=corr_repo,
            )
            stats = builder.build()
            tx.commit()

    return stats
