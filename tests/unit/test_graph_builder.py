"""
Unit tests for GraphBuilder.

All tests mock the Neo4j session — no real Neo4j instance required.
Tests assert that the builder produces the correct Cypher calls with the
correct parameters for a variety of input scenarios.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from pecs.graph.graph_builder import GraphBuilder, BuildStats, _is_chunk_id, _infer_source_type


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_requirement(entity_id: str, text: str = "A requirement", doc: str = "spec.pdf") -> dict:
    return {
        "entity_id": entity_id,
        "text": text,
        "linked_requirement": None,
        "source_document": doc,
        "chunk_id": "a" * 64,
        "author": None,
        "timestamp": None,
        "created_at": "2024-01-01T00:00:00",
    }


def _make_implementation(
    entity_id: str,
    linked_requirement: str | None = None,
    doc: str = "train.py",
    chunk_id: str | None = None,
) -> dict:
    return {
        "entity_id": entity_id,
        "text": "Implementation of something",
        "linked_requirement": linked_requirement,
        "source_document": doc,
        "chunk_id": chunk_id or ("b" * 64),
        "author": None,
        "timestamp": None,
        "created_at": "2024-01-01T00:00:00",
    }


def _make_evaluation(entity_id: str, linked_requirement: str | None = None) -> dict:
    return {
        "entity_id": entity_id,
        "text": "Professor feedback: excellent",
        "linked_requirement": linked_requirement,
        "source_document": "feedback.pdf",
        "chunk_id": "c" * 64,
        "author": "Prof. X",
        "timestamp": None,
        "created_at": "2024-01-01T00:00:00",
    }


def _make_correlation(
    req_id: str,
    ev_id: str,
    status: str = "IMPLEMENTED_AND_VALIDATED",
    confidence: float = 0.92,
    resolution_method: str = "deterministic_rule",
    rule_name: str | None = "exact_requirement_id_match",
) -> dict:
    import json, uuid
    return {
        "correlation_id": str(uuid.uuid4()),
        "requirement_entity_id": req_id,
        "evidence_entity_id": ev_id,
        "status": status,
        "confidence": confidence,
        "resolution_method": resolution_method,
        "rule_name": rule_name,
        "reasoning": "Rule fired.",
        "created_at": "2024-01-01T00:00:00",
        "supporting_chunk_ids": json.dumps(["b" * 64]),
    }


def _make_builder(requirements=None, implementations=None, evaluations=None, correlations=None):
    """Build a GraphBuilder with mocked repos and a mock Neo4j session."""
    session = MagicMock()

    ev_repo = MagicMock()
    ev_repo.get_all_requirements.return_value = requirements or []
    ev_repo.get_all_implementations.return_value = implementations or []
    ev_repo.get_all_evaluations.return_value = evaluations or []
    ev_repo.get_by_chunk_id.return_value = []

    corr_repo = MagicMock()
    corr_repo.get_all_latest.return_value = correlations or []

    builder = GraphBuilder(session=session, evidence_repo=ev_repo, correlation_repo=corr_repo)
    return builder, session


# ── Helper function tests ─────────────────────────────────────────────────────

class TestIsChunkId:
    def test_valid_chunk_id(self):
        assert _is_chunk_id("a" * 64) is True
        assert _is_chunk_id("0" * 64) is True
        # SHA-256 hex
        assert _is_chunk_id("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") is True

    def test_entity_id_not_chunk_id(self):
        assert _is_chunk_id("R3-gnn-training") is False
        assert _is_chunk_id("impl-train-gnn") is False
        assert _is_chunk_id("(none)") is False

    def test_short_hex_not_chunk_id(self):
        assert _is_chunk_id("abcdef") is False

    def test_uppercase_hex_not_chunk_id(self):
        # SHA-256 produces lowercase; uppercase should not match
        assert _is_chunk_id("A" * 64) is False


class TestInferSourceType:
    def test_pdf(self):
        assert _infer_source_type("spec.pdf") == "PDF"

    def test_python(self):
        assert _infer_source_type("train.py") == "PYTHON"

    def test_docx(self):
        assert _infer_source_type("report.docx") == "DOCX"

    def test_whatsapp(self):
        assert _infer_source_type("whatsapp_chat.txt") == "WHATSAPP"

    def test_git_heuristic(self):
        assert _infer_source_type("git_log.txt") == "GIT"

    def test_unknown(self):
        assert _infer_source_type("noextension") == "UNKNOWN"


# ── GraphBuilder tests ────────────────────────────────────────────────────────

class TestRequirementNodes:
    def test_requirements_become_requirement_nodes(self):
        """Three requirement rows → three MERGE calls with :Requirement label."""
        reqs = [_make_requirement(f"R{i}") for i in range(3)]
        builder, session = _make_builder(requirements=reqs)
        stats = builder.build()

        assert stats.requirement_nodes == 3
        # session.run should have been called with :Requirement in cypher
        calls = session.run.call_args_list
        req_calls = [c for c in calls if "Requirement" in str(c)]
        assert len(req_calls) > 0

    def test_empty_requirements_produces_no_node_calls_for_that_type(self):
        builder, session = _make_builder(requirements=[])
        stats = builder.build()
        assert stats.requirement_nodes == 0

    def test_entity_id_used_as_merge_key(self):
        """Verify entity_id is passed as the MERGE key in the batch."""
        req = _make_requirement("R-unique-id")
        builder, session = _make_builder(requirements=[req])
        builder.build()

        # Find the call where the batch contains our entity_id
        found = False
        for c in session.run.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            args = c.args
            # batch is passed as keyword arg
            batch = kwargs.get("batch") or (args[1] if len(args) > 1 else None)
            if batch and any(row.get("entity_id") == "R-unique-id" for row in batch if isinstance(row, dict)):
                found = True
                break
        assert found, "entity_id 'R-unique-id' not found in any session.run batch"


class TestImplementationAndEvaluationNodes:
    def test_implementations_become_implementation_nodes(self):
        impls = [_make_implementation(f"impl-{i}") for i in range(2)]
        builder, session = _make_builder(implementations=impls)
        stats = builder.build()
        assert stats.implementation_nodes == 2

    def test_evaluations_become_evaluation_nodes(self):
        evals = [_make_evaluation(f"eval-{i}") for i in range(4)]
        builder, session = _make_builder(evaluations=evals)
        stats = builder.build()
        assert stats.evaluation_nodes == 4


class TestSourceDocumentNodes:
    def test_source_documents_deduplicated(self):
        """Five entities from two source documents → two SourceDocument nodes."""
        reqs = [_make_requirement(f"R{i}", doc="spec.pdf") for i in range(3)]
        impls = [_make_implementation(f"impl-{i}", doc="train.py") for i in range(2)]
        builder, session = _make_builder(requirements=reqs, implementations=impls)
        stats = builder.build()
        assert stats.source_document_nodes == 2

    def test_all_same_doc_produces_one_source_document_node(self):
        reqs = [_make_requirement(f"R{i}", doc="spec.pdf") for i in range(5)]
        impls = [_make_implementation(f"impl-{i}", doc="spec.pdf") for i in range(3)]
        builder, session = _make_builder(requirements=reqs, implementations=impls)
        stats = builder.build()
        assert stats.source_document_nodes == 1


class TestCorrelatesTo:
    def test_correlation_creates_edge(self):
        """A CorrelationResult linking req A to impl B creates a CORRELATES_TO edge."""
        req = _make_requirement("R1")
        impl = _make_implementation("impl-1", linked_requirement="R1")
        corr = _make_correlation("R1", "impl-1")

        builder, session = _make_builder(
            requirements=[req],
            implementations=[impl],
            correlations=[corr],
        )
        stats = builder.build()
        assert stats.correlates_to_rels == 1

    def test_none_evidence_id_skipped(self):
        """A CorrelationResult with evidence_entity_id='(none)' produces no edge."""
        req = _make_requirement("R1")
        corr = _make_correlation("R1", "(none)", status="REQUIREMENT_NOT_IMPLEMENTED")

        builder, session = _make_builder(
            requirements=[req],
            correlations=[corr],
        )
        stats = builder.build()
        assert stats.correlates_to_rels == 0
        assert stats.skipped_none_evidence == 1

    def test_unlinked_sentinel_skipped(self):
        """evidence_entity_id='(unlinked)' is also skipped."""
        req = _make_requirement("R1")
        corr = _make_correlation("R1", "(unlinked)")
        builder, session = _make_builder(requirements=[req], correlations=[corr])
        stats = builder.build()
        assert stats.correlates_to_rels == 0
        assert stats.skipped_none_evidence == 1

    def test_chunk_id_as_evidence_entity_id_resolved(self):
        """
        When evidence_entity_id is a 64-char hex (chunk_id), the builder
        resolves it to the owning entity's entity_id.
        """
        chunk_id = "e" * 64
        req = _make_requirement("R1")
        impl = _make_implementation("impl-from-chunk", doc="train.py", chunk_id=chunk_id)
        # The correlation references the chunk_id, not the entity_id
        corr = _make_correlation("R1", chunk_id)

        builder, session = _make_builder(
            requirements=[req],
            implementations=[impl],
            correlations=[corr],
        )
        stats = builder.build()
        # chunk_id should have been resolved → no warning, 1 edge
        assert stats.chunk_id_resolutions == 1
        assert stats.correlates_to_rels == 1

    def test_unresolvable_chunk_id_produces_warning_not_error(self):
        """
        When a chunk_id cannot be resolved to an entity, the edge is skipped
        with a warning (no exception raised).
        """
        chunk_id = "f" * 64  # Not in any entity's chunk_id
        req = _make_requirement("R1")
        corr = _make_correlation("R1", chunk_id)

        builder, session = _make_builder(
            requirements=[req],
            correlations=[corr],
        )
        stats = builder.build()
        assert stats.correlates_to_rels == 0
        assert len(stats.warnings) == 1
        assert "edge skipped" in stats.warnings[0]


class TestExplicitLinks:
    def test_linked_requirement_creates_explicit_link(self):
        """
        An Implementation with linked_requirement='R3' gets an EXPLICITLY_LINKS edge.
        """
        req = _make_requirement("R3")
        impl = _make_implementation("impl-x", linked_requirement="R3")
        builder, session = _make_builder(requirements=[req], implementations=[impl])
        stats = builder.build()
        assert stats.explicitly_links_rels == 1

    def test_implementation_without_linked_requirement_has_no_explicit_link(self):
        impl = _make_implementation("impl-y", linked_requirement=None)
        builder, session = _make_builder(implementations=[impl])
        stats = builder.build()
        assert stats.explicitly_links_rels == 0


class TestEvaluatesRelationships:
    def test_evaluation_with_linked_requirement_creates_evaluates_edge(self):
        req = _make_requirement("R2")
        ev = _make_evaluation("eval-a", linked_requirement="R2")
        builder, session = _make_builder(requirements=[req], evaluations=[ev])
        stats = builder.build()
        assert stats.evaluates_rels == 1

    def test_orphan_evaluation_produces_no_evaluates_edge(self):
        """Evaluation with no linked_requirement → isolated node, no EVALUATES edge."""
        ev = _make_evaluation("eval-orphan", linked_requirement=None)
        builder, session = _make_builder(evaluations=[ev])
        stats = builder.build()
        assert stats.evaluates_rels == 0


class TestIdempotency:
    def test_build_stats_consistent_on_same_data(self):
        """
        Running build() with the same input data twice produces the same
        stats. Because we use MERGE, the second run is a no-op at the DB level.
        """
        req = _make_requirement("R1")
        impl = _make_implementation("impl-1", linked_requirement="R1")
        corr = _make_correlation("R1", "impl-1")

        builder, session = _make_builder(
            requirements=[req],
            implementations=[impl],
            correlations=[corr],
        )
        stats1 = builder.build()
        # Reset the session call count but keep same data
        session.run.reset_mock()
        stats2 = builder.build()

        assert stats1.requirement_nodes == stats2.requirement_nodes
        assert stats1.implementation_nodes == stats2.implementation_nodes
        assert stats1.correlates_to_rels == stats2.correlates_to_rels


class TestBuildStats:
    def test_total_nodes_sums_all_types(self):
        stats = BuildStats(
            requirement_nodes=3,
            implementation_nodes=5,
            evaluation_nodes=2,
            source_document_nodes=4,
        )
        assert stats.total_nodes == 14

    def test_total_relationships_sums_all_types(self):
        stats = BuildStats(
            correlates_to_rels=10,
            explicitly_links_rels=3,
            evaluates_rels=2,
            extracted_from_rels=15,
        )
        assert stats.total_relationships == 30

    def test_to_dict_structure(self):
        stats = BuildStats(requirement_nodes=2, correlates_to_rels=1)
        d = stats.to_dict()
        assert "nodes" in d
        assert "relationships" in d
        assert d["nodes"]["Requirement"] == 2
        assert d["relationships"]["CORRELATES_TO"] == 1
