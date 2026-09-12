"""
Unit tests for graph_queries.py.

Tests verify that:
  - Query functions return the correct data structure types.
  - Query functions handle client failures gracefully (return empty, no exception).
  - The Cypher templates contain the expected parameters.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pecs.graph import graph_queries


def _make_client(return_value=None, raise_exc=None):
    """Return a mocked Neo4jClient."""
    client = MagicMock()
    if raise_exc:
        client.run_read.side_effect = raise_exc
        client.get_graph_stats.side_effect = raise_exc
    else:
        client.run_read.return_value = return_value or []
        client.get_graph_stats.return_value = {}
    return client


# ── get_requirement_overview ──────────────────────────────────────────────────

class TestGetRequirementOverview:
    def test_returns_list(self):
        client = _make_client(return_value=[{"entity_id": "R1", "text": "Req 1"}])
        result = graph_queries.get_requirement_overview(client)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["entity_id"] == "R1"

    def test_returns_empty_list_on_failure(self):
        client = _make_client(raise_exc=Exception("connection error"))
        result = graph_queries.get_requirement_overview(client)
        assert result == []

    def test_run_read_called_once(self):
        client = _make_client()
        graph_queries.get_requirement_overview(client)
        client.run_read.assert_called_once()


# ── get_requirement_ego_network ───────────────────────────────────────────────

class TestGetRequirementEgoNetwork:
    def test_returns_dict(self):
        client = _make_client(return_value=[{"requirement": {}, "correlations": []}])
        result = graph_queries.get_requirement_ego_network(client, "R1")
        assert isinstance(result, dict)

    def test_returns_empty_dict_when_not_found(self):
        client = _make_client(return_value=[])
        result = graph_queries.get_requirement_ego_network(client, "R-missing")
        assert result == {}

    def test_returns_empty_dict_on_failure(self):
        client = _make_client(raise_exc=RuntimeError("bolt error"))
        result = graph_queries.get_requirement_ego_network(client, "R1")
        assert result == {}

    def test_entity_id_passed_as_parameter(self):
        client = _make_client(return_value=[{}])
        graph_queries.get_requirement_ego_network(client, "R-target")
        call_kwargs = client.run_read.call_args
        assert call_kwargs.kwargs.get("entity_id") == "R-target"


# ── get_shared_artifacts ──────────────────────────────────────────────────────

class TestGetSharedArtifacts:
    def test_returns_list(self):
        client = _make_client(return_value=[{"req1_id": "R1", "req2_id": "R2"}])
        result = graph_queries.get_shared_artifacts(client)
        assert isinstance(result, list)

    def test_returns_empty_list_on_failure(self):
        client = _make_client(raise_exc=Exception("timeout"))
        result = graph_queries.get_shared_artifacts(client)
        assert result == []


# ── get_orphan_entities ───────────────────────────────────────────────────────

class TestGetOrphanEntities:
    def test_returns_list(self):
        client = _make_client(return_value=[])
        result = graph_queries.get_orphan_entities(client)
        assert isinstance(result, list)

    def test_returns_empty_on_failure(self):
        client = _make_client(raise_exc=Exception("error"))
        result = graph_queries.get_orphan_entities(client)
        assert result == []


# ── get_source_document_coverage ─────────────────────────────────────────────

class TestGetSourceDocumentCoverage:
    def test_returns_list(self):
        client = _make_client(return_value=[{"entity_type": "Requirement", "entity_id": "R1"}])
        result = graph_queries.get_source_document_coverage(client, "spec.pdf")
        assert isinstance(result, list)

    def test_doc_name_passed_as_parameter(self):
        client = _make_client(return_value=[])
        graph_queries.get_source_document_coverage(client, "my_report.pdf")
        call_kwargs = client.run_read.call_args
        assert call_kwargs.kwargs.get("doc_name") == "my_report.pdf"

    def test_returns_empty_on_failure(self):
        client = _make_client(raise_exc=Exception("error"))
        result = graph_queries.get_source_document_coverage(client, "spec.pdf")
        assert result == []


# ── get_full_graph ────────────────────────────────────────────────────────────

class TestGetFullGraph:
    def test_returns_tuple_of_two_lists(self):
        client = _make_client(return_value=[{"label": "Requirement"}])
        nodes, edges = graph_queries.get_full_graph(client)
        assert isinstance(nodes, list)
        assert isinstance(edges, list)

    def test_returns_empty_tuples_on_failure(self):
        client = _make_client(raise_exc=Exception("connection refused"))
        nodes, edges = graph_queries.get_full_graph(client)
        assert nodes == []
        assert edges == []

    def test_run_read_called_twice(self):
        """get_full_graph makes two queries: one for nodes, one for edges."""
        client = _make_client(return_value=[])
        graph_queries.get_full_graph(client)
        assert client.run_read.call_count == 2


# ── get_graph_stats ───────────────────────────────────────────────────────────

class TestGetGraphStats:
    def test_returns_dict(self):
        client = _make_client()
        client.get_graph_stats.return_value = {"Requirement": 3, "CORRELATES_TO": 5}
        result = graph_queries.get_graph_stats(client)
        assert isinstance(result, dict)
        assert result.get("Requirement") == 3

    def test_returns_empty_dict_on_failure(self):
        client = _make_client(raise_exc=Exception("error"))
        result = graph_queries.get_graph_stats(client)
        assert isinstance(result, dict)


# ── Cypher template content checks ───────────────────────────────────────────

class TestCypherTemplates:
    """Verify that the raw Cypher strings contain expected clauses."""

    def test_requirement_overview_has_match_and_return(self):
        assert "MATCH" in graph_queries._REQUIREMENT_OVERVIEW
        assert "RETURN" in graph_queries._REQUIREMENT_OVERVIEW
        assert "Requirement" in graph_queries._REQUIREMENT_OVERVIEW

    def test_ego_network_uses_entity_id_param(self):
        assert "$entity_id" in graph_queries._REQUIREMENT_EGO_NETWORK

    def test_shared_artifacts_matches_two_requirements(self):
        assert "r1:Requirement" in graph_queries._SHARED_ARTIFACTS
        assert "r2:Requirement" in graph_queries._SHARED_ARTIFACTS

    def test_source_document_coverage_uses_doc_name_param(self):
        assert "$doc_name" in graph_queries._SOURCE_DOCUMENT_COVERAGE

    def test_full_graph_nodes_excludes_source_docs_via_param(self):
        assert "$include_source_docs" in graph_queries._FULL_GRAPH_NODES

    def test_full_graph_edges_returns_confidence(self):
        assert "confidence" in graph_queries._FULL_GRAPH_EDGES

    def test_orphan_query_excludes_source_documents(self):
        assert "SourceDocument" in graph_queries._ORPHAN_ENTITIES
