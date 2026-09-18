"""Graph page reads a selected snapshot without any graph service."""

from __future__ import annotations

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from pecs.ui.pages import graph_page


class Reader:
    def list_snapshots(self):
        return [{"run_id": "run-one", "finished_at": "2026-01-01"}]

    def load(self, run_id):
        return {
            "dataset_uuid": "data", "run_id": run_id, "snapshot_key": "data:run-one:2",
            "legacy": False, "not_in_run": [], "requirements": [{
                "row_id": 1, "entity_id": "R1", "text": "Train a model",
                "source_document": "spec.pdf", "status": "REQUIREMENT_NOT_IMPLEMENTED",
                "workflow_state": "assessed", "confidence": 0.8,
                "assessment_ids": [], "diagnostics": [], "conflicting": False,
            }],
            "evidence": {1: {"id": 1, "entity_type": "REQUIREMENT", "entity_id": "R1",
                             "text": "Train a model", "source_document": "spec.pdf", "chunk_id": "req"}},
            "chunks": {}, "associations": [], "correlations": {}, "candidates": {},
            "unlinked_assessments": [], "outcome": "complete",
        }


def test_page_renders_saved_requirement_without_neo4j():
    with patch.object(graph_page, "TraceabilityReader", Reader), \
         patch.object(graph_page, "render_graph", return_value=None):
            page = AppTest.from_string("from pecs.ui.pages.graph_page import render\nrender()").run()
    assert not page.exception
    assert any("REQUIREMENT_NOT_IMPLEMENTED" in item.value for item in page.markdown)


def test_canvas_selection_event_updates_inspector():
    def event(payload, key):
        return {"event_id": "clicked-once", "snapshot_key": payload["snapshot_key"],
                "view_key": payload["view_key"], "action": "select_node",
                "target_id": payload["nodes"][0]["id"]}

    with patch.object(graph_page, "TraceabilityReader", Reader), \
         patch.object(graph_page, "render_graph", side_effect=event):
        page = AppTest.from_string("from pecs.ui.pages.graph_page import render\nrender()").run()
    assert not page.exception
    assert page.session_state["trace_last_event"] == "clicked-once"


def test_stale_canvas_event_cannot_change_selection():
    def stale(payload, key):
        return {"event_id": "stale-click", "snapshot_key": "old-snapshot",
                "view_key": payload["view_key"], "action": "select_node",
                "target_id": payload["nodes"][0]["id"]}

    with patch.object(graph_page, "TraceabilityReader", Reader), \
         patch.object(graph_page, "render_graph", side_effect=stale):
        page = AppTest.from_string("from pecs.ui.pages.graph_page import render\nrender()").run()
    assert not page.exception
    assert "trace_last_event" not in page.session_state


def test_canvas_background_can_clear_selection():
    def clear(payload, key):
        return {"event_id": "clear-once", "snapshot_key": payload["snapshot_key"],
                "view_key": payload["view_key"], "action": "clear_selection", "target_id": None}

    with patch.object(graph_page, "TraceabilityReader", Reader), \
         patch.object(graph_page, "render_graph", side_effect=clear):
        page = AppTest.from_string("from pecs.ui.pages.graph_page import render\nrender()").run()
    assert not page.exception
    assert page.session_state["trace_last_event"] == "clear-once"
    assert page.session_state["trace_selection"] is None


def test_evaluation_only_dataset_opens_unlinked_view_and_searches_off_canvas():
    class EvaluationReader(Reader):
        def load(self, run_id):
            snapshot = super().load(run_id)
            snapshot["requirements"] = []
            snapshot["evidence"] = {2: {"id": 2, "entity_type": "EVALUATION", "entity_id": "E1",
                "text": "Unique evaluation finding", "source_document": "results.txt",
                "chunk_id": "eval", "linked_requirement": None}}
            return snapshot

    with patch.object(graph_page, "TraceabilityReader", EvaluationReader), \
         patch.object(graph_page, "render_graph", return_value=None):
        page = AppTest.from_string("from pecs.ui.pages.graph_page import render\nrender()")
        page.run()
        assert not page.exception
        assert page.session_state["trace_view_mode"] == "Unlinked evidence"
        page.text_input(key="trace_evidence_search").set_value("Unique evaluation finding").run()
        assert not page.exception
        assert page.selectbox(key="trace_evidence_result").value.startswith("ev:data:2")
