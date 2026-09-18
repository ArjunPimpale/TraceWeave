"""Graph edges retain provenance and bounded views retain their anchors."""

from __future__ import annotations

from pecs.graph.projection import build_projection, evidence_key, source_key
from pecs.graph.views import select_view


def _snapshot():
    return {
        "dataset_uuid": "data", "requirements": [{"row_id": 1, "entity_id": "R1", "text": "Train",
            "source_document": "spec.pdf", "status": "IMPLEMENTED_WITHOUT_EVALUATION",
            "workflow_state": "assessed", "confidence": 0.8, "assessment_ids": [],
            "diagnostics": [], "conflicting": False}],
        "evidence": {
            1: {"id": 1, "entity_type": "REQUIREMENT", "entity_id": "R1", "text": "Train",
                "source_document": "spec.pdf", "chunk_id": "req"},
            2: {"id": 2, "entity_type": "IMPLEMENTATION", "entity_id": "same", "text": "Build A",
                "source_document": "same.py", "chunk_id": "a", "linked_requirement": "R1"},
            3: {"id": 3, "entity_type": "IMPLEMENTATION", "entity_id": "same", "text": "Build B",
                "source_document": "same.py", "chunk_id": "b", "linked_requirement": "R1"},
        },
        "chunks": {
            "a": {"chunk_id": "a", "snapshot_id": "sa", "source_hash": "hash-a",
                  "source_document": "same.py", "source_type": "PYTHON", "source_locator": "lines 1-5",
                  "normalized_text": "def a(): pass", "metadata_json": "{}", "availability": "available"},
            "b": {"chunk_id": "b", "snapshot_id": "sb", "source_hash": "hash-b",
                  "source_document": "same.py", "source_type": "PYTHON", "source_locator": "lines 6-10",
                  "normalized_text": "def b(): pass", "metadata_json": "{}", "availability": "available"},
        },
        "associations": [
            {"evidence_row_id": 2, "requirement_row_id": 1, "kind": "REFERENCES_REQUIREMENT"},
            {"evidence_row_id": 3, "requirement_row_id": 1, "kind": "REFERENCES_REQUIREMENT"},
        ],
        "correlations": {}, "candidates": {},
    }


def test_duplicate_descriptive_ids_and_same_filename_do_not_merge():
    graph = build_projection(_snapshot())
    assert evidence_key("data", 2) in graph["nodes"]
    assert evidence_key("data", 3) in graph["nodes"]
    assert source_key("data", _snapshot()["chunks"]["a"]) != source_key("data", _snapshot()["chunks"]["b"])
    direct = [e for e in graph["edges"].values() if e["kind"] == "REFERENCES_REQUIREMENT"]
    assert len(direct) == 2
    assert {e["source"] for e in direct} == {evidence_key("data", 2), evidence_key("data", 3)}


def test_provenance_edges_point_to_the_actual_source_and_filter_does_not_reclassify():
    graph = build_projection(_snapshot())
    anchor = evidence_key("data", 1)
    view = select_view(graph, requirement_ids=[anchor], show_provenance=True)
    from_a = [e for e in view["edges"] if e["source"] == evidence_key("data", 2)
              and e["kind"] == "EXTRACTED_FROM"]
    assert len(from_a) == 1
    assert from_a[0]["target"].startswith("chunk:data:a:")
    filtered = select_view(graph, requirement_ids=[anchor], show_evaluations=False,
                           source_filter="no-such-file")
    assert filtered["nodes"][0]["status"] == "IMPLEMENTED_WITHOUT_EVALUATION"


def test_dense_view_caps_payload_with_closed_edges():
    snapshot = _snapshot()
    for n in range(4, 304):
        snapshot["evidence"][n] = {"id": n, "entity_type": "IMPLEMENTATION",
            "entity_id": f"impl-{n}", "text": f"Build item {n}", "source_document": "a.py",
            "chunk_id": "a", "linked_requirement": "R1"}
        snapshot["associations"].append({"evidence_row_id": n, "requirement_row_id": 1,
                                          "kind": "REFERENCES_REQUIREMENT"})
    graph = build_projection(snapshot)
    anchor = evidence_key("data", 1)
    view = select_view(graph, requirement_ids=[anchor])
    visible = {node["id"] for node in view["nodes"]}
    assert anchor in visible
    assert len(visible) <= 40
    assert view["counts"]["collapsed"] > 200
    assert all(e["source"] in visible and e["target"] in visible for e in view["edges"])


def test_source_group_expansion_recovers_hidden_evidence_without_losing_anchor():
    snapshot = _snapshot()
    for n in range(4, 64):
        snapshot["evidence"][n] = {"id": n, "entity_type": "IMPLEMENTATION",
            "entity_id": f"impl-{n}", "text": f"Build item {n}", "source_document": "same.py",
            "chunk_id": "a", "linked_requirement": "R1"}
        snapshot["associations"].append({"evidence_row_id": n, "requirement_row_id": 1,
                                          "kind": "REFERENCES_REQUIREMENT"})
    graph = build_projection(snapshot)
    anchor = evidence_key("data", 1)
    first = select_view(graph, requirement_ids=[anchor], source_group="hash-a", group_page=1)
    expanded = select_view(graph, requirement_ids=[anchor], source_group="hash-a", group_page=2)
    first_ids = {node["id"] for node in first["nodes"]}
    expanded_ids = {node["id"] for node in expanded["nodes"]}
    assert anchor in first_ids
    assert len(first_ids) == 21
    assert first_ids < expanded_ids
    assert first["groups"]["hash-a"]["count"] == 61
    assert expanded["counts"]["collapsed"] < first["counts"]["collapsed"]


def test_relationship_filter_keeps_requirement_without_misleading_evidence():
    graph = build_projection(_snapshot())
    anchor = evidence_key("data", 1)
    view = select_view(graph, requirement_ids=[anchor], relationship_kinds={"ASSESSMENT_CONTEXT"})
    assert [node["id"] for node in view["nodes"]] == [anchor]
    assert view["edges"] == []
    assert view["nodes"][0]["status"] == "IMPLEMENTED_WITHOUT_EVALUATION"


def test_edge_retains_each_backing_citation_record():
    snapshot = _snapshot()
    snapshot["correlations"] = {"corr-1": {"correlation_id": "corr-1", "requirement_row_id": 1,
        "references": [
            {"reference_id": "ref-a", "target_kind": "evidence", "evidence_row_id": 2,
             "role": "explicit_implementation"},
            {"reference_id": "ref-b", "target_kind": "evidence", "evidence_row_id": 2,
             "role": "explicit_implementation"},
        ]}}
    graph = build_projection(snapshot)
    edge = next(e for e in graph["edges"].values() if e["source"] == evidence_key("data", 2)
                and e["kind"] == "REFERENCES_REQUIREMENT")
    assert edge["assessment_ids"] == ["corr-1"]
    assert edge["reference_ids"] == ["ref-a", "ref-b"]


def test_unlinked_reason_distinguishes_missing_and_ambiguous_reference():
    snapshot = _snapshot()
    snapshot["evidence"][4] = {"id": 4, "entity_type": "EVALUATION", "entity_id": "E4",
        "text": "No link", "source_document": "report.txt", "chunk_id": "x",
        "linked_requirement": None}
    snapshot["evidence"][5] = {"id": 5, "entity_type": "EVALUATION", "entity_id": "E5",
        "text": "Unknown link", "source_document": "report.txt", "chunk_id": "x",
        "linked_requirement": "R-missing"}
    snapshot["evidence"][6] = {"id": 6, "entity_type": "EVALUATION", "entity_id": "E6",
        "text": "Ambiguous link", "source_document": "report.txt", "chunk_id": "x",
        "linked_requirement": "R-duplicate"}
    snapshot["associations"] += [
        {"evidence_row_id": 5, "requirement_row_id": None, "kind": "ASSESSES_REQUIREMENT",
         "unresolved_value": "R-missing", "candidate_ids": []},
        {"evidence_row_id": 6, "requirement_row_id": None, "kind": "ASSESSES_REQUIREMENT",
         "unresolved_value": "R-duplicate", "candidate_ids": [1, 7]},
    ]
    graph = build_projection(snapshot)
    view = select_view(graph, mode="unlinked")
    reasons = {node["id"]: node["details"]["unlinked_reason"] for node in view["nodes"]}
    assert reasons[evidence_key("data", 4)] == "No extracted requirement reference"
    assert reasons[evidence_key("data", 5)] == "Requirement reference target is missing"
    assert reasons[evidence_key("data", 6)] == "Requirement reference is ambiguous"
