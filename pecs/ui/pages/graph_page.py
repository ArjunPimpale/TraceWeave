"""SQLite-backed traceability explorer controls and session state."""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from pecs.graph.projection import build_projection, evidence_key
from pecs.graph.views import select_view
from pecs.traceability.reader import TraceabilityReader
from pecs.ui.components.graph_details import render_detail
from pecs.ui.components.trace_graph import render_graph


def render() -> None:
    st.title("🕸️ Evidence Graph")
    st.caption("Inspect each requirement's assessment, associated evidence, and source context.")
    history = st.session_state.get("trace_history", [])
    if history and st.button("Back to previous view", key="trace_back"):
        previous = history.pop()
        st.session_state.trace_history = history
        for key, value in previous.items():
            st.session_state[key] = value
        st.rerun()
    try:
        reader = TraceabilityReader()
        runs = reader.list_snapshots()
        run_options = [r["run_id"] for r in runs] + ["legacy"]
        requested = st.session_state.get("trace_run")
        if requested not in run_options:
            requested = run_options[0]
        run_id = st.selectbox("Assessment run", run_options, index=run_options.index(requested),
            format_func=lambda rid: "Legacy latest (limited citations)" if rid == "legacy" else
            f"{rid[:8]} · {next(r['finished_at'] for r in runs if r['run_id'] == rid)}",
            key="graph_run_select")
        st.session_state.trace_run = run_id
        snapshot = reader.load(run_id)
    except Exception as exc:
        st.error(f"Could not read traceability data: {exc}")
        return
    if snapshot["legacy"]:
        st.info("Legacy correlations have no reliable run boundary or full participant record.")
        if st.button("Recover available legacy source context", key="trace_hydrate_legacy"):
            try:
                from pecs.vectorstore.chroma_store import ChromaStore
                added = reader.repo.hydrate_legacy(ChromaStore())
                st.success(f"Captured {added} source citation(s)")
                st.rerun()
            except Exception as exc:
                st.warning(f"Source context could not be recovered: {exc}")
    if not snapshot["requirements"] and not snapshot["evidence"]:
        st.info("No extracted evidence yet. Ingest and extract sources to start tracing.")
        return
    if snapshot["not_in_run"]:
        st.caption(f"{len(snapshot['not_in_run'])} requirement(s) were extracted after this run.")
    projection = build_projection(snapshot)
    requirements = snapshot["requirements"]
    by_key = {evidence_key(snapshot["dataset_uuid"], r["row_id"]): r for r in requirements}
    if not requirements and st.session_state.get("trace_view_mode") != "Unlinked evidence":
        st.session_state.trace_view_mode = "Unlinked evidence"
    if st.session_state.pop("trace_clear_filters", False):
        st.session_state.trace_status_filter = []
        st.session_state.trace_requirement_search = ""
        st.session_state.trace_confidence_filter = 0.0
        st.session_state.trace_source_filter = ""
        st.session_state.trace_source_group = ""
        st.session_state.trace_relationship_filter = [
            "REFERENCES_REQUIREMENT", "ASSESSES_REQUIREMENT", "ASSESSMENT_CONTEXT", "RETRIEVED_CANDIDATE"]
    if next_mode := st.session_state.pop("trace_next_mode", None):
        st.session_state.trace_view_mode = next_mode
        if next_mode == "Requirement focus" and (next_focus := st.session_state.pop("trace_next_focus", None)):
            st.session_state.trace_focus_choice = next_focus
            st.session_state.trace_focus_select = next_focus
        if next_mode == "Compare requirements":
            st.session_state.trace_compare_select = st.session_state.pop("trace_next_compare", [])
    mode = st.radio("View", ["Requirement focus", "Compare requirements", "Unlinked evidence"],
                    horizontal=True, key="trace_view_mode")
    anchors: list[str] = []
    if mode != "Unlinked evidence" and requirements:
        statuses = sorted({r["status"] or "NOT_ASSESSED" for r in requirements})
        chosen_statuses = st.multiselect("Requirement status", statuses, key="trace_status_filter")
        min_confidence = st.slider("Minimum assessment score", 0.0, 1.0, 0.0, 0.05,
                                   key="trace_confidence_filter")
        search = st.text_input("Search requirement text, ID, or source", key="trace_requirement_search")
        options = [key for key, row in by_key.items()
                   if (not chosen_statuses or (row["status"] or "NOT_ASSESSED") in chosen_statuses)
                   and (row["confidence"] or 0) >= min_confidence
                   and (not search or search.lower() in
                        (row["text"] + " " + row["entity_id"] + " " + row["source_document"]).lower())]
        options.sort(key=lambda key: (by_key[key]["entity_id"], key))
        if not options:
            st.info("No requirements match those filters.")
            return
        linked = st.session_state.pop("trace_deep_link", None)
        if linked in options:
            st.session_state.trace_focus_choice = linked
            st.session_state.trace_focus_select = linked
        elif linked:
            st.warning("This Matrix trace link is stale or outside the selected run.")
        if mode == "Requirement focus":
            previous = st.session_state.get("trace_focus_choice")
            selected = st.selectbox("Requirement", options,
                index=options.index(previous) if previous in options else 0,
                format_func=lambda key: projection["nodes"][key]["title"], key="trace_focus_select")
            st.session_state.trace_focus_choice = selected
            anchors = [selected]
        else:
            anchors = st.multiselect("Compare up to five requirements", options, max_selections=5,
                format_func=lambda key: projection["nodes"][key]["title"], key="trace_compare_select")
    elif mode != "Unlinked evidence":
        st.info("No requirements yet. Unlinked evidence is available in its own view.")
        return
    settings_col, view_col = st.columns([1, 3])
    with settings_col:
        show_evaluations = st.checkbox("Evaluations", value=True, key="trace_evaluations")
        show_candidates = st.checkbox("Retrieved candidates", value=False, key="trace_candidates")
        show_provenance = st.checkbox("Source chunks", value=False, key="trace_provenance")
        relation_labels = {
            "REFERENCES_REQUIREMENT": "Implementation reference",
            "ASSESSES_REQUIREMENT": "Evaluation reference",
            "ASSESSMENT_CONTEXT": "Assessment context",
            "RETRIEVED_CANDIDATE": "Retrieved candidate",
        }
        selected_relations = st.multiselect("Relationship types", list(relation_labels),
            default=list(relation_labels), format_func=lambda key: relation_labels[key],
            key="trace_relationship_filter")
        source_filter = st.text_input("Filter evidence source", key="trace_source_filter")
        limit = st.slider("Visible items", 20, 150, 40, 10, key="trace_limit")
        st.caption("Solid: extracted reference · dotted: assessment context · dashed: candidate or source")
    view_args = dict(mode="unlinked" if mode == "Unlinked evidence" else "focus",
        requirement_ids=anchors, show_evaluations=show_evaluations, show_candidates=show_candidates,
        show_provenance=show_provenance, source_filter=source_filter or None,
        relationship_kinds=set(selected_relations), limit=limit)
    overview = select_view(projection, **view_args)
    groups = overview["groups"]
    with settings_col:
        group_options = [""] + sorted(groups, key=lambda key: (groups[key]["label"], key))
        if st.session_state.get("trace_source_group", "") not in group_options:
            st.session_state.trace_source_group = ""
        chosen_group = st.selectbox("Source group", group_options,
            format_func=lambda key: "All sources" if not key else
                f"{groups[key]['label']} ({groups[key]['count']} items) · {key[:12]}",
            key="trace_source_group")
        group_page = 1
        if chosen_group:
            pages = min(8, max(1, (groups[chosen_group]["count"] + 19) // 20))
            if st.session_state.get("trace_group_page", 1) > pages:
                st.session_state.trace_group_page = 1
            group_page = st.selectbox("Show first", list(range(1, pages + 1)),
                format_func=lambda page: f"{page * 20} items", key="trace_group_page")
    view = select_view(projection, **view_args, source_group=chosen_group or None, group_page=group_page)
    material = json.dumps([snapshot["snapshot_key"], mode, anchors, show_evaluations,
                           show_candidates, show_provenance, source_filter, limit, chosen_group,
                           group_page, selected_relations], sort_keys=True)
    view_key = hashlib.sha256(material.encode()).hexdigest()[:16]
    selection = st.session_state.get("trace_selection")
    valid_ids = {n["id"] for n in view["nodes"]} | {e["id"] for e in view["edges"]}
    if selection not in valid_ids:
        selection = anchors[0] if anchors else None
        st.session_state.trace_selection = selection
    with view_col:
        counts = view["counts"]
        st.caption(f"Showing {counts['visible']} items, {counts['edges_visible']} connections; "
                   f"{counts['collapsed']} items collapsed by the view limit")
        if anchors and not view["edges"]:
            st.info("No relationships are visible with the current filters. The saved assessment is unchanged.")
        display_nodes = [{k: node[k] for k in ("id", "kind", "title", "subtitle", "status", "x", "y")}
                         for node in view["nodes"]]
        display_edges = [{k: edge[k] for k in ("id", "source", "target", "kind", "label")}
                         for edge in view["edges"]]
        payload = {"schema_version": 2, "snapshot_key": snapshot["snapshot_key"],
                   "view_key": view_key, "nodes": display_nodes, "edges": display_edges,
                   "selection": selection}
        event_accepted = False
        try:
            event = render_graph(payload, key="trace_canvas")
            if event and event.get("event_id") != st.session_state.get("trace_last_event"):
                current_view = (event.get("snapshot_key") == snapshot["snapshot_key"] and
                                event.get("view_key") == view_key)
                if current_view and event.get("action") == "clear_selection":
                    st.session_state.trace_last_event = event["event_id"]
                    st.session_state.trace_selection = None
                    selection = None
                    event_accepted = True
                elif current_view and event.get("target_id") in valid_ids and \
                     event.get("action") in ("select_node", "select_edge"):
                    st.session_state.trace_last_event = event["event_id"]
                    st.session_state.trace_selection = event["target_id"]
                    selection = event["target_id"]
                    event_accepted = True
        except RuntimeError as exc:
            st.warning(str(exc))
        choices = [(None, "Nothing selected")]
        choices += [(item["id"], item["title"]) for item in view["nodes"]]
        choices += [(item["id"], item["label"]) for item in view["edges"]]
        if choices:
            selected_index = next((i for i, item in enumerate(choices) if item[0] == selection), 0)
            if event_accepted or st.session_state.get("trace_accessible_select") not in choices:
                st.session_state.trace_accessible_select = choices[selected_index]
            picked = st.selectbox("Accessible item and relationship list", choices,
                index=selected_index, format_func=lambda item: item[1], key="trace_accessible_select")
            if picked[0] != selection:
                selection = picked[0]
                st.session_state.trace_selection = selection
    inspected = selection
    st.divider()
    evidence_query = st.text_input("Find evidence by text, extracted ID, symbol, or source",
                                   key="trace_evidence_search")
    if evidence_query.strip():
        needle = evidence_query.casefold().strip()
        matching = []
        for key, node in projection["nodes"].items():
            if node["kind"] == "Requirement":
                continue
            details = node.get("details", {})
            searchable = " ".join(str(details.get(field) or "") for field in
                                  ("text", "entity_id", "source_document", "source_locator"))
            searchable += " " + node["title"] + " " + node["subtitle"]
            if needle in searchable.casefold():
                matching.append(key)
        matching.sort(key=lambda key: (projection["nodes"][key]["kind"],
                                       projection["nodes"][key]["title"], key))
        st.caption(f"{len(matching)} matching item(s); showing the first 50")
        if matching:
            inspected = st.selectbox("Inspect matching evidence", matching[:50],
                format_func=lambda key: f"{projection['nodes'][key]['title']} · {projection['nodes'][key]['subtitle']}",
                key="trace_evidence_result")
    render_detail(snapshot, projection, inspected)
    if inspected in projection["nodes"]:
        associated = sorted({edge["target"] for edge in projection["edges"].values()
                             if edge["source"] == inspected and edge["target"] in by_key})
        if associated:
            st.caption("Associated requirements: " + ", ".join(by_key[key]["entity_id"] for key in associated))
            first, second = st.columns(2)
            if first.button("Focus first associated requirement", key="trace_focus_associated"):
                _remember_view()
                st.session_state.trace_next_mode = "Requirement focus"
                st.session_state.trace_next_focus = associated[0]
                st.session_state.trace_clear_filters = True
                st.rerun()
            if len(associated) > 1 and second.button("Compare associated requirements", key="trace_compare_associated"):
                _remember_view()
                st.session_state.trace_next_mode = "Compare requirements"
                st.session_state.trace_next_compare = associated[:5]
                st.session_state.trace_clear_filters = True
                st.rerun()
    export = {"schema_version": 2, "snapshot_key": snapshot["snapshot_key"], "view_key": view_key,
              "nodes": view["nodes"], "edges": view["edges"], "counts": view["counts"],
              "filters": {"mode": mode, "requirements": anchors, "status":
                          st.session_state.get("trace_status_filter", []),
                          "minimum_confidence": st.session_state.get("trace_confidence_filter", 0.0),
                          "relationship_kinds": selected_relations, "source_text": source_filter,
                          "source_group": chosen_group, "group_page": group_page,
                          "show_evaluations": show_evaluations, "show_candidates": show_candidates,
                          "show_provenance": show_provenance, "limit": limit}}
    st.download_button("Download displayed graph JSON", json.dumps(export, ensure_ascii=False,
                       indent=2, default=str), file_name="traceability-graph.json", mime="application/json")


def _remember_view() -> None:
    keys = ("page", "trace_run", "graph_run_select", "trace_view_mode", "trace_focus_choice",
            "trace_focus_select", "trace_compare_select", "trace_selection", "trace_status_filter",
            "trace_requirement_search", "trace_confidence_filter", "trace_relationship_filter",
            "trace_source_filter", "trace_source_group", "trace_group_page", "trace_evaluations",
            "trace_candidates", "trace_provenance", "trace_limit")
    previous = {key: st.session_state[key] for key in keys if key in st.session_state}
    st.session_state.setdefault("trace_history", []).append(previous)
