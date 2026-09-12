"""
Evidence Graph page — interactive Neo4j graph exploration via pyvis.

Layout:
  - Left column: controls (requirement selector, display toggles, sync button)
  - Right column: pyvis network rendered in an iframe via st.components.v1.html
  - Below graph: node/edge detail panel populated by selectbox selection

Interaction model (V1):
  The graph renders the ego network of the selected requirement (1–2 hops).
  Node selection is driven by the requirement dropdown — pyvis click events are
  not wired in V1 because Streamlit's iframe isolation makes bidirectional JS
  communication unreliable. The dropdown is the primary interaction surface.

  A "Full Graph" mode renders all nodes at once but warns the user if > 50 nodes.

EXTENSION POINT (snapshot support): add a `snapshot_timestamp` Streamlit
  selectbox populated from the correlation table's created_at values.
  Pass the selected timestamp to graph_queries functions when implemented.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from pecs.config import settings
from pecs.logging_config import get_logger

logger = get_logger(__name__)

# ── Status colour palette (matches matrix_page.py STATUS_COLORS) ─────────────
_STATUS_COLORS: dict[str, str] = {
    "IMPLEMENTED_AND_VALIDATED":            "#22c55e",
    "IMPLEMENTED_WITHOUT_EVALUATION":       "#3b82f6",
    "IMPLEMENTED_BUT_NEGATIVELY_EVALUATED": "#f97316",
    "PARTIALLY_IMPLEMENTED":               "#eab308",
    "CLAIMED_BUT_NO_EVIDENCE":             "#a855f7",
    "EVALUATION_WITHOUT_REQUIREMENT":      "#6b7280",
    "REQUIREMENT_NOT_IMPLEMENTED":         "#ef4444",
}

# ── Node visual properties by label ──────────────────────────────────────────
_NODE_STYLES: dict[str, dict[str, Any]] = {
    "Requirement":    {"color": "#3b82f6", "shape": "dot",     "size": 22},
    "Implementation": {"color": "#22c55e", "shape": "square",  "size": 16},
    "Evaluation":     {"color": "#f97316", "shape": "triangle","size": 16},
    "SourceDocument": {"color": "#6b7280", "shape": "diamond", "size": 12},
}

# Default edge colour when status is unknown
_DEFAULT_EDGE_COLOR = "#9ca3af"

# pyvis physics options
_PHYSICS_OPTIONS = """{
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -3500,
      "centralGravity": 0.3,
      "springLength": 160,
      "springConstant": 0.04
    },
    "stabilization": {"iterations": 150}
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true
  },
  "edges": {
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
    "smooth": {"type": "dynamic"}
  }
}"""


def render() -> None:
    """Entry point called by app.py."""
    st.title("🕸️ Evidence Graph")
    st.markdown(
        "Explore the evidence network for each requirement. "
        "Select a requirement from the dropdown to view its connected implementations, "
        "evaluations, and source documents."
    )

    if not settings.NEO4J_ENABLED:
        _render_disabled_notice()
        return

    # ── Lazy import graph modules ──────────────────────────────────────────────
    try:
        from pecs.graph.neo4j_client import get_neo4j_client
        from pecs.graph import graph_queries
        from pecs.graph.graph_sync import GraphSync
    except ImportError as exc:
        st.error(f"Graph module import failed: {exc}. Check that `neo4j` and `pyvis` are installed.")
        return

    try:
        client = get_neo4j_client()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    # ── Connection check ───────────────────────────────────────────────────────
    if not client.check_health():
        _render_connection_error()
        return

    # ── Sidebar controls (rendered as left column) ─────────────────────────────
    col_ctrl, col_graph = st.columns([1, 3], gap="medium")

    with col_ctrl:
        _render_controls(client, graph_queries, GraphSync)

    with col_graph:
        _render_graph(client, graph_queries)

    # ── Detail panel ───────────────────────────────────────────────────────────
    _render_detail_panel(client, graph_queries)


# ── Control panel ─────────────────────────────────────────────────────────────

def _render_controls(client, graph_queries, GraphSync) -> None:
    """Left-column controls: mode selector, requirement picker, toggles, sync."""
    st.subheader("Controls")

    # ── Graph mode ────────────────────────────────────────────────────────────
    mode = st.radio(
        "View mode",
        options=["Requirement focus", "Full graph"],
        index=0,
        help=(
            "Requirement focus: show the ego network of one requirement.\n"
            "Full graph: render all entities at once (may be dense)."
        ),
        key="graph_mode",
    )
    st.session_state.graph_view_mode = mode

    # ── Requirement selector (focus mode) ─────────────────────────────────────
    if mode == "Requirement focus":
        reqs = graph_queries.get_requirement_overview(client)
        if not reqs:
            st.info("No requirements found in graph. Run the correlation pipeline then sync.")
        else:
            req_options = {
                f"{r.get('entity_id', '?')} — {str(r.get('text', ''))[:60]}": r.get("entity_id")
                for r in reqs
            }

            # Pre-select if navigated from another page
            default_idx = 0
            preselect = st.session_state.get("selected_graph_node")
            if preselect:
                keys = list(req_options.keys())
                ids = list(req_options.values())
                if preselect in ids:
                    default_idx = ids.index(preselect)

            selected_label = st.selectbox(
                "Select requirement",
                options=list(req_options.keys()),
                index=default_idx,
                key="graph_req_select",
            )
            selected_id = req_options.get(selected_label)
            st.session_state.selected_graph_node  = selected_id
            st.session_state.selected_inspect_node = selected_id

    else:  # Full graph — show a generic node inspector
        nodes, _ = graph_queries.get_full_graph(client, include_source_docs=True)
        if nodes:
            # Build label → entity_id map for all non-source-doc nodes
            icon_map = {"Requirement": "📌", "Implementation": "⚙️",
                        "Evaluation": "📝", "SourceDocument": "📁"}
            node_options: dict[str, str] = {}
            for n in nodes:
                lbl  = n.get("label") or ""
                nid  = n.get("entity_id") or n.get("name") or ""
                text = str(n.get("text") or n.get("name") or "")[:50]
                if nid:
                    icon = icon_map.get(lbl, "●")
                    node_options[f"{icon} {nid} — {text}"] = nid

            # Preserve previous selection across reruns
            prev = st.session_state.get("selected_inspect_node", "")
            prev_keys = list(node_options.keys())
            prev_vals = list(node_options.values())
            default_inspect = prev_vals.index(prev) if prev in prev_vals else 0

            inspect_label = st.selectbox(
                "Inspect node",
                options=prev_keys,
                index=default_inspect,
                key="graph_inspect_select",
                help="Select any node to see its full details in the panel below the graph.",
            )
            st.session_state.selected_inspect_node = node_options.get(inspect_label, "")
        else:
            st.caption("Sync the graph first to enable node inspection.")

    st.divider()

    # ── Display toggles ───────────────────────────────────────────────────────
    st.markdown("**Show nodes**")
    st.session_state.graph_show_evaluations = st.checkbox(
        "Evaluations",
        value=True,
        key="cb_show_evals",
    )
    st.session_state.graph_show_source_docs = st.checkbox(
        "Source Documents",
        value=False,
        key="cb_show_docs",
        help="Source document nodes can add clutter; hide them to focus on entities.",
    )
    st.session_state.graph_show_orphans = st.checkbox(
        "Orphan entities",
        value=False,
        key="cb_show_orphans",
        help="Entities with no relationships (e.g., evaluations with no linked requirement).",
    )

    st.divider()

    # ── Sync controls ─────────────────────────────────────────────────────────
    st.markdown("**Sync**")
    stats = client.get_graph_stats()
    if stats:
        req_count = stats.get("Requirement", 0)
        impl_count = stats.get("Implementation", 0)
        rel_count = stats.get("CORRELATES_TO", 0)
        st.caption(
            f"📊 Graph: {req_count} req · {impl_count} impl · {rel_count} correlations"
        )
    else:
        st.caption("📊 Graph: empty (not yet synced)")

    if st.button("🔄 Rebuild Graph", use_container_width=True, key="btn_rebuild_graph"):
        _do_sync(GraphSync)

    if st.session_state.get("graph_sync_result"):
        res = st.session_state.graph_sync_result
        if res.get("success"):
            st.success(
                f"✅ Synced: {res['nodes_created']} nodes, "
                f"{res['relationships_created']} edges in {res['duration_ms']}ms"
            )
            if res.get("warnings"):
                with st.expander(f"⚠️ {len(res['warnings'])} warning(s)"):
                    for w in res["warnings"]:
                        st.caption(w)
        else:
            st.error(f"❌ Sync failed: {res.get('error')}")


def _do_sync(GraphSync) -> None:
    """Execute a full graph rebuild and store the result in session state."""
    with st.spinner("Rebuilding graph from SQLite…"):
        try:
            sync = GraphSync()
            result = sync.sync_graph()
            st.session_state.graph_sync_result = result.to_dict()
        except Exception as exc:
            st.session_state.graph_sync_result = {
                "success": False,
                "error": str(exc),
                "nodes_created": 0,
                "relationships_created": 0,
                "duration_ms": 0,
                "warnings": [],
            }
    st.rerun()


# ── Graph canvas ──────────────────────────────────────────────────────────────

def _render_graph(client, graph_queries) -> None:
    """Build and render the pyvis graph in the right column."""
    mode = st.session_state.get("graph_view_mode", "Requirement focus")

    try:
        from pyvis.network import Network
    except ImportError:
        st.error("pyvis is not installed. Run `uv add pyvis` or `pip install pyvis`.")
        return

    show_evals   = st.session_state.get("graph_show_evaluations", True)
    show_docs    = st.session_state.get("graph_show_source_docs", False)
    show_orphans = st.session_state.get("graph_show_orphans", False)

    if mode == "Requirement focus":
        selected_id = st.session_state.get("selected_graph_node")
        if not selected_id:
            st.info("Select a requirement from the controls panel to view its evidence network.")
            return

        ego = graph_queries.get_requirement_ego_network(client, selected_id)
        if not ego:
            st.warning(
                f"No graph data found for `{selected_id}`. "
                "The graph may not be synced yet — click **Rebuild Graph**."
            )
            return

        net = _build_ego_network(ego, show_evals=show_evals, show_docs=show_docs)

    else:  # Full graph
        nodes, edges = graph_queries.get_full_graph(client, include_source_docs=show_docs)

        if not nodes:
            st.info("Graph is empty. Run the correlation pipeline then click **Rebuild Graph**.")
            return

        visible_count = len(nodes)
        if visible_count > 80:
            st.warning(
                f"⚠️ Full graph has {visible_count} nodes. "
                "Rendering may be slow. Consider using Requirement focus mode instead."
            )

        net = _build_full_graph_network(
            nodes, edges,
            show_evals=show_evals,
            show_docs=show_docs,
            show_orphans=show_orphans,
        )

    net.set_options(_PHYSICS_OPTIONS)
    html = net.generate_html(notebook=False)
    components.html(html, height=560, scrolling=False)


def _build_ego_network(ego: dict[str, Any], show_evals: bool, show_docs: bool):
    """Construct a pyvis Network for the ego-network of one requirement."""
    from pyvis.network import Network

    net = Network(height="540px", width="100%", directed=True, notebook=False)

    # Central requirement node
    req_props = ego.get("requirement") or {}
    _add_node(net, req_props, "Requirement", central=True)

    # CORRELATES_TO edges → Implementation nodes
    for corr_item in ego.get("correlations") or []:
        if not corr_item:
            continue
        rel = corr_item.get("rel") or {}
        ev_node = corr_item.get("node") or {}
        if not ev_node:
            continue

        ev_label = _node_label(ev_node)
        _add_node(net, ev_node, ev_label)

        status = rel.get("status", "")
        confidence = rel.get("confidence", 0.0) or 0.0
        edge_color = _STATUS_COLORS.get(status, _DEFAULT_EDGE_COLOR)
        tooltip = _edge_tooltip(rel)

        req_id = req_props.get("entity_id") or ""
        ev_id  = ev_node.get("entity_id") or ev_node.get("name") or ""

        if req_id and ev_id:
            try:
                net.add_edge(
                    req_id, ev_id,
                    color=edge_color,
                    width=max(1.0, confidence * 4),
                    title=tooltip,
                    label=f"{confidence:.0%}",
                    font={"size": 9},
                )
            except (AssertionError, ValueError) as exc:
                logger.warning(
                    "Skipped ego-network correlation edge (node not found in pyvis)",
                    extra={"context": {
                        "from_id": req_id,
                        "to_id": ev_id,
                        "error": str(exc),
                    }},
                )

    # Evaluations
    if show_evals:
        req_id = req_props.get("entity_id") or ""
        for ev in ego.get("evaluations") or []:
            if not ev:
                continue
            _add_node(net, ev, "Evaluation")
            ev_id = ev.get("entity_id") or ""
            if req_id and ev_id:
                try:
                    net.add_edge(
                        ev_id, req_id,
                        color="#f97316",
                        dashes=False,
                        title=_simple_edge_tooltip("EVALUATES"),
                        label="evaluates",
                        font={"size": 9},
                    )
                except (AssertionError, ValueError) as exc:
                    logger.warning(
                        "Skipped ego-network evaluation edge (node not found in pyvis)",
                        extra={"context": {
                            "from_id": ev_id,
                            "to_id": req_id,
                            "error": str(exc),
                        }},
                    )

    # Source documents
    if show_docs:
        req_id = req_props.get("entity_id") or ""
        for doc in (ego.get("evidence_docs") or []) + (ego.get("req_docs") or []):
            if not doc:
                continue
            _add_source_doc_node(net, doc)
            doc_id = "doc::" + (doc.get("name") or "")
            if req_id and doc_id:
                try:
                    net.add_edge(doc_id, req_id, color="#9ca3af", dashes=True, title=_simple_edge_tooltip("EXTRACTED_FROM"), width=1)
                except (AssertionError, ValueError) as exc:
                    logger.warning(
                        "Skipped ego-network source-doc edge (node not found in pyvis)",
                        extra={"context": {
                            "from_id": doc_id,
                            "to_id": req_id,
                            "error": str(exc),
                        }},
                    )

    return net


def _build_full_graph_network(
    nodes: list[dict],
    edges: list[dict],
    show_evals: bool,
    show_docs: bool,
    show_orphans: bool,
):
    """Construct a pyvis Network for the full graph view."""
    from pyvis.network import Network

    net = Network(height="540px", width="100%", directed=True, notebook=False)

    # Track which node IDs are added to avoid duplicates.
    # SourceDocument nodes are keyed by "doc::<name>" to match the ID used in
    # _add_source_doc_node, so edge resolution can apply the same prefix.
    added: set[str] = set()

    for node in nodes:
        label = node.get("label") or ""
        if label == "Evaluation" and not show_evals:
            continue
        if label == "SourceDocument" and not show_docs:
            continue

        raw_id = node.get("entity_id") or node.get("name") or ""
        if not raw_id:
            continue

        if label == "SourceDocument":
            # pyvis node ID has "doc::" prefix (set in _add_source_doc_node)
            node_id = "doc::" + raw_id
            if node_id in added:
                continue
            _add_source_doc_node(net, {"name": raw_id, "source_type": node.get("source_type")})
        else:
            node_id = raw_id
            if node_id in added:
                continue
            _add_node(net, node, label)
        added.add(node_id)

    for edge in edges:
        from_label = edge.get("from_label") or ""
        to_label   = edge.get("to_label") or ""
        rel_type   = edge.get("rel_type") or ""

        if from_label == "Evaluation" and not show_evals:
            continue
        if "SourceDocument" in (from_label, to_label) and not show_docs:
            continue

        # Resolve pyvis node IDs: SourceDocument nodes use "doc::" prefix.
        raw_from = edge.get("from_id") or edge.get("from_name") or ""
        raw_to   = edge.get("to_id") or edge.get("to_name") or ""

        from_id = ("doc::" + raw_from) if from_label == "SourceDocument" else raw_from
        to_id   = ("doc::" + raw_to)   if to_label   == "SourceDocument" else raw_to

        if not from_id or not to_id:
            continue
        if from_id not in added or to_id not in added:
            continue

        status = edge.get("status") or ""
        confidence = float(edge.get("confidence") or 0.0)
        edge_color = _STATUS_COLORS.get(status, _DEFAULT_EDGE_COLOR)
        is_dashed = rel_type in ("EXTRACTED_FROM", "EXPLICITLY_LINKS")

        tooltip = _edge_tooltip(edge) if rel_type == "CORRELATES_TO" else _simple_edge_tooltip(rel_type)

        try:
            net.add_edge(
                from_id, to_id,
                color=edge_color,
                dashes=is_dashed,
                width=max(1.0, confidence * 4) if rel_type == "CORRELATES_TO" else 1.0,
                title=tooltip,
            )
        except (AssertionError, ValueError) as exc:
            logger.warning(
                "Skipped edge in full graph (node not found in pyvis)",
                extra={"context": {
                    "from_id": from_id,
                    "to_id": to_id,
                    "rel_type": rel_type,
                    "error": str(exc),
                }},
            )

    return net


# ── Tooltip builders ─────────────────────────────────────────────────────────
# IMPORTANT: vis.js 9.x uses `innerText` (not innerHTML) for the tooltip
# `setText` method, so HTML tags are rendered as literal text.
#
# The workaround used here:
#   1. Build a plain-text tooltip string (used as the vis.js `title`).
#   2. Separately store rich HTML in a `data-tooltip-html` attribute on each
#      vis.js node/edge via a JS post-render hook injected into the pyvis HTML.
#   3. The JS hook intercepts the tooltip show event and replaces the innerText
#      with the real styled HTML card.
#
# The JS is injected via _inject_tooltip_styles() which patches the raw HTML
# string returned by net.generate_html().

_LABEL_ICONS: dict[str, str] = {
    "Requirement":    "📌",
    "Implementation": "⚙️",
    "Evaluation":     "📝",
    "SourceDocument": "📁",
}

_STATUS_EMOJI: dict[str, str] = {
    "IMPLEMENTED_AND_VALIDATED":            "✅",
    "IMPLEMENTED_WITHOUT_EVALUATION":       "🔵",
    "IMPLEMENTED_BUT_NEGATIVELY_EVALUATED": "⚠️",
    "PARTIALLY_IMPLEMENTED":               "🟡",
    "CLAIMED_BUT_NO_EVIDENCE":             "🟣",
    "EVALUATION_WITHOUT_REQUIREMENT":      "⬜",
    "REQUIREMENT_NOT_IMPLEMENTED":         "❌",
}

# Confidence thresholds → bar characters (plain text bar)
def _text_bar(confidence: float) -> str:
    """Return a plain-text progress bar string, e.g. '████░░░░░░ 72%'."""
    pct = int(confidence * 100)
    filled = round(confidence * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar} {pct}%"


def _plain_tooltip(icon: str, type_label: str, rows: list[tuple[str, str]]) -> str:
    """
    Build a clean plain-text tooltip string.

    vis.js 9.x uses `innerText` so HTML is escaped — we use unicode box
    characters and careful spacing to create a readable structured card.

    Example output:
        📌 Requirement
        ─────────────────────────
        ID           r3-gnn-train
        Description  The system must implement…
        Source       spec.pdf
    """
    SEP = "─" * 28
    lines: list[str] = [f"{icon}  {type_label}", SEP]
    for key, val in rows:
        if not val or val in ("", "None", "?", "—", "none", "unknown"):
            continue
        # Truncate long values
        val_str = str(val)
        if len(val_str) > 55:
            val_str = val_str[:52] + "…"
        # Left-pad key to fixed width
        key_padded = key.ljust(14)
        # Wrap second+ lines with indent
        lines.append(f"{key_padded}  {val_str}")
    return "\n".join(lines)


# ── Node helpers ──────────────────────────────────────────────────────────────

_LABEL_ICONS: dict[str, str] = {
    "Requirement":    "📌",
    "Implementation": "⚙️",
    "Evaluation":     "📝",
    "SourceDocument": "📁",
}


# ── Node helpers ──────────────────────────────────────────────────────────────

def _add_node(net, props: dict, label: str, central: bool = False) -> None:
    """Add a standard entity node to the network."""
    style = _NODE_STYLES.get(label, {"color": "#888", "shape": "dot", "size": 14})
    node_id = props.get("entity_id") or ""
    if not node_id:
        return

    display_label = str(props.get("entity_id") or "")
    text = str(props.get("text") or "")[:160]
    source_doc = props.get("source_document") or ""
    author = props.get("author") or ""
    timestamp = props.get("timestamp") or ""
    chunk_id = str(props.get("chunk_id") or "")
    linked_req = props.get("linked_requirement") or ""

    icon = _LABEL_ICONS.get(label, "●")
    type_label = f"{label}{'  ★ focus' if central else ''}"

    rows: list[tuple[str, str]] = [("ID", display_label)]
    if text:
        rows.append(("Description", text))
    if source_doc:
        rows.append(("Source", source_doc))
    if author:
        rows.append(("Author", author))
    if timestamp:
        rows.append(("Timestamp", timestamp))
    if linked_req and label in ("Implementation", "Evaluation"):
        rows.append(("Linked req", linked_req))
    if chunk_id:
        rows.append(("Chunk ID", chunk_id[:24] + "…"))

    tooltip = _plain_tooltip(icon, type_label, rows)
    size = style["size"] * (1.5 if central else 1)

    net.add_node(
        node_id,
        label=display_label,
        color=style["color"],
        shape=style["shape"],
        size=size,
        title=tooltip,
        borderWidth=3 if central else 1,
        font={"size": 11 if central else 9, "bold": central},
    )


def _add_source_doc_node(net, doc: dict) -> None:
    """Add a SourceDocument node with a doc-prefixed ID to avoid collisions."""
    name = doc.get("name") or ""
    if not name:
        return
    node_id = "doc::" + name
    source_type = doc.get("source_type") or "UNKNOWN"
    type_icons = {
        "PDF": "📄", "PYTHON": "🐍", "GIT": "🔀", "WHATSAPP": "💬",
        "EMAIL": "📧", "DOCX": "📝", "MARKDOWN": "📋",
    }
    icon = type_icons.get(source_type, "📁")

    tooltip = _plain_tooltip(icon, "Source Document", [
        ("Filename", name),
        ("Format",   source_type),
    ])

    net.add_node(
        node_id,
        label=f"{icon} {name}",
        color="#6b7280",
        shape="diamond",
        size=10,
        title=tooltip,
        font={"size": 8},
    )


def _node_label(node: dict) -> str:
    """Infer node label from its properties."""
    # neo4j driver returns label information differently depending on query
    # Try to infer from entity_id prefix or fall back to Implementation
    eid = node.get("entity_id") or ""
    if eid.lower().startswith(("r-", "req-", "r1", "r2", "r3", "r4", "r5")):
        return "Requirement"
    if eid.lower().startswith(("eval", "ev-")):
        return "Evaluation"
    return "Implementation"


def _edge_tooltip(rel: dict) -> str:
    """Build a plain-text tooltip for a CORRELATES_TO relationship."""
    status     = rel.get("status") or ""
    confidence = float(rel.get("confidence") or 0.0)
    method     = rel.get("resolution_method") or ""
    rule       = rel.get("rule_name") or ""
    reasoning  = (rel.get("reasoning") or "")[:200]
    corr_id    = str(rel.get("correlation_id") or "")[:16]

    status_emoji = _STATUS_EMOJI.get(status, "◆")
    method_label = {
        "deterministic_rule": "🔧 Deterministic rule",
        "llm_stage2":         "🤖 LLM stage-2",
    }.get(method, method or "?")

    rows: list[tuple[str, str]] = []
    if status:
        rows.append(("Status", f"{status_emoji}  {status.replace('_', ' ').title()}"))
    rows.append(("Confidence", _text_bar(confidence)))
    if method_label:
        rows.append(("Resolution", method_label))
    if rule:
        rows.append(("Rule", rule))
    if reasoning:
        rows.append(("Reasoning", reasoning))
    if corr_id:
        rows.append(("Corr. ID", corr_id + "…"))

    return _plain_tooltip("🔗", "Correlation", rows)


def _simple_edge_tooltip(rel_type: str) -> str:
    """Plain-text tooltip for non-CORRELATES_TO edges."""
    descriptions = {
        "EVALUATES":        ("📝", "Evaluation → Requirement",
                            "Meaning: this evaluation directly assesses the linked requirement."),
        "EXTRACTED_FROM":   ("📂", "Entity → Source Document",
                            "Meaning: this entity was extracted from the source document."),
        "EXPLICITLY_LINKS": ("🔗", "Implementation → Requirement",
                            "Meaning: LLM found an explicit mention of the requirement ID."),
    }
    icon, title, desc = descriptions.get(rel_type, ("↔", rel_type, ""))
    rows = [("Relationship", rel_type)]
    if desc:
        rows.append(("", desc))
    return _plain_tooltip(icon, title, rows)


# ── Detail panel ──────────────────────────────────────────────────────────────

def _render_detail_panel(client, graph_queries) -> None:
    """
    Unified node detail panel shown below the graph in both view modes.

    Focus mode  → details for the selected Requirement (ego network).
    Full graph  → details for whichever node is selected in the inspector
                  selectbox (any node type: Req / Impl / Eval / SourceDoc).
    """
    mode = st.session_state.get("graph_view_mode", "Requirement focus")

    if mode == "Requirement focus":
        selected_id = st.session_state.get("selected_graph_node")
    else:
        selected_id = st.session_state.get("selected_inspect_node")

    if not selected_id:
        return

    st.divider()
    st.subheader(f"📋 Node Details")

    # ── Always render the rich requirement detail when it's a requirement ──
    # Try fetching ego network — if it returns data, it's a Requirement.
    ego = graph_queries.get_requirement_ego_network(client, selected_id)
    if ego and ego.get("requirement"):
        _render_requirement_detail(ego, selected_id, graph_queries, client)
        return

    # ── Otherwise look it up from the full graph node list ─────────────────
    nodes, _ = graph_queries.get_full_graph(client, include_source_docs=True)
    node_data = next(
        (n for n in nodes
         if n.get("entity_id") == selected_id or n.get("name") == selected_id),
        None,
    )
    if node_data:
        _render_generic_node_detail(node_data, selected_id)
    else:
        st.info(f"No detail data found for `{selected_id}`. The graph may need a resync.")


def _render_requirement_detail(
    ego: dict,
    selected_id: str,
    graph_queries,
    client,
) -> None:
    """Full detail card for a Requirement node using its ego-network data."""
    req = ego.get("requirement") or {}
    correlations = ego.get("correlations") or []
    evaluations  = ego.get("evaluations") or []

    # ── Header card ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg,#1e3a5f 0%,#1e2d45 100%);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        '>
        <div style='color:#93c5fd;font-size:11px;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;'>📌 Requirement</div>
        <div style='color:#f0f9ff;font-size:15px;font-weight:600;
                    margin-top:4px;'>{selected_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metadata row ───────────────────────────────────────────────────────
    meta_cols = st.columns([2, 1, 1])
    with meta_cols[0]:
        st.markdown("**📄 Full requirement text**")
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #334155;"
            f"border-radius:6px;padding:12px 16px;color:#e2e8f0;"
            f"font-size:14px;line-height:1.65;'>{req.get('text','—')}</div>",
            unsafe_allow_html=True,
        )
    with meta_cols[1]:
        st.metric("Correlated evidence", len(correlations))
        st.metric("Linked evaluations", len(evaluations))
    with meta_cols[2]:
        st.markdown("**Source**")
        st.code(req.get("source_document", "?"), language=None)
        if req.get("author"):
            st.caption(f"✍️ {req.get('author')}")
        if req.get("timestamp"):
            st.caption(f"🕐 {req.get('timestamp')}")
        if req.get("chunk_id"):
            st.caption(f"Chunk: `{str(req.get('chunk_id'))[:20]}…`")

    # ── Correlated evidence table ───────────────────────────────────────────
    if correlations:
        st.markdown("---")
        st.markdown("**⚙️ Correlated Evidence**")
        for item in correlations:
            if not item:
                continue
            rel  = item.get("rel") or {}
            node = item.get("node") or {}
            if not node:
                continue
            status     = rel.get("status", "?")
            confidence = float(rel.get("confidence") or 0.0)
            color      = _STATUS_COLORS.get(status, "#888")
            method     = rel.get("resolution_method", "?")
            entity_id  = node.get("entity_id", "?")
            node_text  = node.get("text", "—")

            with st.expander(
                f"{_STATUS_EMOJI.get(status,'▪')} [{confidence:.0%}]  {entity_id}",
                expanded=False,
            ):
                badge_col, text_col = st.columns([1, 3])
                with badge_col:
                    st.markdown(
                        f"<span style='background:{color};color:#fff;"
                        f"border-radius:4px;padding:2px 8px;font-size:11px;"
                        f"font-weight:700;'>{status.replace('_',' ')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Confidence: **{confidence:.0%}**")
                    st.caption(f"Method: {method}")
                    if rel.get("rule_name"):
                        st.caption(f"Rule: `{rel.get('rule_name')}`")
                    st.caption(f"Source: `{node.get('source_document','?')}`")
                    if node.get("chunk_id"):
                        st.caption(f"Chunk: `{str(node.get('chunk_id'))[:20]}…`")
                with text_col:
                    st.markdown("**Full evidence text:**")
                    st.markdown(
                        f"<div style='background:#0f172a;border:1px solid #334155;"
                        f"border-radius:6px;padding:10px 14px;color:#e2e8f0;"
                        f"font-size:13px;line-height:1.6;'>{node_text}</div>",
                        unsafe_allow_html=True,
                    )
                    if rel.get("reasoning"):
                        st.markdown("**Reasoning:**")
                        st.info(rel.get("reasoning"))

    # ── Evaluation list ────────────────────────────────────────────────────
    if evaluations:
        st.markdown("---")
        st.markdown("**📝 Linked Evaluations**")
        for ev in evaluations:
            if not ev:
                continue
            with st.expander(f"📝 {ev.get('entity_id', '?')}", expanded=False):
                st.markdown(
                    f"<div style='background:#0f172a;border:1px solid #334155;"
                    f"border-radius:6px;padding:10px 14px;color:#e2e8f0;"
                    f"font-size:13px;line-height:1.6;'>{ev.get('text','—')}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Source: `{ev.get('source_document','?')}`")
                if ev.get("author"):
                    st.caption(f"Author: {ev.get('author')}")

    # ── Shared artifacts ───────────────────────────────────────────────────
    shared = graph_queries.get_shared_artifacts(client)
    shared_for_req = [
        s for s in shared
        if s.get("req1_id") == selected_id or s.get("req2_id") == selected_id
    ]
    if shared_for_req:
        st.markdown("---")
        st.markdown("**🔗 Shared artifacts (connected to other requirements)**")
        for s in shared_for_req:
            other = s.get("req2_id") if s.get("req1_id") == selected_id else s.get("req1_id")
            st.info(
                f"🔗 `{s.get('shared_id')}` ({s.get('shared_type')}) "
                f"is also connected to **{other}**"
            )


def _render_generic_node_detail(node_data: dict, node_id: str) -> None:
    """Detail card for Implementation, Evaluation, or SourceDocument nodes."""
    label = node_data.get("label") or "Entity"
    icon_map = {
        "Requirement":    ("📌", "#3b82f6", "#1e3a5f"),
        "Implementation": ("⚙️",  "#22c55e", "#14291e"),
        "Evaluation":     ("📝", "#f97316", "#2c1a0e"),
        "SourceDocument": ("📁", "#6b7280", "#1c2127"),
    }
    icon, accent, bg = icon_map.get(label, ("●", "#6b7280", "#1c2127"))

    text        = node_data.get("text") or node_data.get("name") or "—"
    source_doc  = node_data.get("source_document") or ""
    source_type = node_data.get("source_type") or ""

    # Header
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg,{bg} 0%,#1e2130 100%);
            border-left: 4px solid {accent};
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        '>
        <div style='color:{accent};font-size:11px;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;'>{icon} {label}</div>
        <div style='color:#f0f9ff;font-size:15px;font-weight:600;
                    margin-top:4px;'>{node_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if label == "SourceDocument":
        st.markdown(f"**Format:** `{source_type or 'unknown'}`")
        st.markdown(f"**Filename:** `{node_data.get('name','?')}`")
        st.caption("Source document nodes represent the files that entities were extracted from.")
        return

    # Full text
    st.markdown("**Full text:**")
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #334155;"
        f"border-radius:6px;padding:12px 16px;color:#e2e8f0;"
        f"font-size:14px;line-height:1.65;'>{text}</div>",
        unsafe_allow_html=True,
    )

    # Metadata
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        if source_doc:
            st.markdown(f"**Source file:** `{source_doc}`")
        extra_source = node_data.get("source_type")
        if extra_source:
            st.markdown(f"**Type:** `{extra_source}`")
    with meta_col2:
        author = node_data.get("author")
        if author:
            st.markdown(f"**Author:** {author}")
        ts = node_data.get("timestamp")
        if ts:
            st.markdown(f"**Timestamp:** {ts}")
        linked = node_data.get("linked_requirement")
        if linked:
            st.markdown(f"**Linked requirement:** `{linked}`")
        chunk_id = node_data.get("chunk_id")
        if chunk_id:
            st.caption(f"Chunk ID: `{str(chunk_id)[:30]}…`")


# ── Error / disabled notices ───────────────────────────────────────────────────

def _render_disabled_notice() -> None:
    st.info(
        "Graph features are disabled (`NEO4J_ENABLED=false`). "
        "To enable them:\n\n"
        "1. Install Neo4j Community Edition locally.\n"
        "2. Start Neo4j (default port 7687).\n"
        "3. Set `NEO4J_ENABLED=true` in your `.env` file.\n"
        "4. Restart PECS."
    )


def _render_connection_error() -> None:
    st.error(
        "Cannot connect to Neo4j. "
        f"Check that Neo4j is running at `{settings.NEO4J_URI}`."
    )
    if st.button("🔁 Retry connection", key="btn_retry_neo4j"):
        st.rerun()
