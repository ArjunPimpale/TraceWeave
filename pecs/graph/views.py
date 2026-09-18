"""Bounded, stable graph views and deterministic coordinates."""

from __future__ import annotations

from collections import defaultdict

DEFAULT_NODE_LIMIT = 40
DEFAULT_EDGE_LIMIT = 80
HARD_NODE_LIMIT = 150
HARD_EDGE_LIMIT = 300


def select_view(projection: dict, *, mode: str = "focus", requirement_ids: list[str] | None = None,
                show_evaluations: bool = True, show_candidates: bool = False,
                show_provenance: bool = False, source_filter: str | None = None,
                relationship_kinds: set[str] | None = None,
                source_group: str | None = None, group_page: int = 1,
                limit: int = DEFAULT_NODE_LIMIT) -> dict:
    nodes = projection["nodes"]
    edges = projection["edges"]
    anchors = set(requirement_ids or [])
    if mode == "unlinked":
        connected = {edge["source"] for edge in edges.values()
                     if edge["kind"] in ("REFERENCES_REQUIREMENT", "ASSESSES_REQUIREMENT", "ASSESSMENT_CONTEXT")}
        visible = {key for key, node in nodes.items() if node["kind"] in ("Implementation", "Evaluation")
                   and key not in connected}
    else:
        visible = set(anchors)
        for edge in edges.values():
            if edge["target"] in anchors and edge["kind"] not in ("EXTRACTED_FROM",) \
               and (relationship_kinds is None or edge["kind"] in relationship_kinds) \
               and (show_candidates or edge["kind"] != "RETRIEVED_CANDIDATE"):
                visible.add(edge["source"])
        if show_provenance:
            for edge in edges.values():
                if edge["kind"] == "EXTRACTED_FROM" and edge["source"] in visible:
                    visible.add(edge["target"])
            for edge in edges.values():
                if edge["kind"] == "PART_OF_SOURCE" and edge["source"] in visible:
                    visible.add(edge["target"])
    filtered = set()
    for key in visible:
        node = nodes.get(key)
        if not node:
            continue
        if node["kind"] == "Evaluation" and not show_evaluations:
            continue
        if source_filter and key not in anchors and source_filter.lower() not in (
            node.get("subtitle", "") + " " + str(node.get("details", {}).get("source_document", ""))).lower():
            continue
        filtered.add(key)
    filtered |= anchors & nodes.keys()
    allowed_edges = [e for e in edges.values() if e["source"] in filtered and e["target"] in filtered
                     and (relationship_kinds is None or e["kind"] in relationship_kinds or
                          (show_provenance and e["kind"] in ("EXTRACTED_FROM", "PART_OF_SOURCE")))
                     and (show_candidates or e["kind"] != "RETRIEVED_CANDIDATE")
                     and (show_provenance or e["kind"] not in ("EXTRACTED_FROM", "PART_OF_SOURCE"))]
    if not show_candidates:
        connected = {e["source"] for e in allowed_edges}
        filtered = {k for k in filtered if k in anchors or nodes[k]["kind"] != "Chunk" or k in connected}
        allowed_edges = [e for e in allowed_edges if e["source"] in filtered and e["target"] in filtered]
    groups: dict[str, dict] = {}
    for key in sorted(filtered - anchors):
        node = nodes[key]
        details = node.get("details", {})
        label = details.get("source_document") or "Unknown source"
        group_id = details.get("source_hash") or f"name:{label}"
        group = groups.setdefault(group_id, {"label": label, "count": 0})
        group["count"] += 1
    if source_group is not None:
        filtered = {key for key in filtered if key in anchors or
                    (nodes[key].get("details", {}).get("source_hash") or
                     f"name:{nodes[key].get('details', {}).get('source_document') or 'Unknown source'}") == source_group}
        allowed_edges = [e for e in allowed_edges if e["source"] in filtered and e["target"] in filtered]
    ordered = sorted(filtered, key=lambda key: (0 if key in anchors else 1, nodes[key]["kind"],
                                                nodes[key]["title"], key))
    page_limit = min(max(1, group_page) * 20 + len(anchors), HARD_NODE_LIMIT) if source_group else limit
    keep = set(ordered[:min(max(page_limit, len(anchors)), HARD_NODE_LIMIT)])
    shown_edges = [e for e in allowed_edges if e["source"] in keep and e["target"] in keep]
    shown_edges.sort(key=lambda e: (e["kind"], e["source"], e["target"], e["id"]))
    shown_edges = shown_edges[:min(HARD_EDGE_LIMIT, DEFAULT_EDGE_LIMIT if limit <= DEFAULT_NODE_LIMIT else HARD_EDGE_LIMIT)]
    lanes: dict[str, list[str]] = defaultdict(list)
    for key in ordered:
        if key not in keep:
            continue
        kind = nodes[key]["kind"]
        lane = "Requirement" if key in anchors else ("Implementation" if kind == "Implementation" else
                "Evaluation" if kind == "Evaluation" else "Context")
        lanes[lane].append(key)
    positions = {}
    for lane, x in (("Implementation", 0), ("Requirement", 450), ("Evaluation", 900), ("Context", 450)):
        members = lanes[lane]
        for i, key in enumerate(members):
            y = (i - (len(members) - 1) / 2) * 110 + (450 if lane == "Context" else 0)
            positions[key] = (x, y)
    output_nodes = []
    for key in ordered:
        if key in keep:
            item = dict(nodes[key])
            item["x"], item["y"] = positions[key]
            output_nodes.append(item)
    return {"nodes": output_nodes, "edges": shown_edges,
            "counts": {"total": len(nodes), "filtered": len(filtered), "visible": len(output_nodes),
                       "collapsed": max(0, len(filtered) - len(output_nodes)),
                       "edges_filtered": len(allowed_edges), "edges_visible": len(shown_edges)},
            "groups": groups,
            "anchor_ids": sorted(anchors)}
