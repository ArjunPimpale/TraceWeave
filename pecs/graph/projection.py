"""Pure projection of a traceability snapshot into typed nodes and edges."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from pecs.graph.labels import chunk_label, evidence_label, short
from pecs.graph.models import GraphEdge, GraphNode


def evidence_key(dataset: str, row_id: int) -> str:
    return f"ev:{dataset}:{row_id}"


def chunk_key(dataset: str, row: dict) -> str:
    return f"chunk:{dataset}:{row['chunk_id']}:{row.get('snapshot_id') or 'missing'}"


def source_key(dataset: str, row: dict) -> str:
    return f"source:{dataset}:{row.get('source_hash') or 'unknown-' + row['chunk_id']}"


def build_projection(snapshot: dict) -> dict:
    dataset = snapshot["dataset_uuid"]
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    req_by_row = {r["row_id"]: r for r in snapshot["requirements"]}
    evidence = snapshot["evidence"]
    chunks = snapshot["chunks"]
    for req in snapshot["requirements"]:
        row = evidence[req["row_id"]]
        key = evidence_key(dataset, req["row_id"])
        nodes[key] = GraphNode(key, "Requirement", evidence_label(row),
            row.get("source_document") or "", req["status"],
            details={"row_id": req["row_id"], "text": req["text"], "workflow_state": req["workflow_state"],
                     "confidence": req["confidence"], "assessment_ids": req["assessment_ids"],
                     "diagnostics": req["diagnostics"], "conflicting": req["conflicting"]})
    for row in evidence.values():
        if row["entity_type"] == "REQUIREMENT":
            continue
        key = evidence_key(dataset, row["id"])
        nodes[key] = GraphNode(key, row["entity_type"].title(), evidence_label(row, chunks.get(row["chunk_id"])),
            f"{row.get('source_document') or 'Unknown source'} · {chunks.get(row['chunk_id'], {}).get('source_locator') or ''}",
            details={"row_id": row["id"], "text": row["text"], "entity_id": row["entity_id"],
                     "source_document": row["source_document"], "chunk_id": row["chunk_id"],
                     "source_hash": chunks.get(row["chunk_id"], {}).get("source_hash"),
                     "linked_requirement": row.get("linked_requirement"),
                     "author": row.get("author"), "timestamp": row.get("timestamp"),
                     "metadata": row.get("metadata")})
    for association in snapshot["associations"]:
        if association["requirement_row_id"] is None:
            key = evidence_key(dataset, association["evidence_row_id"])
            if key in nodes:
                nodes[key].details["unresolved_reference"] = association.get("unresolved_value")
                nodes[key].details["reference_candidates"] = association.get("candidate_ids", [])
    for assessment in snapshot.get("unlinked_assessments", []):
        for ref in assessment.get("references", []):
            if ref.get("target_kind") == "evidence":
                key = evidence_key(dataset, ref["evidence_row_id"])
                if key in nodes:
                    nodes[key].details.setdefault("unlinked_assessment_ids", []).append(assessment["correlation_id"])
    resolved_rows = {association["evidence_row_id"] for association in snapshot["associations"]
                     if association["requirement_row_id"] is not None}
    for row in evidence.values():
        if row["entity_type"] == "REQUIREMENT" or row["id"] in resolved_rows:
            continue
        details = nodes[evidence_key(dataset, row["id"])].details
        if details.get("reference_candidates"):
            details["unlinked_reason"] = "Requirement reference is ambiguous"
        elif details.get("unresolved_reference") or row.get("linked_requirement"):
            details["unlinked_reason"] = "Requirement reference target is missing"
        else:
            details["unlinked_reason"] = "No extracted requirement reference"
    for chunk in chunks.values():
        key = chunk_key(dataset, chunk)
        nodes[key] = GraphNode(key, "Chunk", chunk_label(chunk) if chunk.get("normalized_text") else
            f"Source unavailable · {chunk['chunk_id'][:10]}",
            f"{chunk.get('source_type') or 'UNKNOWN'} · {chunk.get('source_locator') or ''}",
            details={"chunk_id": chunk["chunk_id"], "text": chunk.get("normalized_text"),
                     "source_document": chunk.get("source_document"), "source_locator": chunk.get("source_locator"),
                     "source_hash": chunk.get("source_hash"), "availability": chunk.get("availability")})
        source_id = source_key(dataset, chunk)
        if source_id not in nodes:
            nodes[source_id] = GraphNode(source_id, "SourceDocument",
                chunk.get("source_document") or "Unknown source",
                chunk.get("source_type") or "UNKNOWN",
                details={"source_hash": chunk.get("source_hash"),
                         "source_document": chunk.get("source_document")})

    def add_edge(source: str, target: str, kind: str, label: str, assessment_id: str | None = None,
                 reference_id: str | None = None,
                 details: dict | None = None) -> None:
        if source not in nodes or target not in nodes:
            return
        key = f"edge:{source}:{target}:{kind}:{label}"
        if key not in edges:
            edges[key] = GraphEdge(key, source, target, kind, label, details=details or {})
        if assessment_id and assessment_id not in edges[key].assessment_ids:
            edges[key].assessment_ids.append(assessment_id)
        if reference_id and reference_id not in edges[key].reference_ids:
            edges[key].reference_ids.append(reference_id)

    for assoc in snapshot["associations"]:
        rid = assoc["requirement_row_id"]
        if rid is not None:
            add_edge(evidence_key(dataset, assoc["evidence_row_id"]), evidence_key(dataset, rid),
                     assoc["kind"], "Explicit reference" if assoc["kind"] == "REFERENCES_REQUIREMENT" else "Evaluation reference",
                     details={"basis": "extracted linked_requirement"})
    for corr in snapshot["correlations"].values():
        rid = corr.get("requirement_row_id")
        if rid not in req_by_row:
            continue
        target = evidence_key(dataset, rid)
        for ref in corr["references"]:
            if ref["target_kind"] == "evidence":
                source = evidence_key(dataset, ref["evidence_row_id"])
            elif ref["target_kind"] == "chunk":
                chunk = chunks.get(ref["chunk_id"])
                if not chunk:
                    continue
                source = chunk_key(dataset, chunk)
            else:
                continue
            role = ref["role"]
            if role in ("explicit_implementation", "explicit_evaluation"):
                kind = "REFERENCES_REQUIREMENT" if role == "explicit_implementation" else "ASSESSES_REQUIREMENT"
                label = "Explicit reference" if role == "explicit_implementation" else "Evaluation reference"
            elif role == "rule_selected_chunk":
                kind, label = "ASSESSMENT_CONTEXT", "Rule-selected context"
            elif role == "stage2_context":
                kind, label = "ASSESSMENT_CONTEXT", "Classifier input"
            else:
                kind, label = "ASSESSMENT_CONTEXT", "Legacy citation"
            add_edge(source, target, kind, label, corr["correlation_id"],
                     ref.get("reference_id"), {"basis": role, "confidence_is_aggregate": True})
    for rid, candidates in snapshot["candidates"].items():
        for candidate in candidates:
            chunk = chunks.get(candidate["chunk_id"])
            if chunk:
                add_edge(chunk_key(dataset, chunk), evidence_key(dataset, rid),
                         "RETRIEVED_CANDIDATE", "Retrieved candidate",
                         details={"rank": candidate["rank"], "combined_score": candidate["combined_score"]})
    # Provenance remains optional in the view layer, with actual owner direction.
    for row in evidence.values():
        chunk = chunks.get(row["chunk_id"])
        if chunk:
            add_edge(evidence_key(dataset, row["id"]), chunk_key(dataset, chunk),
                     "EXTRACTED_FROM", "Extracted from")
    for chunk in chunks.values():
        add_edge(chunk_key(dataset, chunk), source_key(dataset, chunk),
                 "PART_OF_SOURCE", "Part of source")
    return {"nodes": {k: v.to_dict() for k, v in nodes.items()},
            "edges": {k: v.to_dict() for k, v in edges.items()}}
