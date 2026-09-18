"""A consistent, read-only traceability view for Matrix and Graph."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from pecs.models.correlation_result import CorrelationStatus
from pecs.store.traceability_repo import TraceabilityRepo


def _decode(value: str | None, fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except (ValueError, TypeError):
        return fallback


class TraceabilityReader:
    def __init__(self, repo: TraceabilityRepo | None = None) -> None:
        self.repo = repo or TraceabilityRepo()

    def list_snapshots(self) -> list[dict]:
        return self.repo.list_runs()

    def load(self, run_id: str | None = None) -> dict:
        runs = self.repo.list_runs()
        if run_id is None:
            run_id = runs[0]["run_id"] if runs else "legacy"
        return self._legacy() if run_id == "legacy" else self._new(run_id)

    def _new(self, run_id: str) -> dict:
        data = self.repo.read_run(run_id)
        run = data["run"]
        evidence = {row["id"]: row for row in data["evidence"]}
        chunks = {row["chunk_id"]: row for row in data["chunks"]}
        by_corr: dict[str, list[dict]] = defaultdict(list)
        for ref in data["references"]:
            ref["candidate_ids"] = _decode(ref.get("candidate_ids_json"), [])
            by_corr[ref["correlation_id"]].append(ref)
        by_req: dict[int, list[dict]] = defaultdict(list)
        unlinked: list[dict] = []
        for corr in data["correlations"]:
            corr["references"] = by_corr[corr["correlation_id"]]
            corr["supporting_chunk_ids"] = _decode(corr.get("supporting_chunk_ids"), [])
            if corr["requirement_row_id"] is None:
                unlinked.append(corr)
            else:
                by_req[corr["requirement_row_id"]].append(corr)
        by_candidate: dict[int, list[dict]] = defaultdict(list)
        for candidate in data["candidates"]:
            by_candidate[candidate["requirement_row_id"]].append(candidate)
        names: dict[str, list[int]] = defaultdict(list)
        for rr in data["requirements"]:
            names[rr["entity_id"]].append(rr["requirement_row_id"])
        associations: list[dict] = []
        for row in evidence.values():
            link = row.get("linked_requirement")
            if row["entity_type"] not in ("IMPLEMENTATION", "EVALUATION") or not link:
                continue
            targets = names.get(link, [])
            associations.append({
                "evidence_row_id": row["id"], "requirement_row_id": targets[0] if len(targets) == 1 else None,
                "kind": "REFERENCES_REQUIREMENT" if row["entity_type"] == "IMPLEMENTATION" else "ASSESSES_REQUIREMENT",
                "unresolved_value": link if len(targets) != 1 else None,
                "candidate_ids": targets if len(targets) != 1 else [],
            })
        requirements = []
        for rr in data["requirements"]:
            rid = rr["requirement_row_id"]
            corrs = by_req.get(rid, [])
            selected = next((c for c in corrs if c["correlation_id"] == rr["selected_correlation_id"]), None)
            refs = [r for c in corrs for r in c["references"]]
            assessment_targets = {(r["target_kind"], r["evidence_row_id"] or r["chunk_id"])
                                  for r in refs if r["target_kind"] != "unresolved" and r["role"] != "retrieved_candidate"}
            source_docs = set()
            for kind, target in assessment_targets:
                source = evidence.get(target) if kind == "evidence" else chunks.get(target)
                if source and source.get("source_document"):
                    source_docs.add(source["source_document"])
            requirements.append({
                "row_id": rid, "entity_id": rr["entity_id"], "text": rr["text"],
                "source_document": rr["source_document"], "chunk_id": rr["chunk_id"],
                "workflow_state": rr["workflow_state"], "diagnostics": _decode(rr["diagnostics_json"], []),
                "status": selected["status"] if selected else None,
                "confidence": selected["confidence"] if selected else 0.0,
                "reasoning": selected["reasoning"] if selected else "",
                "resolution_method": selected["resolution_method"] if selected else "none",
                "selected_correlation_id": selected["correlation_id"] if selected else None,
                "assessment_ids": [c["correlation_id"] for c in corrs],
                "conflicting": len({c["status"] for c in corrs}) > 1,
                "evidence_count": len(assessment_targets), "source_documents": sorted(source_docs),
                "supporting_chunk_ids": sorted({r["chunk_id"] for r in refs if r["chunk_id"]}),
                "implementation_entities": len({r["evidence_row_id"] for r in refs if r["role"] == "explicit_implementation"}),
                "evaluation_entities": len({r["evidence_row_id"] for r in refs if r["role"] == "explicit_evaluation"}),
                "assessment_context_chunks": len({r["chunk_id"] for r in refs if r["target_kind"] == "chunk"}),
                "retrieved_candidate_chunks": len(by_candidate[rid]),
                "unresolved_references": sum(r["target_kind"] == "unresolved" for r in refs),
            })
        return {
            "schema_version": 2, "dataset_uuid": run["dataset_uuid"], "run_id": run_id,
            "snapshot_key": f"{run['dataset_uuid']}:{run_id}:2", "legacy": False,
            "requirements": requirements, "evidence": evidence, "chunks": chunks,
            "correlations": {c["correlation_id"]: c for c in data["correlations"]},
            "candidates": dict(by_candidate), "associations": associations,
            "unlinked_assessments": unlinked, "outcome": run["outcome"],
            "not_in_run": [dict(row) for row in self.repo.conn.execute(
                "SELECT * FROM evidence WHERE entity_type='REQUIREMENT' AND id>? ORDER BY id", (run["evidence_watermark"],))],
        }

    def _legacy(self) -> dict:
        conn = self.repo.conn
        evidence_rows = [dict(row) for row in conn.execute("SELECT * FROM evidence ORDER BY id")]
        evidence = {row["id"]: row for row in evidence_rows}
        reqs = [row for row in evidence_rows if row["entity_type"] == "REQUIREMENT"]
        names: dict[str, list[int]] = defaultdict(list)
        by_entity: dict[str, list[int]] = defaultdict(list)
        for row in evidence_rows:
            by_entity[row["entity_id"]].append(row["id"])
            if row["entity_type"] == "REQUIREMENT":
                names[row["entity_id"]].append(row["id"])
        corr_rows = [dict(row) for row in conn.execute(
            "SELECT c.* FROM correlation c LEFT JOIN correlation_context x ON x.correlation_id=c.correlation_id WHERE x.correlation_id IS NULL ORDER BY c.created_at,c.id")]
        snaps_by_chunk: dict[str, list[dict]] = defaultdict(list)
        for row in conn.execute("SELECT * FROM chunk_snapshot WHERE origin='legacy_hydration' ORDER BY rowid"):
            snaps_by_chunk[row["chunk_id"]].append(dict(row))
        chunks = {cid: dict(rows[0], availability="available") for cid, rows in snaps_by_chunk.items() if len(rows) == 1}
        for row in evidence_rows:
            chunks.setdefault(row["chunk_id"], {"chunk_id": row["chunk_id"], "availability": "missing",
                                                    "source_document": row["source_document"]})
        by_req: dict[int, list[dict]] = defaultdict(list)
        unlinked: list[dict] = []
        for corr in corr_rows:
            ids = names.get(corr["requirement_entity_id"], [])
            corr["requirement_row_id"] = ids[0] if len(ids) == 1 else None
            corr["references"] = []
            corr["supporting_chunk_ids"] = _decode(corr.get("supporting_chunk_ids"), [])
            raw_target = corr["evidence_entity_id"]
            matches = by_entity.get(raw_target, [])
            chunk_known = raw_target in chunks
            if len(matches) == 1 and not chunk_known:
                corr["references"].append({"target_kind": "evidence", "evidence_row_id": matches[0],
                                           "chunk_id": None, "role": "legacy_primary_target"})
            elif chunk_known and not matches:
                corr["references"].append({"target_kind": "chunk", "evidence_row_id": None,
                                           "chunk_id": raw_target, "role": "legacy_primary_target"})
            elif raw_target not in ("(none)", "(unlinked)", "(unknown)"):
                corr["references"].append({"target_kind": "unresolved", "evidence_row_id": None,
                                           "chunk_id": None, "unresolved_value": raw_target, "role": "legacy_primary_target"})
            for cid in corr["supporting_chunk_ids"]:
                corr["references"].append({"target_kind": "chunk", "chunk_id": cid,
                                           "evidence_row_id": None, "role": "legacy_citation"})
                chunks.setdefault(cid, {"chunk_id": cid, "availability": "missing", "source_document": None})
            if corr["requirement_row_id"] is None:
                unlinked.append(corr)
            else:
                by_req[corr["requirement_row_id"]].append(corr)
        requirements = []
        for req in reqs:
            rows = by_req[req["id"]]
            latest = max((c["created_at"] for c in rows), default=None)
            current = [c for c in rows if c["created_at"] == latest]
            selected = min(current, key=lambda c: (-c["confidence"], c["id"])) if current else None
            ids = sorted({cid for c in current for cid in c["supporting_chunk_ids"]})
            sources = sorted({chunks[cid]["source_document"] for cid in ids if chunks.get(cid, {}).get("source_document")})
            requirements.append({"row_id": req["id"], "entity_id": req["entity_id"], "text": req["text"],
                "source_document": req["source_document"], "chunk_id": req["chunk_id"],
                "workflow_state": "assessed" if selected else "classification_failed",
                "diagnostics": ["Legacy correlations have no reliable run boundary"],
                "status": selected["status"] if selected else None,
                "confidence": selected["confidence"] if selected else 0.0,
                "reasoning": selected["reasoning"] if selected else "",
                "resolution_method": selected["resolution_method"] if selected else "none",
                "selected_correlation_id": selected["correlation_id"] if selected else None,
                "assessment_ids": [c["correlation_id"] for c in current],
                "conflicting": len({c["status"] for c in current}) > 1,
                "evidence_count": len(ids), "source_documents": sources,
                "supporting_chunk_ids": ids, "implementation_entities": 0, "evaluation_entities": 0,
                "assessment_context_chunks": len(ids), "retrieved_candidate_chunks": 0,
                "unresolved_references": sum(r["target_kind"] == "unresolved" for c in current for r in c["references"]),
            })
        associations = []
        for row in evidence_rows:
            if row["entity_type"] not in ("IMPLEMENTATION", "EVALUATION") or not row.get("linked_requirement"):
                continue
            targets = names.get(row["linked_requirement"], [])
            associations.append({"evidence_row_id": row["id"], "requirement_row_id": targets[0] if len(targets) == 1 else None,
                "kind": "REFERENCES_REQUIREMENT" if row["entity_type"] == "IMPLEMENTATION" else "ASSESSES_REQUIREMENT",
                "unresolved_value": row["linked_requirement"] if len(targets) != 1 else None,
                "candidate_ids": targets if len(targets) != 1 else []})
        revision = conn.execute("SELECT COUNT(*) FROM chunk_snapshot WHERE origin='legacy_hydration'").fetchone()[0]
        return {"schema_version": 2, "dataset_uuid": self.repo.dataset_uuid(), "run_id": "legacy",
                "snapshot_key": f"{self.repo.dataset_uuid()}:legacy:{len(evidence_rows)}:{len(corr_rows)}:{revision}",
                "legacy": True, "requirements": requirements, "evidence": evidence,
                "chunks": chunks, "correlations": {c["correlation_id"]: c for c in corr_rows},
                "candidates": {}, "associations": associations, "unlinked_assessments": unlinked,
                "not_in_run": [], "outcome": "legacy"}
