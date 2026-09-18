"""Atomic run persistence and typed reads. The old evidence/correlation rows stay intact."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from pecs.models.traceability import AssessmentEnvelope
from pecs.models.retrieved_evidence import RetrievedEvidence
from pecs.store.database import Database, get_database
from pecs.store.correlation_repo import _INSERT_CORRELATION_SQL, _insert_params
from pecs.traceability.provenance import ChunkSnapshot


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class TraceabilityRepo:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.db.initialize_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        return self.db.connect()

    def dataset_uuid(self) -> str:
        return self.conn.execute("SELECT uuid FROM traceability_dataset WHERE id=1").fetchone()[0]

    def save_run(
        self, *, requirements: list[dict], envelopes: list[AssessmentEnvelope],
        candidates: dict[int, list[RetrievedEvidence]], snapshots: dict[str, ChunkSnapshot | None],
        conflicts: set[str] | None = None,
        states: dict[int, tuple[str, list[str]]], evidence_watermark: int,
        started_at: str, settings: dict[str, Any] | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        conflicts = conflicts or set()
        finished = datetime.now(timezone.utc).isoformat()
        outcome = "complete_with_errors" if any(
            state in ("classification_failed", "retrieval_failed", "ambiguous_reference")
            for state, _ in states.values()
        ) else "complete"
        by_req: dict[int, list[AssessmentEnvelope]] = {}
        for envelope in envelopes:
            if envelope.requirement_row_id is not None:
                by_req.setdefault(envelope.requirement_row_id, []).append(envelope)
        with self.conn:
            self.conn.execute(
                "INSERT INTO traceability_run(run_id,dataset_uuid,started_at,finished_at,evidence_watermark,settings_json,producer_version,schema_version,outcome) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, self.dataset_uuid(), started_at, finished, evidence_watermark,
                 _json(settings or {}), "traceability-v2", 2, outcome),
            )
            for chunk_id, snapshot in sorted(snapshots.items()):
                if snapshot:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO chunk_snapshot(snapshot_id,chunk_id,source_hash,source_document,source_type,source_locator,chunk_index,normalized_text,metadata_json,captured_at,origin) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (snapshot.snapshot_id, snapshot.chunk_id, snapshot.source_hash, snapshot.source_document,
                         snapshot.source_type, snapshot.source_locator, snapshot.chunk_index,
                         snapshot.normalized_text, _json(snapshot.metadata), finished, snapshot.origin),
                    )
                self.conn.execute(
                    "INSERT INTO traceability_run_chunk(run_id,chunk_id,snapshot_id,availability) VALUES (?,?,?,?)",
                    (run_id, chunk_id, snapshot.snapshot_id if snapshot else None,
                     "conflict" if chunk_id in conflicts else "available" if snapshot else "missing"),
                )
            for envelope in envelopes:
                self.conn.execute(_INSERT_CORRELATION_SQL, _insert_params(envelope.result))
                self.conn.execute(
                    "INSERT INTO correlation_context(correlation_id,run_id,requirement_row_id,scope,sequence) VALUES (?,?,?,?,?)",
                    (envelope.result.correlation_id, run_id, envelope.requirement_row_id,
                     envelope.scope, envelope.sequence),
                )
                for ordinal, ref in enumerate(envelope.references):
                    snap = snapshots.get(ref.chunk_id or "")
                    self.conn.execute(
                        """INSERT INTO correlation_reference(reference_id,correlation_id,ordinal,target_kind,
                        evidence_row_id,chunk_id,unresolved_value,candidate_ids_json,role,snapshot_id,rank,
                        vector_score,bm25_score,combined_score,retrieval_method,id_match,visible_text)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (str(uuid.uuid4()), envelope.result.correlation_id, ordinal, ref.target_kind,
                         ref.evidence_row_id, ref.chunk_id, ref.unresolved_value, _json(ref.candidates),
                         ref.role, snap.snapshot_id if snap else None, ref.rank, ref.vector_score,
                         ref.bm25_score, ref.combined_score, ref.retrieval_method,
                         int(ref.id_match) if ref.id_match is not None else None, ref.visible_text),
                    )
            for req in requirements:
                req_id = req["id"]
                assessments = by_req.get(req_id, [])
                selected = min(assessments, key=lambda e: (-e.result.confidence, e.sequence, e.result.correlation_id)) if assessments else None
                state, diagnostics = states.get(req_id, ("assessed" if selected else "classification_failed", []))
                self.conn.execute(
                    "INSERT INTO traceability_run_requirement(run_id,requirement_row_id,workflow_state,selected_correlation_id,diagnostics_json) VALUES (?,?,?,?,?)",
                    (run_id, req_id, state, selected.result.correlation_id if selected else None, _json(diagnostics)),
                )
                seen: set[str] = set()
                for rank, candidate in enumerate(candidates.get(req_id, []), 1):
                    if candidate.chunk_id in seen:
                        continue
                    seen.add(candidate.chunk_id)
                    self.conn.execute(
                        "INSERT INTO traceability_run_candidate(run_id,requirement_row_id,chunk_id,rank,vector_score,bm25_score,combined_score,retrieval_method,id_match) VALUES (?,?,?,?,?,?,?,?,?)",
                        (run_id, req_id, candidate.chunk_id, rank, candidate.vector_score, candidate.bm25_score,
                         candidate.combined_score, candidate.retrieval_method, int(candidate.id_match)),
                    )
        return run_id

    def list_runs(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM traceability_run ORDER BY id DESC")]

    def read_run(self, run_id: str) -> dict[str, list[dict] | dict]:
        run = self.conn.execute("SELECT * FROM traceability_run WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise KeyError(run_id)
        queries = {
            "requirements": "SELECT rr.*,e.* FROM traceability_run_requirement rr JOIN evidence e ON e.id=rr.requirement_row_id WHERE rr.run_id=? ORDER BY e.id",
            "evidence": "SELECT * FROM evidence WHERE id<=? ORDER BY id",
            "correlations": "SELECT c.*,x.requirement_row_id,x.scope,x.sequence FROM correlation_context x JOIN correlation c ON c.correlation_id=x.correlation_id WHERE x.run_id=? ORDER BY x.sequence",
            "references": "SELECT r.* FROM correlation_reference r JOIN correlation_context x ON x.correlation_id=r.correlation_id WHERE x.run_id=? ORDER BY x.sequence,r.ordinal",
            "candidates": "SELECT * FROM traceability_run_candidate WHERE run_id=? ORDER BY requirement_row_id,rank",
            "chunks": "SELECT rc.*,s.source_hash,s.source_document,s.source_type,s.source_locator,s.chunk_index,s.normalized_text,s.metadata_json FROM traceability_run_chunk rc LEFT JOIN chunk_snapshot s ON s.snapshot_id=rc.snapshot_id WHERE rc.run_id=?",
        }
        result: dict = {"run": dict(run)}
        for key, sql in queries.items():
            param = run["evidence_watermark"] if key == "evidence" else run_id
            result[key] = [dict(row) for row in self.conn.execute(sql, (param,))]
        return result

    def hydrate_legacy(self, chroma) -> int:
        """Capture available old citations without asserting historical equivalence."""
        from pecs.traceability.provenance import from_chunk
        from pecs.vectorstore.chunk_codec import reconstruct_evidence_chunk
        evidence_chunks = {r[0] for r in self.conn.execute("SELECT chunk_id FROM evidence")}
        old = self.conn.execute("""SELECT supporting_chunk_ids FROM correlation c
            LEFT JOIN correlation_context x ON x.correlation_id=c.correlation_id
            WHERE x.correlation_id IS NULL""")
        for row in old:
            try:
                evidence_chunks.update(json.loads(row[0] or "[]"))
            except (ValueError, TypeError):
                continue
        if not evidence_chunks:
            return 0
        records = chroma.get_by_ids(sorted(evidence_chunks))
        captured = datetime.now(timezone.utc).isoformat()
        count = 0
        with self.conn:
            for cid, text, metadata in zip(records.get("ids", []), records.get("documents", []),
                                           records.get("metadatas", [])):
                snap = from_chunk(reconstruct_evidence_chunk(cid, text, metadata), "legacy_hydration")
                cursor = self.conn.execute("""INSERT OR IGNORE INTO chunk_snapshot
                    (snapshot_id,chunk_id,source_hash,source_document,source_type,source_locator,
                     chunk_index,normalized_text,metadata_json,captured_at,origin)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (snap.snapshot_id, snap.chunk_id, snap.source_hash, snap.source_document,
                     snap.source_type, snap.source_locator, snap.chunk_index, snap.normalized_text,
                     _json(snap.metadata), captured, snap.origin))
                count += cursor.rowcount
        return count
