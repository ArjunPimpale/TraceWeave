"""Additive schema for durable traceability runs and citations."""

from __future__ import annotations

import sqlite3
import uuid

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS traceability_dataset (
    id INTEGER PRIMARY KEY CHECK (id = 1), uuid TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traceability_run (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, dataset_uuid TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT NOT NULL, evidence_watermark INTEGER NOT NULL,
    settings_json TEXT NOT NULL, producer_version TEXT NOT NULL, schema_version INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('complete','complete_with_errors'))
);
CREATE TABLE IF NOT EXISTS chunk_snapshot (
    snapshot_id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL, source_hash TEXT,
    source_document TEXT, source_type TEXT, source_locator TEXT, chunk_index INTEGER,
    normalized_text TEXT NOT NULL, metadata_json TEXT NOT NULL, captured_at TEXT NOT NULL,
    origin TEXT NOT NULL CHECK(origin IN ('pipeline','legacy_hydration'))
);
CREATE INDEX IF NOT EXISTS idx_chunk_snapshot_chunk ON chunk_snapshot(chunk_id);
CREATE TABLE IF NOT EXISTS traceability_run_chunk (
    run_id TEXT NOT NULL REFERENCES traceability_run(run_id), chunk_id TEXT NOT NULL,
    snapshot_id TEXT REFERENCES chunk_snapshot(snapshot_id),
    availability TEXT NOT NULL CHECK(availability IN ('available','missing','conflict')),
    diagnostics_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(run_id,chunk_id)
);
CREATE TABLE IF NOT EXISTS correlation_context (
    correlation_id TEXT PRIMARY KEY REFERENCES correlation(correlation_id),
    run_id TEXT NOT NULL REFERENCES traceability_run(run_id),
    requirement_row_id INTEGER REFERENCES evidence(id),
    scope TEXT NOT NULL CHECK(scope IN ('requirement_assessment','evaluation_review','claim_review')),
    sequence INTEGER NOT NULL, UNIQUE(run_id,sequence)
);
CREATE INDEX IF NOT EXISTS idx_context_run_requirement ON correlation_context(run_id,requirement_row_id);
CREATE TABLE IF NOT EXISTS correlation_reference (
    reference_id TEXT PRIMARY KEY, correlation_id TEXT NOT NULL REFERENCES correlation_context(correlation_id),
    ordinal INTEGER NOT NULL, target_kind TEXT NOT NULL CHECK(target_kind IN ('evidence','chunk','unresolved')),
    evidence_row_id INTEGER REFERENCES evidence(id), chunk_id TEXT,
    unresolved_value TEXT, candidate_ids_json TEXT NOT NULL DEFAULT '[]',
    role TEXT NOT NULL, snapshot_id TEXT REFERENCES chunk_snapshot(snapshot_id), rank INTEGER,
    vector_score REAL, bm25_score REAL, combined_score REAL, retrieval_method TEXT,
    id_match INTEGER, visible_text TEXT, basis_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(correlation_id,ordinal),
    CHECK ((target_kind='evidence' AND evidence_row_id IS NOT NULL AND chunk_id IS NULL AND unresolved_value IS NULL)
       OR (target_kind='chunk' AND evidence_row_id IS NULL AND chunk_id IS NOT NULL AND unresolved_value IS NULL)
       OR (target_kind='unresolved' AND evidence_row_id IS NULL AND chunk_id IS NULL AND unresolved_value IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_reference_chunk ON correlation_reference(chunk_id);
CREATE TABLE IF NOT EXISTS traceability_run_requirement (
    run_id TEXT NOT NULL REFERENCES traceability_run(run_id),
    requirement_row_id INTEGER NOT NULL REFERENCES evidence(id),
    workflow_state TEXT NOT NULL CHECK(workflow_state IN
        ('assessed','stage2_disabled','classification_failed','retrieval_failed','ambiguous_reference')),
    selected_correlation_id TEXT REFERENCES correlation_context(correlation_id),
    diagnostics_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(run_id,requirement_row_id)
);
CREATE TABLE IF NOT EXISTS traceability_run_candidate (
    run_id TEXT NOT NULL, requirement_row_id INTEGER NOT NULL, chunk_id TEXT NOT NULL,
    rank INTEGER NOT NULL, vector_score REAL, bm25_score REAL, combined_score REAL,
    retrieval_method TEXT, id_match INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id,requirement_row_id,chunk_id),
    FOREIGN KEY(run_id,requirement_row_id) REFERENCES traceability_run_requirement(run_id,requirement_row_id),
    FOREIGN KEY(run_id,chunk_id) REFERENCES traceability_run_chunk(run_id,chunk_id)
);
"""


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return
    # No executescript inside an open transaction: sqlite3 executescript commits it.
    conn.executescript(DDL)
    with conn:
        conn.execute("INSERT OR IGNORE INTO traceability_dataset(id,uuid) VALUES (1,?)", (str(uuid.uuid4()),))
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
