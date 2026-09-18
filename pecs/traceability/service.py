"""Run existing correlation logic while recording its actual participants."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from pecs.correlation.confidence import ConfidenceScorer
from pecs.correlation.rules import RuleEngine, HIGH_SCORE_THRESHOLD
from pecs.correlation.stage2 import Stage2Classifier
from pecs.models.traceability import AssessmentEnvelope, AssessmentReference, RetrievalOutcome
from pecs.models.retrieved_evidence import RetrievedEvidence
from pecs.store.evidence_repo import EvidenceRepo
from pecs.store.traceability_repo import TraceabilityRepo
from pecs.traceability.provenance import ChunkSnapshot, from_chunk
from pecs.vectorstore.chroma_store import ChromaStore
from pecs.vectorstore.chunk_codec import reconstruct_evidence_chunk


def _candidate_ref(candidate: RetrievedEvidence, rank: int, role: str, visible_text: str | None = None) -> AssessmentReference:
    return AssessmentReference(
        role=role, target_kind="chunk", chunk_id=candidate.chunk_id, rank=rank,
        vector_score=candidate.vector_score, bm25_score=candidate.bm25_score,
        combined_score=candidate.combined_score, retrieval_method=candidate.retrieval_method,
        id_match=candidate.id_match, visible_text=visible_text,
    )


class TraceabilityService:
    def __init__(self, evidence_repo: EvidenceRepo | None = None, trace_repo: TraceabilityRepo | None = None,
                 rules: RuleEngine | None = None, stage2: Stage2Classifier | None = None,
                 scorer: ConfidenceScorer | None = None, chroma: ChromaStore | None = None) -> None:
        self.evidence_repo = evidence_repo or EvidenceRepo()
        self.trace_repo = trace_repo or TraceabilityRepo()
        self.rules = rules or RuleEngine()
        self.stage2 = stage2 or Stage2Classifier()
        self.scorer = scorer or ConfidenceScorer()
        self.chroma = chroma or ChromaStore()

    def run(self, outcomes: dict[int, RetrievalOutcome], use_stage2: bool = True) -> str:
        started = datetime.now(timezone.utc).isoformat()
        # Materialize the immutable input set before any slow retrieval/LLM work.
        all_evidence = self.evidence_repo.get_all()
        all_evidence.sort(key=lambda row: row["id"])
        watermark = max((row["id"] for row in all_evidence), default=0)
        requirements = [row for row in all_evidence if row["entity_type"] == "REQUIREMENT"]
        implementations = [row for row in all_evidence if row["entity_type"] == "IMPLEMENTATION"]
        evaluations = [row for row in all_evidence if row["entity_type"] == "EVALUATION"]
        req_by_name: dict[str, list[dict]] = defaultdict(list)
        for req in requirements:
            req_by_name[req["entity_id"]].append(req)
        by_chunk: dict[str, list[dict]] = defaultdict(list)
        for ev in all_evidence:
            by_chunk[ev["chunk_id"]].append(ev)

        envelopes: list[AssessmentEnvelope] = []
        states: dict[int, tuple[str, list[str]]] = {}
        candidates_by_row: dict[int, list[RetrievedEvidence]] = {}
        sequence = 0
        for req in requirements:
            req_row = req["id"]
            outcome = outcomes.get(req_row, RetrievalOutcome([], "No retrieval outcome"))
            candidates = list(outcome.candidates)
            candidates_by_row[req_row] = candidates
            ambiguous_links = [ev for ev in implementations + evaluations
                               if ev.get("linked_requirement") == req["entity_id"]
                               and len(req_by_name[req["entity_id"]]) != 1]
            diagnostics = [f"Ambiguous explicit reference from evidence row {ev['id']}" for ev in ambiguous_links]
            if outcome.error:
                diagnostics.append(f"Retrieval failed: {outcome.error}")
            linked_impl = [ev for ev in implementations if ev.get("linked_requirement") == req["entity_id"]
                           and len(req_by_name[req["entity_id"]]) == 1]
            linked_eval = [ev for ev in evaluations if ev.get("linked_requirement") == req["entity_id"]
                           and len(req_by_name[req["entity_id"]]) == 1]
            # Unlinked evaluations are handled once below, not once per requirement.
            resolved, ambiguous = self.rules.apply_rules(
                [req], linked_impl, linked_eval,
                {req["entity_id"]: candidates if not outcome.error else []},
            )
            if outcome.error and not linked_impl:
                resolved = [c for c in resolved if c.rule_name != "requirement_with_no_evidence"]
            for corr in resolved:
                refs: list[AssessmentReference] = []
                if corr.rule_name == "exact_requirement_id_match":
                    refs.extend(AssessmentReference("explicit_implementation", "evidence", evidence_row_id=ev["id"]) for ev in linked_impl)
                    refs.extend(AssessmentReference("explicit_evaluation", "evidence", evidence_row_id=ev["id"]) for ev in linked_eval)
                elif corr.rule_name == "strong_semantic_match":
                    refs.extend(_candidate_ref(c, rank, "rule_selected_chunk") for rank, c in enumerate(candidates, 1)
                                if c.combined_score >= HIGH_SCORE_THRESHOLD)
                    refs.extend(AssessmentReference("explicit_evaluation", "evidence", evidence_row_id=ev["id"]) for ev in linked_eval)
                sequence += 1
                envelopes.append(AssessmentEnvelope(corr, req_row, "requirement_assessment", sequence, refs))
            if use_stage2:
                for pair in ambiguous:
                    corr = self.stage2.classify(pair)
                    if corr is None:
                        diagnostics.append("Stage 2 classification failed")
                        continue
                    refs = [_candidate_ref(c, rank, "stage2_context", c.chunk.normalized_text[:800])
                            for rank, c in enumerate(pair.get("retrieval_candidates", []), 1)]
                    if pre := pair.get("evidence"):
                        refs.append(AssessmentReference("stage2_context", "evidence", evidence_row_id=pre["id"], visible_text=pre.get("text", "")[:800]))
                    sequence += 1
                    scope = "claim_review" if pair.get("type") == "verbal_claim_without_evidence" else (
                        "evaluation_review" if pair.get("type") == "orphan_evaluation_with_candidate_req" else "requirement_assessment")
                    envelopes.append(AssessmentEnvelope(corr, req_row, scope, sequence, refs))
            elif ambiguous:
                diagnostics.append("Stage 2 disabled")
            current = [e for e in envelopes if e.requirement_row_id == req_row]
            state = "assessed" if current else ("retrieval_failed" if outcome.error else
                    "stage2_disabled" if ambiguous and not use_stage2 else
                    "classification_failed" if ambiguous else "ambiguous_reference" if diagnostics else "classification_failed")
            states[req_row] = state, diagnostics

        # Unlinked evaluations are independent of a requirement decision.
        orphan_resolved: list = []
        orphan_ambiguous: list = []
        self.rules._apply_orphan_evaluations(requirements, evaluations, orphan_resolved, orphan_ambiguous)
        for corr in orphan_resolved:
            targets = [ev for ev in evaluations if ev["entity_id"] == corr.evidence_entity_id]
            refs = [AssessmentReference("explicit_evaluation", "evidence", evidence_row_id=ev["id"]) for ev in targets]
            sequence += 1
            envelopes.append(AssessmentEnvelope(corr, None, "evaluation_review", sequence, refs))
        if use_stage2:
            for pair in orphan_ambiguous:
                req_candidates = req_by_name.get(pair["requirement"]["entity_id"], [])
                if len(req_candidates) != 1:
                    continue
                corr = self.stage2.classify(pair)
                if corr is None:
                    continue
                sequence += 1
                envelopes.append(AssessmentEnvelope(
                    corr, req_candidates[0]["id"], "evaluation_review", sequence,
                    [AssessmentReference("stage2_context", "evidence", evidence_row_id=pair["evidence"]["id"],
                                         visible_text=pair["evidence"].get("text", "")[:800])],
                ))

        for req in requirements:
            state, diagnostics = states[req["id"]]
            if any(e.requirement_row_id == req["id"] for e in envelopes) and state not in ("assessed", "ambiguous_reference"):
                states[req["id"]] = ("assessed", diagnostics)

        # Keep the scorer unchanged; only replace the result in each envelope.
        scoring_chunk_to_evidence = {
            row["chunk_id"]: row for row in requirements + implementations + evaluations
        }
        for envelope in envelopes:
            req_candidates = candidates_by_row.get(envelope.requirement_row_id or -1, [])
            corr = envelope.result
            corr.confidence = self.scorer.score(corr, retrieval_candidates=req_candidates,
                supporting_evidence=[scoring_chunk_to_evidence[cid] for cid in corr.supporting_chunk_ids
                                     if cid in scoring_chunk_to_evidence])

        snapshots: dict[str, ChunkSnapshot | None] = {}
        conflicts: set[str] = set()
        candidate_chunks = {candidate.chunk_id: candidate.chunk for values in candidates_by_row.values() for candidate in values}
        chunk_ids = {ev["chunk_id"] for ev in all_evidence} | set(candidate_chunks)
        for envelope in envelopes:
            chunk_ids.update(envelope.result.supporting_chunk_ids)
            chunk_ids.update(ref.chunk_id for ref in envelope.references if ref.chunk_id)
        try:
            raw = self.chroma.get_by_ids(sorted(chunk_ids)) if chunk_ids else {"ids": [], "documents": [], "metadatas": []}
        except Exception:
            # Candidate chunks are still available in memory; all other citations
            # remain explicit missing references in the committed run.
            raw = {"ids": [], "documents": [], "metadatas": []}
        persisted = {cid: reconstruct_evidence_chunk(cid, text, meta)
                     for cid, text, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", []))}
        for cid in sorted(chunk_ids):
            persisted_chunk = persisted.get(cid)
            candidate_chunk = candidate_chunks.get(cid)
            if persisted_chunk and candidate_chunk and from_chunk(persisted_chunk).snapshot_id != from_chunk(candidate_chunk).snapshot_id:
                conflicts.add(cid)
                snapshots[cid] = None
                continue
            chosen = persisted_chunk or candidate_chunk
            snapshots[cid] = from_chunk(chosen) if chosen else None
        return self.trace_repo.save_run(
            requirements=requirements, envelopes=envelopes, candidates=candidates_by_row,
            snapshots=snapshots, conflicts=conflicts, states=states, evidence_watermark=watermark,
            started_at=started, settings={"use_stage2": use_stage2},
        )
