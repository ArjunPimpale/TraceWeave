"""Tests for the deterministic rule engine (evidence-first revamp)."""

from __future__ import annotations

import pytest

from pecs.correlation.rules import (
    HIGH_SCORE_THRESHOLD,
    NO_EVIDENCE_FLOOR,
    RuleEngine,
    _is_negative_eval,
    _is_positive_eval,
)
from pecs.models.correlation_result import CorrelationStatus
from pecs.models.evidence_chunk import EvidenceChunk, SourceType
from pecs.models.retrieved_evidence import RetrievedEvidence


@pytest.fixture
def engine():
    return RuleEngine()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _req(entity_id, text="Some requirement text."):
    return {"entity_id": entity_id, "text": text, "chunk_id": f"chunk_{entity_id}"}


def _impl(entity_id, linked_req=None, text="Implementation evidence.", source_doc="code.py"):
    return {
        "entity_id": entity_id,
        "entity_type": "IMPLEMENTATION",
        "text": text,
        "linked_requirement": linked_req,
        "chunk_id": f"chunk_{entity_id}",
        "source_document": source_doc,
    }


def _eval(entity_id, linked_req=None, text="Evaluation feedback.", source_doc="report.pdf"):
    return {
        "entity_id": entity_id,
        "entity_type": "EVALUATION",
        "text": text,
        "linked_requirement": linked_req,
        "chunk_id": f"chunk_{entity_id}",
        "source_document": source_doc,
    }


def _candidate(score: float, text: str = "Some chunk text", req_id_in_text: str = "") -> list[RetrievedEvidence]:
    """Build a list[RetrievedEvidence] with one candidate at the given score."""
    full_text = text
    if req_id_in_text:
        full_text = f"{text} Implements {req_id_in_text}."
    chunk = EvidenceChunk(
        chunk_id="chunk1",
        source_document="doc1",
        source_hash="hash",
        source_type=SourceType.PYTHON,
        chunk_index=0,
        source_locator="line 1",
        normalized_text=full_text,
        char_count=len(full_text),
    )
    return [RetrievedEvidence(chunk=chunk, combined_score=score)]


def _no_candidates() -> list[RetrievedEvidence]:
    return []


# ── Rule 1: exact_requirement_id_match ───────────────────────────────────────

class TestRule1ExactIDMatch:
    def test_linked_impl_resolves_requirement(self, engine):
        req = _req("R1")
        impl = _impl("impl-R1", linked_req="R1")
        resolved, ambiguous = engine.apply_rules(
            [req], [impl], [], {"R1": _candidate(0.9)}
        )

        assert any(r.requirement_entity_id == "R1" for r in resolved)
        corr = next(r for r in resolved if r.requirement_entity_id == "R1")
        assert corr.rule_name == "exact_requirement_id_match"
        assert corr.status in (
            CorrelationStatus.IMPLEMENTED_AND_VALIDATED,
            CorrelationStatus.IMPLEMENTED_WITHOUT_EVALUATION,
        )

    def test_positive_eval_yields_validated(self, engine):
        req = _req("R2")
        impl = _impl("impl-R2", linked_req="R2")
        ev = _eval("eval-R2", linked_req="R2", text="Excellent implementation, well done. Score: 9/10")
        resolved, _ = engine.apply_rules([req], [impl], [ev], {"R2": _candidate(0.9)})

        corr = next(r for r in resolved if r.requirement_entity_id == "R2")
        assert corr.status == CorrelationStatus.IMPLEMENTED_AND_VALIDATED

    def test_negative_eval_yields_negatively_evaluated(self, engine):
        req = _req("R3")
        impl = _impl("impl-R3", linked_req="R3")
        ev = _eval("eval-R3", linked_req="R3", text="Poor implementation, incomplete and missing key features.")
        resolved, _ = engine.apply_rules([req], [impl], [ev], {"R3": _candidate(0.9)})

        corr = next(r for r in resolved if r.requirement_entity_id == "R3")
        assert corr.status == CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED


# ── Rule 2: strong_retrieval_with_id_mention ─────────────────────────────────

class TestRule2StrongRetrieval:
    def test_high_score_with_id_mention_resolves_deterministically(self, engine):
        req = _req("R10")
        # Score >= HIGH_SCORE_THRESHOLD and req ID in text
        candidates = _candidate(HIGH_SCORE_THRESHOLD + 0.05, req_id_in_text="R10")
        resolved, ambiguous = engine.apply_rules([req], [], [], {"R10": candidates})

        r10_resolved = [r for r in resolved if r.requirement_entity_id == "R10"]
        assert r10_resolved, "Should be deterministically resolved by R2"
        assert r10_resolved[0].rule_name == "strong_retrieval_with_id_mention"

    def test_high_score_without_id_mention_goes_to_ambiguous(self, engine):
        req = _req("R11")
        # High score but req ID NOT in text
        candidates = _candidate(HIGH_SCORE_THRESHOLD + 0.05, text="Generic implementation work")
        resolved, ambiguous = engine.apply_rules([req], [], [], {"R11": candidates})

        r11_resolved = [r for r in resolved if r.requirement_entity_id == "R11"]
        # Should NOT be resolved deterministically — goes to ambiguous (R4)
        assert not any(r.rule_name == "strong_retrieval_with_id_mention" for r in r11_resolved)


# ── Rule 3: requirement_with_no_evidence ─────────────────────────────────────

class TestRule3NoEvidence:
    def test_requirement_with_no_impl_and_low_score(self, engine):
        req = _req("R4")
        # Score below NO_EVIDENCE_FLOOR
        resolved, _ = engine.apply_rules(
            [req], [], [], {"R4": _candidate(NO_EVIDENCE_FLOOR - 0.05)}
        )

        assert any(r.requirement_entity_id == "R4" for r in resolved)
        corr = next(r for r in resolved if r.requirement_entity_id == "R4")
        assert corr.status == CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED
        assert corr.rule_name == "requirement_with_no_evidence"

    def test_requirement_with_no_candidates_at_all(self, engine):
        req = _req("R99")
        resolved, _ = engine.apply_rules([req], [], [], {})

        r99 = [r for r in resolved if r.requirement_entity_id == "R99"]
        assert r99
        assert r99[0].status == CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED


# ── Rule 4: ambiguous_with_retrieval_evidence ─────────────────────────────────

class TestRule4AmbiguousDeferral:
    def test_moderate_score_goes_to_ambiguous(self, engine):
        req = _req("R5")
        # Score between floor and high threshold → ambiguous
        score = (NO_EVIDENCE_FLOOR + HIGH_SCORE_THRESHOLD) / 2
        resolved, ambiguous = engine.apply_rules([req], [], [], {"R5": _candidate(score)})

        r5_resolved = [r for r in resolved if r.requirement_entity_id == "R5"]
        r5_ambiguous = [a for a in ambiguous if a.get("requirement", {}).get("entity_id") == "R5"]

        # Should be in ambiguous, not deterministically resolved
        assert r5_ambiguous, "Moderate score should go to ambiguous (Stage 2)"
        assert not r5_resolved


# ── Rule 5: orphan_evaluation ─────────────────────────────────────────────────

class TestRule5OrphanEvaluation:
    def test_unlinked_eval_flagged_as_orphan(self, engine):
        ev = _eval("eval-orphan", linked_req=None, text="Good overall project performance, well done.")
        resolved, _ = engine.apply_rules([], [], [ev], {})

        orphan_corrs = [r for r in resolved if r.status == CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT]
        assert orphan_corrs

    def test_linked_eval_not_orphaned(self, engine):
        req = _req("R1")
        impl = _impl("impl-R1", linked_req="R1")
        ev = _eval("eval-R1", linked_req="R1", text="Good work.")
        resolved, _ = engine.apply_rules([req], [impl], [ev], {"R1": _candidate(0.9)})

        orphan_corrs = [r for r in resolved if r.status == CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT]
        assert not orphan_corrs


# ── Sentiment detection ───────────────────────────────────────────────────────

class TestSentimentDetection:
    @pytest.mark.parametrize("text", [
        "Excellent implementation, well done.",
        "Good work, meets requirements.",
        "Score: 8/10",
        "Well-structured and correct.",
    ])
    def test_positive_markers_detected(self, text):
        assert _is_positive_eval(text) is True

    @pytest.mark.parametrize("text", [
        "Poor implementation, missing key features.",
        "Incomplete and inadequate.",
        "Score: 3/10",
        "Does not meet requirements.",
        "The module is absent from submission.",
    ])
    def test_negative_markers_detected(self, text):
        assert _is_negative_eval(text) is True


# ── Multiple requirements ─────────────────────────────────────────────────────

class TestMultipleRequirements:
    def test_multiple_requirements_independently_resolved(self, engine):
        reqs = [_req(f"R{i}") for i in range(1, 5)]
        impls = [_impl(f"impl-R{i}", linked_req=f"R{i}") for i in range(1, 4)]
        candidates = {f"R{i}": _candidate(0.9) for i in range(1, 4)}
        candidates["R4"] = _candidate(NO_EVIDENCE_FLOOR - 0.05)  # Below floor

        resolved, _ = engine.apply_rules(reqs, impls, [], candidates)
        resolved_ids = {r.requirement_entity_id for r in resolved}

        assert "R1" in resolved_ids
        assert "R2" in resolved_ids
        assert "R3" in resolved_ids
        assert "R4" in resolved_ids  # Resolved as NOT_IMPLEMENTED

    def test_deterministic_resolution_method(self, engine):
        req = _req("R1")
        impl = _impl("impl-R1", linked_req="R1")
        resolved, _ = engine.apply_rules([req], [impl], [], {"R1": _candidate(0.9)})

        for corr in resolved:
            assert corr.resolution_method == "deterministic_rule"
