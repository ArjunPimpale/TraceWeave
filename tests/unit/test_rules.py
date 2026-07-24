"""Tests for the deterministic rule engine."""

from __future__ import annotations

import pytest

from pecs.correlation.rules import RuleEngine, _is_positive_eval, _is_negative_eval
from pecs.models.correlation_result import CorrelationStatus


@pytest.fixture
def engine():
    return RuleEngine()


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


class TestRule1ExactIDMatch:
    def test_linked_impl_resolves_requirement(self, engine):
        req = _req("R1")
        impl = _impl("impl-R1", linked_req="R1")
        resolved, ambiguous = engine.apply_rules([req], [impl], [], {"R1": 0.9})

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
        resolved, _ = engine.apply_rules([req], [impl], [ev], {"R2": 0.9})

        corr = next(r for r in resolved if r.requirement_entity_id == "R2")
        assert corr.status == CorrelationStatus.IMPLEMENTED_AND_VALIDATED

    def test_negative_eval_yields_negatively_evaluated(self, engine):
        req = _req("R3")
        impl = _impl("impl-R3", linked_req="R3")
        ev = _eval("eval-R3", linked_req="R3", text="Poor implementation, incomplete and missing key features.")
        resolved, _ = engine.apply_rules([req], [impl], [ev], {"R3": 0.9})

        corr = next(r for r in resolved if r.requirement_entity_id == "R3")
        assert corr.status == CorrelationStatus.IMPLEMENTED_BUT_NEGATIVELY_EVALUATED


class TestRule2NoEvidence:
    def test_requirement_with_no_impl_and_low_score(self, engine):
        req = _req("R4")
        resolved, _ = engine.apply_rules([req], [], [], {"R4": 0.1})

        assert any(r.requirement_entity_id == "R4" for r in resolved)
        corr = next(r for r in resolved if r.requirement_entity_id == "R4")
        assert corr.status == CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED
        assert corr.rule_name == "requirement_with_no_evidence"

    def test_requirement_with_high_retrieval_score_not_resolved(self, engine):
        """If retrieval score is high but no explicit link, goes to ambiguous."""
        req = _req("R5")
        resolved, ambiguous = engine.apply_rules([req], [], [], {"R5": 0.7})

        # Not deterministically resolved as NOT_IMPLEMENTED because retrieval score is high
        r5_resolved = [r for r in resolved if r.requirement_entity_id == "R5"]
        # May be in ambiguous instead
        assert not r5_resolved or r5_resolved[0].status != CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED


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
        resolved, _ = engine.apply_rules([req], [impl], [ev], {"R1": 0.9})

        # Eval linked to R1 should not produce EVALUATION_WITHOUT_REQUIREMENT
        orphan_corrs = [r for r in resolved if r.status == CorrelationStatus.EVALUATION_WITHOUT_REQUIREMENT]
        assert not orphan_corrs


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


class TestMultipleRequirements:
    def test_multiple_requirements_independently_resolved(self, engine):
        reqs = [_req(f"R{i}") for i in range(1, 5)]
        impls = [_impl(f"impl-R{i}", linked_req=f"R{i}") for i in range(1, 4)]
        scores = {f"R{i}": 0.9 for i in range(1, 4)}
        scores["R4"] = 0.1

        resolved, _ = engine.apply_rules(reqs, impls, [], scores)
        resolved_ids = {r.requirement_entity_id for r in resolved}

        assert "R1" in resolved_ids
        assert "R2" in resolved_ids
        assert "R3" in resolved_ids
        assert "R4" in resolved_ids  # Resolved as NOT_IMPLEMENTED

    def test_deterministic_resolution_method(self, engine):
        req = _req("R1")
        impl = _impl("impl-R1", linked_req="R1")
        resolved, _ = engine.apply_rules([req], [impl], [], {"R1": 0.9})

        for corr in resolved:
            assert corr.resolution_method == "deterministic_rule"
