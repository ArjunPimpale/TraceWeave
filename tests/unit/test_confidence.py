"""Tests for the confidence scorer."""

from __future__ import annotations

import pytest

from pecs.correlation.confidence import ConfidenceScorer
from pecs.models.correlation_result import CorrelationResult, CorrelationStatus


@pytest.fixture
def scorer():
    return ConfidenceScorer()


def _make_corr(status=CorrelationStatus.IMPLEMENTED_AND_VALIDATED,
               method="deterministic_rule", chunks=None):
    return CorrelationResult(
        requirement_entity_id="R1",
        evidence_entity_id="impl-R1",
        status=status,
        resolution_method=method,
        confidence=0.0,  # Will be set by scorer
        supporting_chunk_ids=chunks or [],
    )


def _make_evidence(chunk_id, source_type="PYTHON"):
    return {"chunk_id": chunk_id, "source_type": source_type, "entity_type": "IMPLEMENTATION"}


class TestConfidenceScorer:
    def test_score_in_valid_range(self, scorer):
        corr = _make_corr()
        score = scorer.score(corr, retrieval_score=0.8)
        assert 0.0 <= score <= 1.0

    def test_deterministic_rule_higher_than_llm(self, scorer):
        rule_corr = _make_corr(method="deterministic_rule")
        llm_corr = _make_corr(method="llm_stage2")
        rule_score = scorer.score(rule_corr, retrieval_score=0.7)
        llm_score = scorer.score(llm_corr, retrieval_score=0.7)
        assert rule_score > llm_score

    def test_more_evidence_increases_score(self, scorer):
        corr_no_ev = _make_corr(chunks=[])
        corr_with_ev = _make_corr(chunks=["c1", "c2", "c3"])
        evidence = [_make_evidence("c1"), _make_evidence("c2"), _make_evidence("c3")]

        score_no_ev = scorer.score(corr_no_ev, retrieval_score=0.5, supporting_evidence=[])
        score_with_ev = scorer.score(corr_with_ev, retrieval_score=0.5, supporting_evidence=evidence)
        assert score_with_ev > score_no_ev

    def test_source_diversity_increases_score(self, scorer):
        chunks_single = ["c1", "c2"]
        chunks_diverse = ["c3", "c4"]

        ev_single = [_make_evidence("c1", "PYTHON"), _make_evidence("c2", "PYTHON")]
        ev_diverse = [_make_evidence("c3", "PYTHON"), _make_evidence("c4", "PDF")]

        corr1 = _make_corr(chunks=chunks_single)
        corr2 = _make_corr(chunks=chunks_diverse)

        s1 = scorer.score(corr1, retrieval_score=0.7, supporting_evidence=ev_single)
        s2 = scorer.score(corr2, retrieval_score=0.7, supporting_evidence=ev_diverse)
        assert s2 >= s1  # Diverse sources should score >= single source

    def test_high_retrieval_score_increases_confidence(self, scorer):
        corr = _make_corr()
        low_score = scorer.score(corr, retrieval_score=0.1)
        high_score = scorer.score(corr, retrieval_score=0.9)
        assert high_score > low_score

    def test_claimed_but_no_evidence_lower_than_implemented(self, scorer):
        claimed = _make_corr(status=CorrelationStatus.CLAIMED_BUT_NO_EVIDENCE)
        implemented = _make_corr(status=CorrelationStatus.IMPLEMENTED_AND_VALIDATED)
        ev = [_make_evidence("c1")]

        s_claimed = scorer.score(claimed, retrieval_score=0.7, supporting_evidence=ev)
        s_impl = scorer.score(implemented, retrieval_score=0.7, supporting_evidence=ev)
        assert s_impl > s_claimed

    def test_score_never_exceeds_1(self, scorer):
        corr = _make_corr(method="deterministic_rule")
        evidence = [_make_evidence(f"c{i}", src) for i, src in enumerate(["PYTHON", "PDF", "GIT", "DOCX"])]
        score = scorer.score(corr, retrieval_score=1.0, supporting_evidence=evidence)
        assert score <= 1.0

    def test_score_batch_updates_correlations(self, scorer):
        corrs = [_make_corr() for _ in range(3)]
        ev_all = [_make_evidence("c1"), _make_evidence("c2")]
        retrieval = {"R1": 0.75}

        updated = scorer.score_batch(corrs, retrieval, ev_all)
        assert len(updated) == 3
        assert all(c.confidence > 0.0 for c in updated)

    def test_confidence_rounded_to_4_decimals(self, scorer):
        corr = _make_corr()
        score = scorer.score(corr, retrieval_score=0.555)
        # Check it's a float with at most 4 decimal places
        assert score == round(score, 4)


class TestConfidenceEdgeCases:
    def test_zero_retrieval_score(self, scorer):
        corr = _make_corr()
        score = scorer.score(corr, retrieval_score=0.0, supporting_evidence=[])
        assert 0.0 <= score <= 1.0

    def test_requirement_not_implemented_confidence(self, scorer):
        corr = _make_corr(status=CorrelationStatus.REQUIREMENT_NOT_IMPLEMENTED)
        score = scorer.score(corr, retrieval_score=0.0, supporting_evidence=[])
        # Should be positive because deterministic rule has W_M component
        assert score >= 0.0
