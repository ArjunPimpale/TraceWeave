"""
Correlate page — runs the full correlation pipeline.

Pipeline:
1. Load requirements from SQLite
2. Run retrieval pipeline to get top-K evidence candidates per requirement
3. Rule engine (deterministic, evidence-first)
4. Stage 2 LLM classifier (ambiguous pairs with real evidence)
5. Confidence scoring with real retrieval scores
6. Store results in SQLite
"""

from __future__ import annotations

import streamlit as st

from pecs.embeddings.embedder import Embedder
from pecs.logging_config import get_logger
from pecs.models.traceability import RetrievalOutcome
from pecs.retrieval.pipeline import RetrievalPipeline
from pecs.store.evidence_repo import EvidenceRepo
from pecs.traceability.matrix import MatrixBuilder

logger = get_logger(__name__)


def render() -> None:
    st.title("🔗 Correlate Requirements")
    st.markdown(
        "Run the correlation pipeline to map requirements to implementation evidence "
        "and evaluation feedback."
    )

    repo = EvidenceRepo()
    req_count = len(repo.get_all_requirements())
    impl_count = len(repo.get_all_implementations())
    eval_count = len(repo.get_all_evaluations())

    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Requirements", req_count)
    col2.metric("⚙️ Implementations", impl_count)
    col3.metric("📝 Evaluations", eval_count)

    if req_count == 0:
        st.warning("⚠️ No requirements found. Run extraction first.")
        return

    st.divider()

    # ── Options ────────────────────────────────────────────────────────────
    use_stage2 = st.checkbox(
        "Use Stage 2 LLM classifier for ambiguous pairs",
        value=True,
        help="Disabling this uses only deterministic rules (faster but less accurate).",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Run Correlation Pipeline", type="primary", use_container_width=True):
            _run_correlation(use_stage2)


def _run_correlation(use_stage2: bool) -> None:
    """Run the full correlation pipeline with retrieval wired in."""
    with st.spinner("Running correlation pipeline…"):
        try:
            repo = EvidenceRepo()
            requirements = repo.get_all_requirements()

            if not requirements:
                st.warning("No requirements to correlate.")
                return

            # ── Step 1: Run retrieval pipeline for each requirement ────────
            retrieval_candidates: dict[int, RetrievalOutcome] = {}

            with st.spinner(f"Retrieving evidence for {len(requirements)} requirements…"):
                pipeline = RetrievalPipeline()
                for req in requirements:
                    req_id = req["entity_id"]
                    req_text = req.get("text", "")
                    if not req_text:
                        retrieval_candidates[req["id"]] = RetrievalOutcome([], "Requirement has no text")
                        continue
                    retrieval_candidates[req["id"]] = pipeline.retrieve_outcome(
                        query_text=req_text, requirement_ids=[req_id], top_k=10)

            logger.info(
                "Retrieval complete",
                extra={"context": {
                    "requirements_retrieved": len(retrieval_candidates),
                    "total_requirements": len(requirements),
                }},
            )

            # ── Step 2: Build traceability matrix ─────────────────────────
            builder = MatrixBuilder()
            matrix = builder.build(
                retrieval_candidates=retrieval_candidates,
                use_stage2=use_stage2,
            )

            st.success("✅ Correlation complete!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Requirements", matrix.total_requirements)
            col2.metric("✅ Implemented", matrix.implemented_count)
            col3.metric("📈 Coverage", f"{matrix.coverage_percentage}%")
            col4.metric("🎯 Avg. Confidence", f"{matrix.avg_confidence:.1%}")

            # Status breakdown
            st.subheader("Status Distribution")
            for status, count in matrix.status_distribution.items():
                st.text(f"  {status}: {count}")

            # Resolution method breakdown
            det_count = matrix.summary.get("deterministic_resolved", 0)
            llm_count = matrix.summary.get("llm_resolved", 0)
            st.caption(f"Resolved: {det_count} deterministic · {llm_count} via LLM")

            st.session_state.matrix_built = True
            st.session_state.last_matrix = None
            st.session_state.trace_run = matrix.run_id
            st.session_state.graph_run_select = matrix.run_id
            st.info("Navigate to 📊 Traceability Matrix to view the full results.")

        except Exception as exc:
            st.error(f"Correlation pipeline failed: {exc}")
            logger.error("Correlation pipeline error", extra={"context": {"error": str(exc)}})
