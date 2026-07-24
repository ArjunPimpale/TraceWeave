"""
Correlate page — runs the full correlation pipeline.

1. Rule engine (deterministic)
2. Stage 2 LLM classifier (ambiguous pairs)
3. Confidence scoring
4. Stores results in SQLite
"""

from __future__ import annotations

import streamlit as st

from pecs.logging_config import get_logger
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
    """Run the full correlation pipeline."""
    with st.spinner("Running correlation pipeline…"):
        try:
            builder = MatrixBuilder()
            matrix = builder.build(use_stage2=use_stage2)

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

            st.session_state.matrix_built = True
            st.session_state.last_matrix = matrix
            st.info("Navigate to 📊 Traceability Matrix to view the full results.")

        except Exception as exc:
            st.error(f"Correlation pipeline failed: {exc}")
            logger.error("Correlation pipeline error", extra={"context": {"error": str(exc)}})
