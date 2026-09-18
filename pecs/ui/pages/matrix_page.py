"""
Traceability Matrix page.

Displays the full requirement → evidence traceability matrix with:
- Color-coded status cells
- Confidence scores
- Evidence count
- Source document citations
- Expandable reasoning text
- Export to CSV / JSON
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from pecs.logging_config import get_logger
from pecs.models.correlation_result import CorrelationStatus
from pecs.traceability.matrix import MatrixBuilder, STATUS_COLORS, TraceabilityMatrix

logger = get_logger(__name__)

# Status emoji mapping
STATUS_EMOJI = {
    "IMPLEMENTED_AND_VALIDATED": "✅",
    "IMPLEMENTED_WITHOUT_EVALUATION": "🔵",
    "IMPLEMENTED_BUT_NEGATIVELY_EVALUATED": "🟠",
    "PARTIALLY_IMPLEMENTED": "🟡",
    "CLAIMED_BUT_NO_EVIDENCE": "🟣",
    "EVALUATION_WITHOUT_REQUIREMENT": "⬜",
    "REQUIREMENT_NOT_IMPLEMENTED": "❌",
}


def render() -> None:
    st.title("📊 Traceability Matrix")
    st.markdown(
        "The full requirement-to-evidence traceability matrix. "
        "Each row represents one requirement and its correlation status."
    )

    # ── Load matrix ────────────────────────────────────────────────────────
    matrix = _load_matrix()

    if matrix is None or not matrix.rows:
        st.info("No matrix data found. Run the correlation pipeline first.")
        if st.button("🔗 Go to Correlate", type="primary"):
            st.session_state.page = "correlate"
            st.rerun()
        return

    # ── Summary metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requirements", matrix.total_requirements)
    col2.metric("Implemented", matrix.implemented_count)
    col3.metric("Coverage", f"{matrix.coverage_percentage}%")
    col4.metric("Avg. Confidence", f"{matrix.avg_confidence:.1%}")

    # ── Status distribution ────────────────────────────────────────────────
    with st.expander("📊 Status Distribution"):
        for status_val, count in matrix.status_distribution.items():
            emoji = STATUS_EMOJI.get(status_val, "❓")
            color = STATUS_COLORS.get(CorrelationStatus(status_val), "#888") if status_val != "NOT_ASSESSED" else "#888"
            pct = round(count / max(matrix.total_requirements, 1) * 100, 1)
            st.markdown(
                f"<span style='color:{color}'>{emoji} **{status_val}**</span>: {count} ({pct}%)",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Filters ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        filter_status = st.multiselect(
            "Filter by status",
            options=list(matrix.status_distribution.keys()),
            default=[],
            placeholder="All statuses",
        )
    with col2:
        min_confidence = st.slider("Min. confidence", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    rows = matrix.rows
    if filter_status:
        rows = [r for r in rows if (r.status.value if r.status else "NOT_ASSESSED") in filter_status]
    rows = [r for r in rows if r.confidence >= min_confidence]

    st.markdown(f"**Showing {len(rows)} of {matrix.total_requirements} requirements**")

    # ── Matrix table ───────────────────────────────────────────────────────
    for row in rows:
        status_text = row.status.value if row.status else "NOT_ASSESSED"
        emoji = STATUS_EMOJI.get(status_text, "❔")
        color = row.color
        with st.expander(
            f"{emoji} **{row.requirement_id}** — {status_text.replace('_', ' ')}"
            f" (confidence: {row.confidence:.0%})"
        ):
            st.markdown(f"**Requirement:** {row.requirement_text}")
            st.write(f"Status: {status_text}; workflow: {row.workflow_state}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{row.confidence:.0%}")
            col2.metric("Evidence pieces", row.evidence_count)
            col3.metric("Resolution", row.resolution_method.replace("_", " "))
            st.caption(f"Implementation entities: {row.implementation_entities} · Evaluation entities: {row.evaluation_entities} · Assessment context chunks: {row.assessment_context_chunks} · Retrieved candidates: {row.retrieved_candidate_chunks}")
            if row.conflicting:
                st.warning("Assessments disagree; inspect all decisions in the Graph.")
            if st.button("View trace", key=f"view_trace_{row.requirement_key}"):
                st.session_state.setdefault("trace_history", []).append({"page": "matrix"})
                st.session_state.trace_run = row.run_id
                st.session_state.graph_run_select = row.run_id
                st.session_state.trace_deep_link = row.requirement_key
                st.session_state.trace_next_mode = "Requirement focus"
                st.session_state.trace_clear_filters = True
                st.session_state.page = "graph"
                st.rerun()

            if row.source_documents:
                st.markdown("**Source documents:**")
                for doc in row.source_documents:
                    st.markdown(f"  - `{doc}`")

            if row.reasoning:
                st.markdown("**Reasoning:**")
                st.markdown(f"> {row.reasoning}")

    # ── Export ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Export")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download CSV", use_container_width=True):
            df = pd.DataFrame([r.to_dict() for r in matrix.rows])
            csv = df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                data=csv,
                file_name="traceability_matrix.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("Download JSON", use_container_width=True):
            json_data = json.dumps(matrix.to_dict(), indent=2)
            st.download_button(
                "💾 Download JSON",
                data=json_data,
                file_name="traceability_matrix.json",
                mime="application/json",
            )


def _load_matrix() -> TraceabilityMatrix | None:
    """Load the matrix from session state or SQLite."""
    try:
        from pecs.traceability.reader import TraceabilityReader
        matrix = MatrixBuilder._matrix_from_snapshot(
            TraceabilityReader().load(st.session_state.get("trace_run")))
        return matrix
    except Exception as exc:
        logger.warning(f"Could not load matrix: {exc}")
        return None
