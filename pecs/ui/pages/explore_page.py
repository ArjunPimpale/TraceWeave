"""
Explore Evidence page — semantic search over the evidence store.

Allows querying ChromaDB with free-text queries.
Shows retrieved chunks with scores, source locations, and text.
"""

from __future__ import annotations

import streamlit as st

from pecs.embeddings.embedder import Embedder
from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.retrieval.metadata_filter import MetadataFilter
from pecs.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)


def render() -> None:
    st.title("🔎 Explore Evidence")
    st.markdown(
        "Search the evidence store using semantic similarity. "
        "Combines vector search (nomic-embed-text) and BM25 keyword matching."
    )

    # ── Search form ────────────────────────────────────────────────────────
    with st.form("search_form"):
        query = st.text_area(
            "Search query",
            placeholder="e.g., 'GNN training pipeline implementation' or 'professor feedback on module'",
            height=80,
        )

        col1, col2 = st.columns(2)
        with col1:
            source_type_filter = st.multiselect(
                "Filter by source type",
                options=[s.value for s in SourceType],
                default=[],
                placeholder="All types",
            )
        with col2:
            top_k = st.slider("Max results", 3, 20, 10)

        submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    if submitted and query.strip():
        _run_search(query, source_type_filter, top_k)


def _run_search(query: str, source_type_filter: list[str], top_k: int) -> None:
    """Execute retrieval and display results."""
    with st.spinner("Searching…"):
        try:
            pipeline = RetrievalPipeline()
            metadata_filter = None
            if source_type_filter:
                metadata_filter = MetadataFilter(source_types=source_type_filter)

            results = pipeline.retrieve(
                query_text=query,
                metadata_filter=metadata_filter,
                top_k=top_k,
            )

            st.session_state.query_results = results

        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

    if not st.session_state.query_results:
        st.info("No results found. Try a different query or check that documents are ingested.")
        return

    results = st.session_state.query_results
    st.markdown(f"**Found {len(results)} results**")
    st.divider()

    for i, result in enumerate(results):
        chunk = result.chunk
        score_label = f"Score: {result.combined_score:.3f}"
        method_label = f"[{result.retrieval_method}]"

        with st.expander(
            f"#{i + 1} — `{chunk.source_document}` | {chunk.source_locator} | {score_label} {method_label}"
        ):
            # Score breakdown
            col1, col2, col3 = st.columns(3)
            col1.metric("Combined score", f"{result.combined_score:.3f}")
            col2.metric("Vector score", f"{result.vector_score:.3f}")
            col3.metric(
                "BM25 score",
                f"{result.bm25_score:.3f}" if result.bm25_score is not None else "N/A",
            )

            if result.id_match:
                st.badge("🎯 Requirement ID Match", color="green")

            st.markdown("**Source:**")
            st.caption(f"📄 {chunk.source_document} | 📍 {chunk.source_locator} | 🏷️ {chunk.source_type.value}")

            st.markdown("**Text:**")
            st.text_area(
                label="Chunk text",
                value=chunk.normalized_text,
                height=200,
                key=f"chunk_text_{i}",
                disabled=True,
                label_visibility="collapsed",
            )
