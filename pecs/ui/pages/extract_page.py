"""
Extract Evidence page.

Runs Stage 1 extraction (Phi-4-mini) on all ingested chunks
and stores results in the SQLite evidence table.
"""

from __future__ import annotations

import streamlit as st

from pecs.extraction.extractor import Extractor
from pecs.logging_config import get_logger
from pecs.store.evidence_repo import EvidenceRepo
from pecs.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


def render() -> None:
    st.title("🔬 Extract Evidence")
    st.markdown(
        "Run Stage 1 extraction using **Phi-4-mini** to identify requirements, "
        "implementations, and evaluations from ingested documents."
    )

    # ── Status check ───────────────────────────────────────────────────────
    chroma = ChromaStore()
    chunk_count = chroma.count()
    repo = EvidenceRepo()
    evidence_count = len(repo.get_all())

    col1, col2 = st.columns(2)
    col1.metric("📦 Chunks in ChromaDB", chunk_count)
    col2.metric("🗄️ Evidence rows in SQLite", evidence_count)

    if chunk_count == 0:
        st.warning("⚠️ No chunks found. Please ingest documents first.")
        return

    st.divider()

    # ── Extract button ─────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Extraction will process all chunks not yet extracted.**")
    with col2:
        if st.button("🚀 Run Extraction", type="primary", use_container_width=True):
            _run_extraction(repo)

    # ── Evidence overview ──────────────────────────────────────────────────
    st.divider()
    _render_evidence_overview(repo)


def _run_extraction(repo: EvidenceRepo) -> None:
    """Run Stage 1 extraction on all ChromaDB chunks using concurrent workers."""
    from pecs.vectorstore.chroma_store import ChromaStore
    from pecs.models.evidence_chunk import EvidenceChunk, SourceType
    from pecs.config import settings

    chroma = ChromaStore()
    extractor = Extractor()

    # Get all chunks from ChromaDB
    total = chroma.count()
    if total == 0:
        st.warning("No chunks to extract.")
        return

    # Get all chunks via a broad query
    all_data = chroma._get_collection().get(include=["documents", "metadatas"])
    chunk_ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    # Reconstruct EvidenceChunks
    chunks = []
    for cid, doc_text, meta in zip(chunk_ids, documents, metadatas):
        source_type_str = meta.get("source_type", "MARKDOWN")
        try:
            source_type = SourceType(source_type_str)
        except ValueError:
            source_type = SourceType.MARKDOWN

        chunks.append(EvidenceChunk(
            chunk_id=cid,
            source_document=meta.get("source_document", ""),
            source_hash=meta.get("source_hash", ""),
            source_type=source_type,
            chunk_index=meta.get("chunk_index", 0),
            source_locator=meta.get("source_locator", ""),
            normalized_text=doc_text,
            char_count=len(doc_text),
        ))

    n_workers = settings.EXTRACTION_WORKERS
    st.info(
        f"Extracting from **{len(chunks)} chunks** using **{n_workers} parallel workers**…"
    )
    progress = st.progress(0, text="Starting extraction…")
    status_text = st.empty()

    total_extracted = 0
    total_stored = 0
    # Lock protects the Streamlit UI widgets which must only be updated from
    # the main thread. on_progress is invoked inside as_completed() which runs
    # on the calling (main) thread, so no lock is strictly required — but we
    # keep one for safety if Streamlit's internals ever change.
    progress_lock = __import__("threading").Lock()

    def on_progress(completed: int, total_chunks: int, source_doc: str) -> None:
        """Called by extract_batch_concurrent after each chunk completes."""
        with progress_lock:
            pct = completed / total_chunks
            progress.progress(pct, text=f"Chunk {completed}/{total_chunks} — {source_doc}")

    # ── Concurrent extraction ─────────────────────────────────────────────────
    all_results = extractor.extract_batch_concurrent(
        chunks,
        max_workers=n_workers,
        on_progress=on_progress,
    )

    # ── SQLite writes — main thread only ──────────────────────────────────────
    # Writing to SQLite from the main thread after extraction is complete is the
    # safest approach. SQLite connections are not thread-safe by default, and
    # WAL mode only helps concurrent *reads*, not concurrent writes.
    status_text.info("💾 Storing extracted evidence…")
    for chunk_results in all_results.values():
        total_extracted += len(chunk_results)
        stored_ids = repo.insert_batch(chunk_results)
        total_stored += len(stored_ids)

    progress.progress(1.0, text="✅ Extraction complete!")
    status_text.empty()
    st.success(
        f"Extracted **{total_extracted}** entities, stored **{total_stored}** new rows "
        f"({total_extracted - total_stored} already existed). "
        f"Used **{n_workers} parallel workers**."
    )
    st.session_state.extraction_complete = True


def _render_evidence_overview(repo: EvidenceRepo) -> None:
    """Show breakdown of evidence by type."""
    st.subheader("📋 Evidence Store Overview")

    requirements = repo.get_all_requirements()
    implementations = repo.get_all_implementations()
    evaluations = repo.get_all_evaluations()

    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Requirements", len(requirements))
    col2.metric("⚙️ Implementations", len(implementations))
    col3.metric("📝 Evaluations", len(evaluations))

    if requirements:
        with st.expander("📋 Requirements"):
            import pandas as pd
            df = pd.DataFrame(requirements)[["entity_id", "text", "source_document"]]
            df.columns = ["Entity ID", "Text", "Source"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    if evaluations:
        with st.expander("📝 Evaluations"):
            import pandas as pd
            df = pd.DataFrame(evaluations)[["entity_id", "text", "source_document"]]
            df.columns = ["Entity ID", "Text", "Source"]
            st.dataframe(df, use_container_width=True, hide_index=True)
