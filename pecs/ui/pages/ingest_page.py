"""
Ingest Documents page.

Allows uploading PDF, DOCX, email, WhatsApp, Git log, Markdown, and Python files.
Runs the ingestion pipeline for each file and shows a summary.
"""

from __future__ import annotations

import streamlit as st

from pecs.embeddings.embedder import Embedder
from pecs.ingestion.ingestor import Ingestor
from pecs.logging_config import get_logger
from pecs.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "doc",
    "md", "markdown",
    "py", "pyw",
    "txt", "log", "eml",
]


def render() -> None:
    st.title("📤 Ingest Documents")
    st.markdown(
        "Upload project documents to add them to the evidence store. "
        "Each file is parsed, chunked, embedded, and indexed automatically."
    )

    # ── Upload widget ──────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help=(
            "Supported: PDF, DOCX, Markdown, Python (.py), "
            "Email (.eml, .txt), WhatsApp export (.txt), Git log (.log, .txt)"
        ),
    )

    # ── Optional metadata ──────────────────────────────────────────────────
    with st.expander("📋 Optional metadata (applied to all uploaded files)"):
        project_name = st.text_input("Project name", placeholder="e.g., AnderBahar")
        description = st.text_area("Description", placeholder="What are these documents about?")
        user_metadata: dict = {}
        if project_name:
            user_metadata["project"] = project_name
        if description:
            user_metadata["description"] = description

    # ── Ingest button ──────────────────────────────────────────────────────
    if uploaded_files and st.button("🚀 Ingest All Files", type="primary", use_container_width=True):
        _ingest_files(uploaded_files, user_metadata)

    # ── Ingestion history ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Ingestion History")
    _render_ingestion_log()


def _ingest_files(uploaded_files, user_metadata: dict) -> None:
    """Run ingestion pipeline for all uploaded files."""
    ingestor = Ingestor()
    embedder = Embedder()
    chroma = ChromaStore()

    progress_bar = st.progress(0, text="Starting ingestion…")
    results_container = st.empty()

    all_results = []
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(
            (i) / total,
            text=f"Processing {uploaded_file.name}… ({i + 1}/{total})",
        )

        file_bytes = uploaded_file.read()
        result = ingestor.ingest_file(
            file_bytes=file_bytes,
            filename=uploaded_file.name,
            user_metadata=user_metadata,
        )
        all_results.append(result)

        if result.success and not result.skipped and result.evidence_chunks:
            # Embed and store in ChromaDB
            with st.spinner(f"Embedding {uploaded_file.name}…"):
                chunk_embeddings = embedder.embed_chunks(result.evidence_chunks)
                chunk_ids_to_embed = [c for c in result.evidence_chunks
                                       if c.chunk_id in {cid for cid, _ in chunk_embeddings}]
                vectors = [v for _, v in chunk_embeddings]
                chroma.upsert_chunks(result.evidence_chunks, vectors)

    progress_bar.progress(1.0, text="✅ Ingestion complete!")

    # Show results summary
    success_count = sum(1 for r in all_results if r.success and not r.skipped)
    skipped_count = sum(1 for r in all_results if r.skipped)
    failed_count = sum(1 for r in all_results if not r.success)
    total_chunks = sum(r.chunk_count for r in all_results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Ingested", success_count)
    col2.metric("⏭️ Skipped (duplicate)", skipped_count)
    col3.metric("❌ Failed", failed_count)
    col4.metric("📦 Total chunks", total_chunks)

    # Per-file details
    for result in all_results:
        if result.skipped:
            st.warning(f"⏭️ **{result.filename}** — already ingested (duplicate)")
        elif not result.success:
            st.error(f"❌ **{result.filename}** — {result.error}")
        else:
            st.success(
                f"✅ **{result.filename}** — {result.chunk_count} chunks "
                f"({result.source_type})"
            )
            if result.warnings:
                for w in result.warnings:
                    st.caption(f"  ⚠️ {w}")

    # Store results in session state
    st.session_state.ingestion_results.extend(all_results)


def _render_ingestion_log() -> None:
    """Display the ingestion log from SQLite."""
    try:
        from pecs.store.evidence_repo import EvidenceRepo
        repo = EvidenceRepo()
        log = repo.get_ingestion_log()
        if not log:
            st.info("No files ingested yet. Upload files above to get started.")
            return

        import pandas as pd
        df = pd.DataFrame(log)
        df = df[["filename", "source_type", "chunk_count", "file_size", "status", "ingested_at"]]
        df["file_size"] = df["file_size"].apply(lambda x: f"{x / 1024:.1f} KB")
        df.columns = ["File", "Type", "Chunks", "Size", "Status", "Ingested At"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not load ingestion log: {exc}")
