"""
Sidebar navigation component.

Renders the PECS sidebar with:
- App branding
- Navigation menu
- System status indicators (Ollama, DB)
"""

from __future__ import annotations

import streamlit as st


def render_sidebar() -> str:
    """
    Render the sidebar and return the selected page key.

    Returns:
        Page key string (e.g., "ingest", "matrix", etc.).
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="margin: 0; font-size: 1.5rem;">🧬 PECS</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #888;">
                    Project Evidence Correlation System
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Navigation ─────────────────────────────────────────────────────
        st.markdown("**Navigation**")

        pages = [
            ("📤 Ingest Documents", "ingest"),
            ("🔬 Extract Evidence", "extract"),
            ("🔗 Correlate", "correlate"),
            ("📊 Traceability Matrix", "matrix"),
            ("🔎 Explore Evidence", "explore"),
            ("⚙️ Settings & Status", "settings"),
        ]

        if "page" not in st.session_state:
            st.session_state.page = "ingest"

        for label, key in pages:
            if st.sidebar.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if st.session_state.page == key else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        st.divider()

        # ── System status ──────────────────────────────────────────────────
        st.markdown("**System Status**")
        _render_status()

        st.divider()
        st.caption("Privacy-preserving local RAG")

    return st.session_state.page


def _render_status() -> None:
    """Render compact status indicators for Ollama and ChromaDB."""
    from pecs.embeddings.embedder import Embedder

    try:
        embedder = Embedder()
        health = embedder.check_health()
        if health["ollama_running"] and health["model_available"]:
            st.success("✅ Ollama: Running")
        elif health["ollama_running"]:
            st.warning("⚠️ Ollama: No model")
        else:
            st.error("❌ Ollama: Offline")
    except Exception:
        st.error("❌ Ollama: Error")

    try:
        from pecs.vectorstore.chroma_store import ChromaStore
        store = ChromaStore()
        count = store.count()
        st.info(f"📦 Vectors: {count:,}")
    except Exception:
        st.warning("📦 Vectors: N/A")

    try:
        from pecs.store.evidence_repo import EvidenceRepo
        repo = EvidenceRepo()
        ev_count = len(repo.get_all())
        st.info(f"🗄️ Evidence rows: {ev_count:,}")
    except Exception:
        st.warning("🗄️ Evidence: N/A")
