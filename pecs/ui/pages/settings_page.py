"""
Settings & Status page.

Shows:
- System health (Ollama, ChromaDB, SQLite)
- Current configuration values
- Database reset controls (with confirmation)
"""

from __future__ import annotations

import streamlit as st

from pecs.config import settings
from pecs.logging_config import get_logger

logger = get_logger(__name__)


def render() -> None:
    st.title("⚙️ Settings & Status")

    # ── System health ──────────────────────────────────────────────────────
    st.subheader("🔍 System Health")
    _render_health()

    st.divider()

    # ── Configuration ──────────────────────────────────────────────────────
    st.subheader("⚙️ Configuration")
    _render_config()

    st.divider()

    # ── Data management ────────────────────────────────────────────────────
    st.subheader("🗄️ Data Management")
    _render_data_management()


def _render_health() -> None:
    """Show Ollama, embedding model, LLM, and database status."""
    from pecs.embeddings.embedder import Embedder
    from pecs.vectorstore.chroma_store import ChromaStore
    from pecs.store.evidence_repo import EvidenceRepo

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ollama & Models**")
        try:
            embedder = Embedder()
            health = embedder.check_health()

            if health["ollama_running"]:
                st.success("✅ Ollama server: Running")
            else:
                st.error(f"❌ Ollama server: Offline")
                st.code(f"Start with: ollama serve")

            if health["model_available"]:
                st.success(f"✅ Embedding model: {settings.EMBEDDING_MODEL}")
            else:
                st.warning(f"⚠️ Embedding model not found")
                st.code(f"ollama pull {settings.EMBEDDING_MODEL}")

            if health["ollama_running"]:
                # Check LLM model
                import ollama
                client = ollama.Client(host=settings.OLLAMA_BASE_URL)
                models = client.list()
                model_names = [m.model for m in (models.models or [])]
                llm_available = any(settings.LLM_MODEL in name for name in model_names)
                if llm_available:
                    st.success(f"✅ LLM model: {settings.LLM_MODEL}")
                else:
                    st.warning(f"⚠️ LLM model not found: {settings.LLM_MODEL}")
                    st.code(f"ollama pull {settings.LLM_MODEL}")

        except Exception as exc:
            st.error(f"❌ Ollama check failed: {exc}")

    with col2:
        st.markdown("**Storage**")
        try:
            chroma = ChromaStore()
            count = chroma.count()
            st.success(f"✅ ChromaDB: {count:,} vectors")
        except Exception as exc:
            st.error(f"❌ ChromaDB: {exc}")

        try:
            repo = EvidenceRepo()
            log = repo.get_ingestion_log()
            ev = repo.get_all()
            st.success(f"✅ SQLite: {len(ev):,} evidence rows, {len(log)} files ingested")
            st.caption(f"Database: {settings.SQLITE_DB_PATH}")
        except Exception as exc:
            st.error(f"❌ SQLite: {exc}")


def _render_config() -> None:
    """Display current configuration values."""
    config_items = {
        "Ollama URL": settings.OLLAMA_BASE_URL,
        "LLM Model": settings.LLM_MODEL,
        "Embedding Model": settings.EMBEDDING_MODEL,
        "LLM Temperature": settings.LLM_TEMPERATURE,
        "Max Chunk Size": f"{settings.MAX_CHUNK_CHARS} chars",
        "Overlap": f"{settings.OVERLAP_CHARS} chars",
        "Retrieval Top-K": settings.RETRIEVAL_TOP_K,
        "Vector Weight": settings.VECTOR_WEIGHT,
        "BM25 Weight": settings.BM25_WEIGHT,
        "SQLite DB": settings.SQLITE_DB_PATH,
        "ChromaDB Path": settings.CHROMADB_PATH,
        "Log File": settings.LOG_FILE,
        "Log Level": settings.LOG_LEVEL,
        "Max File Size": f"{settings.MAX_FILE_SIZE_MB} MB",
    }

    import pandas as pd
    df = pd.DataFrame(
        [{"Setting": k, "Value": str(v)} for k, v in config_items.items()]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Modify these values in your `.env` file and restart the app.")


def _render_data_management() -> None:
    """Database reset controls with confirmation."""
    st.warning("⚠️ These operations are irreversible. Use with caution.")

    with st.expander("🗑️ Reset Database"):
        st.markdown("**What will be deleted:**")
        st.markdown("- All evidence rows in SQLite (requirements, implementations, evaluations)")
        st.markdown("- All correlations in SQLite")
        st.markdown("- All ingestion log entries")
        st.markdown("- All vectors in ChromaDB")

        confirm = st.checkbox("I understand this will delete all data permanently")
        if confirm and st.button("🗑️ Reset All Data", type="secondary"):
            try:
                from pecs.store.database import get_database
                from pecs.vectorstore.chroma_store import ChromaStore

                db = get_database()
                conn = db.connect()
                with conn:
                    conn.execute("DELETE FROM evidence")
                    conn.execute("DELETE FROM correlation")
                    conn.execute("DELETE FROM ingestion_log")

                chroma = ChromaStore()
                chroma.reset_collection()

                st.session_state.matrix_built = False
                st.session_state.last_matrix = None
                st.session_state.ingestion_results = []

                st.success("✅ All data deleted successfully.")
            except Exception as exc:
                st.error(f"❌ Reset failed: {exc}")
