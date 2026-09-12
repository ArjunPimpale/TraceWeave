"""
Streamlit application entry point for PECS.

Multi-page app organized as:
1. 📤 Ingest — upload documents
2. 🔍 Extract — run Stage 1 extraction
3. 🔗 Correlate — build traceability matrix
4. 📊 Matrix — view the traceability matrix
5. 🔎 Explore — search/query the evidence store
6. ⚙️ Settings — app configuration and status
"""

import streamlit as st

# Must be the first Streamlit call
st.set_page_config(
    page_title="PECS — Project Evidence Correlation System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pecs.logging_config import get_logger
from pecs.ui.components.sidebar import render_sidebar
from pecs.ui.pages import (
    ingest_page,
    extract_page,
    correlate_page,
    matrix_page,
    explore_page,
    graph_page,
    settings_page,
)

logger = get_logger(__name__)


def main() -> None:
    """Main Streamlit application entry point."""
    # Initialize session state
    _init_session_state()

    # Render sidebar navigation
    page = render_sidebar()

    # Route to selected page
    if page == "ingest":
        ingest_page.render()
    elif page == "extract":
        extract_page.render()
    elif page == "correlate":
        correlate_page.render()
    elif page == "matrix":
        matrix_page.render()
    elif page == "graph":
        graph_page.render()
    elif page == "explore":
        explore_page.render()
    elif page == "settings":
        settings_page.render()
    else:
        ingest_page.render()


def _init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults = {
        "page": "ingest",
        "ingestion_results": [],
        "extraction_complete": False,
        "matrix_built": False,
        "last_matrix": None,
        "query_results": [],
        # Graph page state
        "selected_graph_node": None,
        "graph_synced": False,
        "graph_sync_result": None,
        "graph_view_mode": "Requirement focus",
        "graph_show_evaluations": True,
        "graph_show_source_docs": False,
        "graph_show_orphans": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


if __name__ == "__main__":
    main()
