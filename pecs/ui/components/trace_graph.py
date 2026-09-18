"""Bidirectional, locally bundled graph canvas."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_BUILD = Path(__file__).resolve().parent / "trace_graph_frontend" / "dist"


def render_graph(payload: dict, key: str) -> dict | None:
    if not (_BUILD / "index.html").exists():
        raise RuntimeError("Graph frontend is not built; run npm ci and npm run build in trace_graph_frontend")
    component = components.declare_component("pecs_trace_graph", path=str(_BUILD))
    return component(payload=payload, key=key, default=None)
