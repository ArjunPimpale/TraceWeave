"""Short, deterministic names for traceability records."""

from __future__ import annotations

import json
import re


def short(text: str, limit: int = 68) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[:limit - 1].rsplit(" ", 1)[0] + "…"


def evidence_label(row: dict, chunk: dict | None = None) -> str:
    kind = row.get("entity_type")
    name = str(row.get("entity_id") or "")
    if kind == "REQUIREMENT":
        token = re.match(r"(?i)^(?:REQ[-_ ]?\d+|R\d+)(?:\b|[-_])", name)
        return f"{token.group(0).rstrip('-_ ')} · {short(row.get('text', ''), 48)}" if token else f"Requirement #{row['id']} · {short(row.get('text', ''), 45)}"
    meta = json.loads(chunk.get("metadata_json") or "{}") if chunk else {}
    if kind == "IMPLEMENTATION" and meta.get("element_name"):
        return short(f"{meta['element_name']} · {row.get('source_document', '')}")
    prefix = "Implementation claim" if kind == "IMPLEMENTATION" and str(row.get("source_document", "")).endswith((".txt", ".eml")) else kind.title()
    return short(f"{prefix} · {row.get('text', '')}")


def chunk_label(chunk: dict) -> str:
    meta = json.loads(chunk.get("metadata_json") or "{}")
    subject = meta.get("element_name") or meta.get("heading_text") or meta.get("commit_sha")
    if subject:
        return short(f"{subject} · {chunk.get('source_document') or 'unknown source'}")
    return short(f"{chunk.get('source_document') or 'Source chunk'} · {chunk.get('source_locator') or chunk.get('chunk_id','')[:10]}")
