"""Text-safe, full traceability inspector for a selected graph record."""

from __future__ import annotations

import streamlit as st


def render_detail(snapshot: dict, projection: dict, selected_id: str | None) -> None:
    if not selected_id:
        return
    node = projection["nodes"].get(selected_id)
    edge = projection["edges"].get(selected_id)
    if not node and not edge:
        return
    st.divider()
    st.subheader("Trace details")
    if edge:
        st.write(f"{edge['label']}: {projection['nodes'][edge['source']]['title']} → "
                 f"{projection['nodes'][edge['target']]['title']}")
        st.caption(f"Basis: {edge['details'].get('basis', edge['kind'])}")
        if edge.get("reference_ids"):
            st.caption(f"Stored citation records: {', '.join(edge['reference_ids'])}")
        if edge["assessment_ids"]:
            for correlation_id in edge["assessment_ids"]:
                _assessment(snapshot["correlations"].get(correlation_id))
        else:
            st.caption("Extracted reference; no assessment used this link in the selected run.")
        return
    st.write(node["title"])
    st.caption(f"{node['kind']} · {node['subtitle']}")
    details = node["details"]
    if node["kind"] == "Requirement":
        st.write(details.get("text") or "")
        st.write(f"State: {details.get('workflow_state')}; status: {node.get('status') or 'Not assessed'}")
        if node.get("status"):
            st.caption(f"Assessment score: {details.get('confidence', 0):.0%}")
        if details.get("conflicting"):
            st.warning("Assessments disagree. Inspect every decision below.")
        for diagnostic in details.get("diagnostics", []):
            st.warning(diagnostic)
        for correlation_id in details.get("assessment_ids", []):
            _assessment(snapshot["correlations"].get(correlation_id))
    else:
        text = details.get("text")
        if text:
            if node["kind"] == "Chunk" and str(details.get("source_document", "")).endswith(".py"):
                st.code(text, language="python")
            else:
                st.text(text)
        elif node["kind"] == "Chunk":
            st.warning("Source content is unavailable for this citation.")
        if details.get("source_document"):
            st.caption(f"Source: {details['source_document']} · {details.get('source_locator') or 'location unknown'}")
        if details.get("chunk_id"):
            st.caption(f"Chunk ID: {details['chunk_id']}")
        if details.get("entity_id"):
            st.caption(f"Extracted ID: {details['entity_id']}")
        if details.get("linked_requirement"):
            st.caption(f"Extracted requirement reference: {details['linked_requirement']}")
        if details.get("unresolved_reference"):
            st.warning(f"Requirement reference could not be resolved uniquely: {details['unresolved_reference']}")
        if details.get("unlinked_reason"):
            st.caption(details["unlinked_reason"])
        if node["kind"] in ("Implementation", "Evaluation"):
            sources = [projection["nodes"][edge["target"]] for edge in projection["edges"].values()
                       if edge["kind"] == "EXTRACTED_FROM" and edge["source"] == selected_id
                       and edge["target"] in projection["nodes"]]
            for source in sources:
                with st.expander(f"Source context · {source['title']}"):
                    source_text = source["details"].get("text")
                    if source_text:
                        if str(source["details"].get("source_document", "")).endswith(".py"):
                            st.code(source_text, language="python")
                        else:
                            st.text(source_text)
                    else:
                        st.warning("Source content is unavailable for this citation.")
        for correlation_id in details.get("unlinked_assessment_ids", []):
            _assessment(snapshot["correlations"].get(correlation_id))


def _assessment(correlation: dict | None) -> None:
    if not correlation:
        st.warning("Assessment record unavailable")
        return
    label = f"{correlation['status']} · {correlation['confidence']:.0%} assessment score · {correlation['correlation_id'][:8]}"
    with st.expander(label):
        st.write(correlation.get("reasoning") or "No reasoning saved")
        st.caption(f"Method: {correlation['resolution_method']} · Rule: {correlation.get('rule_name') or 'none'}")
        for ref in correlation.get("references", []):
            kind = ref.get("target_kind")
            target = ref.get("evidence_row_id") if kind == "evidence" else ref.get("chunk_id") or ref.get("unresolved_value")
            st.text(f"{ref.get('role', 'citation')}: {kind} {target}")
            if ref.get("visible_text"):
                st.caption("Text supplied to classifier:")
                st.text(ref["visible_text"])
