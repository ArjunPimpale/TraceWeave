import cytoscape from "cytoscape";
import { Streamlit } from "streamlit-component-lib";
import { validEvent } from "./protocol.js";
import "./style.css";

type CanvasNode = { id: string; title: string; kind: string; status?: string; x: number; y: number };
type CanvasEdge = { id: string; source: string; target: string; label: string; kind: string };
type Payload = { snapshot_key: string; view_key: string; nodes: CanvasNode[];
  edges: CanvasEdge[]; selection?: string };

let cy: cytoscape.Core | undefined;
let lastView: string | undefined;
let lastSelection: string | undefined;

function highlight(selection: string) {
  if (!cy) return;
  cy.elements().removeClass("dimmed");
  if (!selection) return;
  const chosen = cy.getElementById(selection);
  if (chosen.empty()) return;
  cy.elements().addClass("dimmed");
  const related = chosen.isNode() ? chosen.union(chosen.connectedEdges()).union(chosen.neighborhood("node"))
    : chosen.union(chosen.source()).union(chosen.target());
  related.removeClass("dimmed");
}

function assessmentLabel(status?: string) {
  const labels: Record<string, string> = {
    IMPLEMENTED_AND_VALIDATED: "Validated", IMPLEMENTED_WITHOUT_EVALUATION: "Implemented",
    IMPLEMENTED_BUT_NEGATIVELY_EVALUATED: "Negative evaluation",
    PARTIALLY_IMPLEMENTED: "Partial", CLAIMED_BUT_NO_EVIDENCE: "Claim only",
    EVALUATION_WITHOUT_REQUIREMENT: "Evaluation only",
    REQUIREMENT_NOT_IMPLEMENTED: "No implementation"
  };
  return status ? (labels[status] || status) : "Not assessed";
}

function onRender(event: CustomEvent<{ args: { payload: Payload } }>) {
  const payload = event.detail.args.payload;
  if (!payload || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) return;
  const view = `${payload.snapshot_key}:${payload.view_key}`;
  const selection = payload.selection || "";
  if (view === lastView && cy) {
    if (selection !== lastSelection) {
      cy.elements().unselect();
      if (selection) cy.getElementById(selection).select();
      lastSelection = selection;
      highlight(selection);
    }
    return;
  }
  if (cy) cy.destroy();
  const elements = [
    ...payload.nodes.map(n => ({ data: { id:n.id, label:n.kind === "Requirement" ?
        `${n.title}\n${assessmentLabel(n.status)}` : n.title, kind:n.kind, status:n.status || "" },
      position: { x:n.x, y:n.y } })),
    ...payload.edges.map(e => ({ data: { id:e.id, source:e.source, target:e.target,
      label:e.label, kind:e.kind } }))
  ];
  cy = cytoscape({
    container: document.getElementById("canvas"), elements,
    layout: { name:"preset", fit:true, padding:60 },
    userZoomingEnabled:true, userPanningEnabled:true,
    style: [
      { selector:"node", style:{ "background-color":"#64748b", "label":"data(label)",
        "font-size":"11px", "color":"#e2e8f0", "text-wrap":"wrap", "text-max-width":"140px",
        "text-valign":"bottom", "text-margin-y":"8px", "width":"38px", "height":"38px" } },
      { selector:'node[kind="Requirement"]', style:{ "background-color":"#3b82f6", "shape":"round-rectangle", "width":"52px", "height":"52px" } },
      { selector:'node[status="IMPLEMENTED_AND_VALIDATED"]', style:{ "background-color":"#16a34a" } },
      { selector:'node[status="IMPLEMENTED_BUT_NEGATIVELY_EVALUATED"]', style:{ "background-color":"#ea580c" } },
      { selector:'node[status="PARTIALLY_IMPLEMENTED"]', style:{ "background-color":"#ca8a04" } },
      { selector:'node[status="CLAIMED_BUT_NO_EVIDENCE"]', style:{ "background-color":"#9333ea" } },
      { selector:'node[status="REQUIREMENT_NOT_IMPLEMENTED"]', style:{ "background-color":"#dc2626" } },
      { selector:'node[kind="Implementation"]', style:{ "background-color":"#22c55e", "shape":"rectangle" } },
      { selector:'node[kind="Evaluation"]', style:{ "background-color":"#f97316", "shape":"diamond" } },
      { selector:'node[kind="Chunk"]', style:{ "background-color":"#94a3b8", "shape":"ellipse", "width":"30px", "height":"30px" } },
      { selector:"edge", style:{ "width":2, "line-color":"#94a3b8", "target-arrow-color":"#94a3b8",
        "target-arrow-shape":"triangle", "curve-style":"bezier", "label":"data(label)",
        "font-size":"9px", "color":"#cbd5e1", "text-background-color":"#0f172a",
        "text-background-opacity":1 } },
      { selector:'edge[kind="RETRIEVED_CANDIDATE"]', style:{ "line-style":"dashed", "opacity":0.6 } },
      { selector:'edge[kind="ASSESSMENT_CONTEXT"]', style:{ "line-style":"dotted" } },
      { selector:'edge[kind="EXTRACTED_FROM"]', style:{ "line-style":"dashed", "opacity":0.5 } },
      { selector: ":selected", style:{ "border-color":"#f8fafc", "border-width":4,
        "line-color":"#f8fafc", "target-arrow-color":"#f8fafc" } },
      { selector: ".dimmed", style: { "opacity": 0.18 } }
    ]
  });
  cy.on("tap", "node, edge", e => {
    const message = { event_id: crypto.randomUUID(), snapshot_key:payload.snapshot_key,
      view_key:payload.view_key, action:e.target.isNode() ? "select_node" : "select_edge",
      target_id:e.target.id() };
    if (validEvent(message, payload)) Streamlit.setComponentValue(message);
    highlight(e.target.id());
  });
  cy.on("tap", e => {
    if (e.target !== cy) return;
    cy?.elements().unselect();
    highlight("");
    Streamlit.setComponentValue({ event_id: crypto.randomUUID(), snapshot_key: payload.snapshot_key,
      view_key: payload.view_key, action: "clear_selection", target_id: null });
  });
  lastView = view;
  lastSelection = selection;
  if (selection) cy.getElementById(selection).select();
  highlight(selection);
  Streamlit.setFrameHeight(600);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(600);
