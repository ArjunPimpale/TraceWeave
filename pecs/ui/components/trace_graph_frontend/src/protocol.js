export function validEvent(event, payload) {
  return Boolean(event && typeof event.event_id === "string" &&
    event.snapshot_key === payload.snapshot_key && event.view_key === payload.view_key &&
    (event.action === "clear_selection" ||
      ((event.action === "select_node" || event.action === "select_edge") &&
       [...payload.nodes, ...payload.edges].some(item => item.id === event.target_id))));
}
