import test from "node:test";
import assert from "node:assert/strict";
import { validEvent } from "../src/protocol.js";
import { openLayoutOptions } from "../src/layout.js";

const payload = { snapshot_key: "dataset:run", view_key: "focused", nodes: [{ id: "ev:1" }],
  edges: [{ id: "edge:1" }] };

test("accepts current node and edge selections", () => {
  assert.equal(validEvent({ event_id: "click-1", snapshot_key: "dataset:run",
    view_key: "focused", action: "select_node", target_id: "ev:1" }, payload), true);
  assert.equal(validEvent({ event_id: "click-2", snapshot_key: "dataset:run",
    view_key: "focused", action: "select_edge", target_id: "edge:1" }, payload), true);
  assert.equal(validEvent({ event_id: "click-3", snapshot_key: "dataset:run",
    view_key: "focused", action: "clear_selection", target_id: null }, payload), true);
});

test("rejects stale, unknown, and unsupported events", () => {
  const base = { event_id: "click-1", snapshot_key: "dataset:run",
    view_key: "focused", action: "select_node", target_id: "ev:1" };
  assert.equal(validEvent({ ...base, snapshot_key: "old" }, payload), false);
  assert.equal(validEvent({ ...base, view_key: "other" }, payload), false);
  assert.equal(validEvent({ ...base, target_id: "ev:other" }, payload), false);
  assert.equal(validEvent({ ...base, action: "expand_group" }, payload), false);
  assert.equal(validEvent({ ...base, event_id: 3 }, payload), false);
});

test("the opening layout randomizes nodes and pushes them apart", () => {
  const layout = openLayoutOptions();
  assert.equal(layout.name, "cose");
  assert.equal(layout.randomize, true);
  assert.equal(layout.fit, true);
  assert.ok(layout.nodeRepulsion >= 10000);
  assert.ok(layout.idealEdgeLength >= 100);
});
