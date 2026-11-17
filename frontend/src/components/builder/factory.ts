import type { XYPosition } from "@xyflow/react";
import type { BuilderNode, BuilderNodeData, PaletteDragPayload } from "./types";

let counter = 0;
function nextId(prefix: string): string {
  counter += 1;
  return `${prefix}-${counter}`;
}

function dataFor(payload: PaletteDragPayload): BuilderNodeData {
  switch (payload.type) {
    case "restore":
      return {
        kind: "restore",
        label: payload.label,
        restorer_name: payload.restorer_name ?? "",
        params: {},
      };
    case "merge":
      return { kind: "merge", label: payload.label, strategy: "blend", select_index: 0 };
    case "parallel":
      return { kind: "parallel", label: payload.label, branches: [] };
    case "pass":
      return { kind: "pass", label: payload.label };
    default:
      return { kind: "pass", label: payload.label };
  }
}

/** Build a ReactFlow node from a dropped palette payload. */
export function createNode(payload: PaletteDragPayload, position: XYPosition): BuilderNode {
  return {
    id: nextId(payload.type),
    type: payload.type,
    position,
    data: dataFor(payload),
  };
}

/** Clone a node with a fresh id, offset slightly so it doesn't overlap. */
export function duplicateNode(node: BuilderNode): BuilderNode {
  return {
    id: nextId(node.type ?? "node"),
    type: node.type,
    position: { x: node.position.x + 40, y: node.position.y + 40 },
    data: { ...node.data },
    selected: false,
  };
}
