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
      return { kind: "parallel", label: payload.label };
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
