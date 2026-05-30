import type { Node } from "@xyflow/react";

/** DAG node kinds we support on the canvas. Mirrors backend node `type` values. */
export type BuilderNodeType =
  | "video_input"
  | "video_output"
  | "restore"
  | "parallel"
  | "merge"
  | "pass";

/** Per-kind data carried on a ReactFlow node. */
export interface RestoreNodeData {
  kind: "restore";
  label: string;
  restorer_name: string;
  params: Record<string, unknown>;
  [key: string]: unknown;
}

export interface MergeNodeData {
  kind: "merge";
  label: string;
  strategy: "blend" | "select";
  select_index: number;
  [key: string]: unknown;
}

export interface StructuralNodeData {
  kind: "video_input" | "video_output" | "parallel" | "pass";
  label: string;
  [key: string]: unknown;
}

export type BuilderNodeData = RestoreNodeData | MergeNodeData | StructuralNodeData;

export type BuilderNode = Node<BuilderNodeData>;

/** Data passed through the HTML5 drag payload. */
export interface PaletteDragPayload {
  type: BuilderNodeType;
  label: string;
  restorer_name?: string;
}

export const DRAG_MIME = "application/restorax-node";
