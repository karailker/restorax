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

/** One step inside a parallel branch: restorer slug + its params dict. */
export type BranchStep = [string, Record<string, unknown>];

/** A named branch of restorer steps — mirrors the backend BranchConfig. */
export interface BranchConfig {
  name: string;
  restorer_steps: BranchStep[];
}

export interface ParallelNodeData {
  kind: "parallel";
  label: string;
  branches: BranchConfig[];
  [key: string]: unknown;
}

export interface StructuralNodeData {
  kind: "video_input" | "video_output" | "pass";
  label: string;
  [key: string]: unknown;
}

export type BuilderNodeData =
  | RestoreNodeData
  | MergeNodeData
  | ParallelNodeData
  | StructuralNodeData;

export type BuilderNode = Node<BuilderNodeData>;

/** Data passed through the HTML5 drag payload. */
export interface PaletteDragPayload {
  type: BuilderNodeType;
  label: string;
  restorer_name?: string;
}

export const DRAG_MIME = "application/restorax-node";
