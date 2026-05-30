import type { Edge } from "@xyflow/react";
import type { BuilderNode } from "./types";

/**
 * A RestoraX workflow file — the full canvas state (node positions + data +
 * edges), distinct from the backend DAGConfig (which drops positions). This is
 * the ComfyUI-style "save/load the graph you're editing" artifact.
 */
export const WORKFLOW_SCHEMA = "restorax-workflow";
export const WORKFLOW_VERSION = 1;

export interface WorkflowFile {
  schema: typeof WORKFLOW_SCHEMA;
  version: number;
  name: string;
  nodes: BuilderNode[];
  edges: Edge[];
}

export function serializeWorkflow(
  name: string,
  nodes: BuilderNode[],
  edges: Edge[],
): WorkflowFile {
  return { schema: WORKFLOW_SCHEMA, version: WORKFLOW_VERSION, name, nodes, edges };
}

/** Trigger a browser download of the workflow as a .json file. */
export function downloadWorkflow(
  name: string,
  nodes: BuilderNode[],
  edges: Edge[],
): void {
  const data = serializeWorkflow(name, nodes, edges);
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${slugifyFilename(name)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Parse + validate a workflow file. Throws a descriptive Error if the payload
 * is not a recognizable RestoraX workflow.
 */
export function parseWorkflow(text: string): WorkflowFile {
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    throw new Error("Not valid JSON");
  }
  if (typeof obj !== "object" || obj === null) {
    throw new Error("Workflow must be a JSON object");
  }
  const w = obj as Partial<WorkflowFile>;
  if (w.schema !== WORKFLOW_SCHEMA) {
    throw new Error("Not a RestoraX workflow file");
  }
  if (!Array.isArray(w.nodes) || !Array.isArray(w.edges)) {
    throw new Error("Workflow is missing nodes/edges");
  }
  return {
    schema: WORKFLOW_SCHEMA,
    version: typeof w.version === "number" ? w.version : WORKFLOW_VERSION,
    name: typeof w.name === "string" ? w.name : "Imported workflow",
    nodes: w.nodes as BuilderNode[],
    edges: w.edges as Edge[],
  };
}

function slugifyFilename(name: string): string {
  return (
    name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "workflow"
  );
}
