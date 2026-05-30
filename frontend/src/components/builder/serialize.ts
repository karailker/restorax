import type { Edge } from "@xyflow/react";
import type { DAGConfig, DAGEdge, DAGNode } from "@/types";
import type { BuilderNode } from "./types";

/** Default port name used by most nodes for both input and output. */
const DEFAULT_PORT = "video";

/**
 * Map a single canvas node to its backend DAGNode shape.
 * Field shapes per backend contract:
 *  - restore: { restorer_name, params }
 *  - merge:   { strategy, select_index? }
 *  - parallel: { branches: [] }  (branch authoring is out of scope; emit empty)
 *  - video_input / video_output / pass: structural, no extra fields
 */
function toDagNode(node: BuilderNode): DAGNode {
  const data = node.data;
  const base = { id: node.id, name: data.label };

  switch (data.kind) {
    case "restore":
      return {
        ...base,
        type: "restore",
        restorer_name: data.restorer_name,
        params: data.params,
      };
    case "merge": {
      const out: DAGNode = {
        ...base,
        type: "merge",
        strategy: data.strategy,
      };
      if (data.strategy === "select") out.select_index = data.select_index;
      return out;
    }
    case "parallel":
      return { ...base, type: "parallel", branches: [] };
    case "video_input":
    case "video_output":
    case "pass":
      return { ...base, type: data.kind };
  }
}

function toDagEdge(edge: Edge): DAGEdge {
  return {
    source_node_id: edge.source,
    source_port: edge.sourceHandle ?? DEFAULT_PORT,
    target_node_id: edge.target,
    target_port: edge.targetHandle ?? DEFAULT_PORT,
  };
}

/** Pure serializer: canvas state -> DAGConfig payload. */
export function serializeDag(
  id: string,
  name: string,
  nodes: BuilderNode[],
  edges: Edge[],
): DAGConfig {
  return {
    schema_type: "dag",
    id,
    name,
    nodes: nodes.map(toDagNode),
    edges: edges.map(toDagEdge),
  };
}

/** Build a URL-safe id from a free-text name. */
export function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "dag"
  );
}
