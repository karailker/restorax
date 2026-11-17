import type { Edge } from "@xyflow/react";
import type { DAGConfig, DAGEdge, DAGNode } from "@/types";
import type { BuilderNode, BuilderNodeType } from "./types";
import { NODE_PORTS } from "./ports";

/**
 * Map a single canvas node to its backend DAGNode shape.
 * Field shapes per backend contract:
 *  - restore: { restorer_name, params_dict }
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
        params_dict: data.params,
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
      return { ...base, type: "parallel", branches: data.branches ?? [] };
    case "video_input":
    case "video_output":
    case "pass":
      return { ...base, type: data.kind };
  }
}

/** First declared port name for a node type — the fallback when an edge lacks an explicit handle id. */
function firstPort(nodeType: BuilderNodeType | undefined, side: "inputs" | "outputs"): string {
  return (nodeType && NODE_PORTS[nodeType][side][0]?.name) ?? "";
}

function toDagEdge(edge: Edge, typeOf: (id: string) => BuilderNodeType | undefined): DAGEdge {
  return {
    source_node_id: edge.source,
    source_port: edge.sourceHandle ?? firstPort(typeOf(edge.source), "outputs"),
    target_node_id: edge.target,
    target_port: edge.targetHandle ?? firstPort(typeOf(edge.target), "inputs"),
  };
}

/** Pure serializer: canvas state -> DAGConfig payload. */
export function serializeDag(
  id: string,
  name: string,
  nodes: BuilderNode[],
  edges: Edge[],
): DAGConfig {
  const typeById = new Map(nodes.map((n) => [n.id, n.type as BuilderNodeType | undefined]));
  const typeOf = (nodeId: string) => typeById.get(nodeId);
  return {
    schema_type: "dag",
    id,
    name,
    nodes: nodes.map(toDagNode),
    edges: edges.map((e) => toDagEdge(e, typeOf)),
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
