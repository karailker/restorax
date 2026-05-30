import { useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Node,
  type Edge,
} from "@xyflow/react";

const initialNodes: Node[] = [
  { id: "input", position: { x: 0, y: 120 }, data: { label: "Video Input" }, type: "input" },
  { id: "output", position: { x: 520, y: 120 }, data: { label: "Video Output" }, type: "output" },
];
const initialEdges: Edge[] = [];

/**
 * Pipeline Builder — ReactFlow canvas.
 * FOUNDATION STUB: working canvas with input/output nodes. The restorer palette,
 * parallel swim-lanes, node config panel, and save-to-/pipelines/dag are layered
 * on by the Pipeline Builder view work.
 */
export default function PipelineBuilder() {
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const onConnect = useCallback((c: Connection) => setEdges((eds) => addEdge(c, eds)), [setEdges]);

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-[var(--color-border)] px-8 py-4">
        <h1 className="text-2xl font-semibold tracking-tight">Pipeline Builder</h1>
        <p className="text-sm text-[var(--color-muted-foreground)]">
          Drag restorers onto the canvas, wire up branches, and save a DAG.
        </p>
      </header>
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
          colorMode="dark"
        >
          <Background />
          <Controls />
          <MiniMap pannable zoomable />
        </ReactFlow>
      </div>
    </div>
  );
}
