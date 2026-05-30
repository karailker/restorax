import { useCallback, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Connection,
  type Edge,
} from "@xyflow/react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { createDag } from "@/lib/api";
import { Palette } from "@/components/builder/Palette";
import { ConfigPanel } from "@/components/builder/ConfigPanel";
import { nodeTypes } from "@/components/builder/nodes";
import { createNode } from "@/components/builder/factory";
import { serializeDag, slugify } from "@/components/builder/serialize";
import { DRAG_MIME, type BuilderNode, type BuilderNodeData, type PaletteDragPayload } from "@/components/builder/types";

const initialNodes: BuilderNode[] = [
  {
    id: "video_input",
    type: "video_input",
    position: { x: 40, y: 160 },
    data: { kind: "video_input", label: "Video Input" },
  },
  {
    id: "video_output",
    type: "video_output",
    position: { x: 560, y: 160 },
    data: { kind: "video_output", label: "Video Output" },
  },
];

type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "ok"; id: string }
  | { kind: "error"; message: string };

function Builder() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const [nodes, setNodes, onNodesChange] = useNodesState<BuilderNode>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [name, setName] = useState("Untitled DAG");
  const [save, setSave] = useState<SaveState>({ kind: "idle" });

  const onConnect = useCallback(
    (c: Connection) => setEdges((eds) => addEdge(c, eds)),
    [setEdges],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData(DRAG_MIME);
      if (!raw) return;
      const payload = JSON.parse(raw) as PaletteDragPayload;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setNodes((nds) => nds.concat(createNode(payload, position)));
    },
    [screenToFlowPosition, setNodes],
  );

  const patchNode = useCallback(
    (id: string, patch: Partial<BuilderNodeData>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, ...patch } as BuilderNodeData } : n,
        ),
      );
    },
    [setNodes],
  );

  const selectedNode = nodes.find((n) => n.id === selectedId) ?? null;

  const onSave = useCallback(async () => {
    const id = slugify(name);
    const config = serializeDag(id, name, nodes, edges);
    setSave({ kind: "saving" });
    try {
      const res = await createDag({ id, name, config });
      setSave({ kind: "ok", id: res.id });
    } catch (err) {
      setSave({ kind: "error", message: err instanceof Error ? err.message : "Save failed" });
    }
  }, [name, nodes, edges]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-[var(--color-border)] px-6 py-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Pipeline Builder</h1>
          <p className="text-xs text-[var(--color-muted-foreground)]">
            Drag restorers onto the canvas, wire up branches, and save a DAG.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {save.kind === "ok" && (
            <Badge variant="success">saved: {save.id}</Badge>
          )}
          {save.kind === "error" && (
            <Badge variant="destructive" title={save.message}>
              {save.message.slice(0, 40)}
            </Badge>
          )}
          <input
            className="h-9 w-56 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="DAG name"
          />
          <Button onClick={onSave} disabled={save.kind === "saving" || !name.trim()}>
            {save.kind === "saving" ? "Saving…" : "Save DAG"}
          </Button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <Palette />
        <div ref={wrapperRef} className="min-w-0 flex-1" onDragOver={onDragOver} onDrop={onDrop}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onSelectionChange={({ nodes: sel }) => setSelectedId(sel[0]?.id ?? null)}
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
            colorMode="dark"
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>
        </div>
        <ConfigPanel node={selectedNode} onChange={patchNode} />
      </div>
    </div>
  );
}

export default function PipelineBuilder() {
  return (
    <ReactFlowProvider>
      <Builder />
    </ReactFlowProvider>
  );
}
