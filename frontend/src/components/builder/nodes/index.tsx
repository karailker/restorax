import { type NodeProps } from "@xyflow/react";
import type { BuilderNodeData, MergeNodeData, RestoreNodeData } from "../types";
import { NodeShell } from "./NodeShell";

type Props = NodeProps & { data: BuilderNodeData };

function RestoreNode({ data, selected }: Props) {
  const d = data as RestoreNodeData;
  return (
    <NodeShell
      selected={selected}
      title={d.label}
      subtitle={d.restorer_name || "— pick a restorer —"}
      accent="bg-[var(--color-primary)]"
    />
  );
}

function MergeNode({ data, selected }: Props) {
  const d = data as MergeNodeData;
  return (
    <NodeShell
      selected={selected}
      title={d.label}
      subtitle={
        d.strategy === "select" ? `select #${d.select_index}` : "blend"
      }
      accent="bg-[var(--color-warning)]"
    />
  );
}

function ParallelNode({ data, selected }: Props) {
  return (
    <NodeShell
      selected={selected}
      title={data.label}
      subtitle="parallel branches"
      accent="bg-[var(--color-success)]"
    />
  );
}

function PassNode({ data, selected }: Props) {
  return (
    <NodeShell
      selected={selected}
      title={data.label}
      accent="bg-[var(--color-muted-foreground)]"
    />
  );
}

function VideoInputNode({ data, selected }: Props) {
  return (
    <NodeShell
      selected={selected}
      title={data.label}
      accent="bg-[var(--color-secondary)]"
      hasInput={false}
    />
  );
}

function VideoOutputNode({ data, selected }: Props) {
  return (
    <NodeShell
      selected={selected}
      title={data.label}
      accent="bg-[var(--color-secondary)]"
      hasOutput={false}
    />
  );
}

/** Registry passed to ReactFlow `nodeTypes`. Keys mirror DAG node `type`. */
export const nodeTypes = {
  restore: RestoreNode,
  merge: MergeNode,
  parallel: ParallelNode,
  pass: PassNode,
  video_input: VideoInputNode,
  video_output: VideoOutputNode,
};
