import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { BuilderNodeType } from "../types";
import { NODE_PORTS, PORT_COLOR, type PortDef } from "../ports";

interface NodeShellProps {
  selected?: NodeProps["selected"];
  /** Node type — drives which typed sockets are rendered. */
  nodeType: BuilderNodeType;
  title: string;
  subtitle?: string;
  /** Tailwind accent class for the left bar, e.g. "bg-primary". */
  accent: string;
}

const HANDLE_CLASS = "!h-3 !w-3 !rounded-full !border-2 !border-card";

function PortRow({ port, side }: { port: PortDef; side: "input" | "output" }) {
  const isInput = side === "input";
  // Handle anchors to the node's left/right border (left:0 / right:0 of this
  // row); padding keeps the label clear of the socket.
  return (
    <div
      className={cn(
        "relative flex items-center",
        isInput ? "justify-start pl-3" : "justify-end pr-3",
      )}
    >
      <Handle
        id={port.name}
        type={isInput ? "target" : "source"}
        position={isInput ? Position.Left : Position.Right}
        className={cn(HANDLE_CLASS, PORT_COLOR[port.type])}
      />
      <span className="text-[10px] leading-none text-muted-foreground">
        {port.name}
      </span>
    </div>
  );
}

/** Shared visual chrome for builder nodes — typed sockets stay consistent. */
export function NodeShell({
  selected,
  nodeType,
  title,
  subtitle,
  accent,
}: NodeShellProps) {
  const { inputs, outputs } = NODE_PORTS[nodeType];
  return (
    <div
      className={cn(
        "relative min-w-[180px] rounded-lg border bg-card shadow-sm transition-colors",
        selected ? "border-ring ring-1 ring-ring" : "border-border",
      )}
    >
      <div className={cn("absolute inset-y-0 left-0 w-1 rounded-l-lg", accent)} />
      <div className="border-b border-border px-3 py-2 pl-3.5">
        <p className="text-sm font-medium leading-tight text-foreground">{title}</p>
        {subtitle && (
          <p className="truncate text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
      <div className="flex justify-between gap-6 py-2">
        <div className="flex flex-col gap-2">
          {inputs.map((p) => (
            <PortRow key={p.name} port={p} side="input" />
          ))}
        </div>
        <div className="flex flex-col gap-2">
          {outputs.map((p) => (
            <PortRow key={p.name} port={p} side="output" />
          ))}
        </div>
      </div>
    </div>
  );
}
