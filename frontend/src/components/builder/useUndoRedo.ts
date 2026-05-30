import { useCallback, useState } from "react";
import type { Edge } from "@xyflow/react";
import type { BuilderNode } from "./types";

export interface Snapshot {
  nodes: BuilderNode[];
  edges: Edge[];
}

const MAX_HISTORY = 50;

/**
 * Generic nodes+edges undo/redo history. The caller records the state *before*
 * a mutation via `takeSnapshot`, and applies the returned snapshot from
 * `undo`/`redo` to its own React Flow state.
 */
export function useUndoRedo() {
  const [past, setPast] = useState<Snapshot[]>([]);
  const [future, setFuture] = useState<Snapshot[]>([]);

  /** Record the current state before a change; clears the redo stack. */
  const takeSnapshot = useCallback((snap: Snapshot) => {
    setPast((p) => [...p, snap].slice(-MAX_HISTORY));
    setFuture([]);
  }, []);

  const undo = useCallback(
    (current: Snapshot): Snapshot | null => {
      if (past.length === 0) return null;
      const previous = past[past.length - 1];
      setPast(past.slice(0, -1));
      setFuture([current, ...future].slice(0, MAX_HISTORY));
      return previous;
    },
    [past, future],
  );

  const redo = useCallback(
    (current: Snapshot): Snapshot | null => {
      if (future.length === 0) return null;
      const next = future[0];
      setFuture(future.slice(1));
      setPast([...past, current].slice(-MAX_HISTORY));
      return next;
    },
    [past, future],
  );

  /** Forget all history (e.g. after importing/clearing a workflow). */
  const reset = useCallback(() => {
    setPast([]);
    setFuture([]);
  }, []);

  return {
    takeSnapshot,
    undo,
    redo,
    reset,
    canUndo: past.length > 0,
    canRedo: future.length > 0,
  };
}
