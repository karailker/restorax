import { useEffect, useRef, useState } from "react";
import type { ProgressEvent } from "@/types";
import { API_BASE } from "@/lib/api";

interface JobProgressState {
  /** Latest overall progress 0..1. */
  progress: number;
  status?: string;
  /** Most recent event per branch_index (DAG jobs). */
  branches: Record<number, ProgressEvent>;
  connected: boolean;
  lastEvent: ProgressEvent | null;
}

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

/**
 * Subscribes to /ws/jobs/{id}/progress and tracks overall + per-branch progress.
 * Reconnects on unexpected close until the job reaches a terminal status.
 */
export function useJobProgress(jobId: string | undefined): JobProgressState {
  const [state, setState] = useState<JobProgressState>({
    progress: 0,
    branches: {},
    connected: false,
    lastEvent: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    if (!jobId) return;
    doneRef.current = false;
    let retry: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const wsBase = API_BASE.replace(/^http/, "ws");
      const ws = new WebSocket(`${wsBase}/ws/jobs/${jobId}/progress`);
      wsRef.current = ws;

      ws.onopen = () => setState((s) => ({ ...s, connected: true }));
      ws.onmessage = (e) => {
        const evt = JSON.parse(e.data) as ProgressEvent;
        setState((s) => {
          const branches = { ...s.branches };
          if (typeof evt.branch_index === "number") branches[evt.branch_index] = evt;
          return {
            ...s,
            progress: typeof evt.progress === "number" ? evt.progress : s.progress,
            status: evt.status ?? s.status,
            branches,
            lastEvent: evt,
          };
        });
        if (evt.status && TERMINAL.has(evt.status)) {
          doneRef.current = true;
          ws.close();
        }
      };
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!doneRef.current) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      doneRef.current = true;
      if (retry) clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [jobId]);

  return state;
}
