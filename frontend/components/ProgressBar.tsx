"use client";

import { useEffect, useRef, useState } from "react";
import { wsJobProgress, type Job } from "@/lib/api";

interface ProgressEvent {
  job_id: string;
  progress: number;
  status: string;
  output_path?: string;
  error?: string;
}

interface Props {
  jobId: string;
  initialStatus: Job["status"];
  initialProgress: number;
  onComplete?: (outputPath: string) => void;
  onFail?: (error: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  pending:   "bg-gray-400",
  queued:    "bg-yellow-400",
  running:   "bg-indigo-500",
  completed: "bg-green-500",
  failed:    "bg-red-500",
  cancelled: "bg-gray-500",
};

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

export default function ProgressBar({ jobId, initialStatus, initialProgress, onComplete, onFail }: Props) {
  const [progress, setProgress] = useState(initialProgress);
  const [status, setStatus] = useState<string>(initialStatus);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (TERMINAL.has(status)) return;

    const ws = wsJobProgress(jobId);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      const data: ProgressEvent = JSON.parse(evt.data);
      setProgress(data.progress);
      setStatus(data.status);
      if (data.status === "completed" && data.output_path) {
        onComplete?.(data.output_path);
      }
      if (data.status === "failed" && data.error) {
        onFail?.(data.error);
      }
    };

    ws.onerror = () => setStatus("failed");

    return () => { ws.close(); };
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  const pct = Math.round(progress * 100);
  const barColor = STATUS_COLORS[status] ?? "bg-gray-400";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-500">
        <span className="capitalize font-medium">{status}</span>
        <span>{pct}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
