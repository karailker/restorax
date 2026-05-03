const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Job {
  id: string;
  status: "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  pipeline_id: string;
  input_path: string;
  output_path: string | null;
  error: string | null;
  metrics: Record<string, number>;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  celery_task_id: string | null;
}

export interface RestorerInfo {
  name: string;
  category: string;
  scale_factor: number;
  min_vram_gb: number;
  tags: string[];
  loaded: boolean;
}

export async function fetchJobs(limit = 20): Promise<Job[]> {
  const res = await fetch(`${API_BASE}/jobs?limit=${limit}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch jobs: ${res.status}`);
  const data = await res.json();
  return data.jobs as Job[];
}

export async function fetchJob(id: string): Promise<Job> {
  const res = await fetch(`${API_BASE}/jobs/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Job ${id} not found`);
  return res.json();
}

export async function submitJob(
  file: File,
  pipelineId: string,
  opts?: { outputFormat?: string; outputCodec?: string; outputCrf?: number; preserveAudio?: boolean }
): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  form.append("pipeline_id", pipelineId);
  form.append("output_format", opts?.outputFormat ?? "mp4");
  form.append("output_codec", opts?.outputCodec ?? "libx264");
  form.append("output_crf", String(opts?.outputCrf ?? 18));
  form.append("preserve_audio", String(opts?.preserveAudio ?? true));

  const res = await fetch(`${API_BASE}/jobs`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Failed to submit job");
  }
  return res.json();
}

export async function fetchModels(): Promise<RestorerInfo[]> {
  const res = await fetch(`${API_BASE}/models`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch models");
  const data = await res.json();
  return data.restorers;
}

export function wsJobProgress(jobId: string): WebSocket {
  const wsBase = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000")
    .replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/ws/jobs/${jobId}/progress`);
}
