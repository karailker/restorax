import type {
  BranchList,
  CeleryHealth,
  DAGConfig,
  DAGResponse,
  Job,
  RestorerInfo,
} from "@/types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Jobs ──────────────────────────────────────────────────────────────────────

export async function fetchJobs(limit = 20): Promise<Job[]> {
  const data = await request<{ jobs: Job[]; total: number }>(`/jobs?limit=${limit}`);
  return data.jobs;
}

export function fetchJob(id: string): Promise<Job> {
  return request<Job>(`/jobs/${id}`);
}

export interface SubmitJobOpts {
  outputFormat?: string;
  outputCodec?: string;
  outputCrf?: number;
  preserveAudio?: boolean;
}

/** Submit a job against a sequential pipeline preset OR a DAG (pass `dagId`). */
export async function submitJob(
  file: File,
  target: { pipelineId: string } | { dagId: string },
  opts: SubmitJobOpts = {},
): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  if ("pipelineId" in target) form.append("pipeline_id", target.pipelineId);
  else form.append("dag_id", target.dagId);
  form.append("output_format", opts.outputFormat ?? "mp4");
  form.append("output_codec", opts.outputCodec ?? "libx264");
  form.append("output_crf", String(opts.outputCrf ?? 18));
  form.append("preserve_audio", String(opts.preserveAudio ?? true));
  return request<Job>(`/jobs`, { method: "POST", body: form });
}

export function fetchJobBranches(id: string): Promise<BranchList> {
  return request<BranchList>(`/jobs/${id}/branches`);
}

export function mergeJobBranches(
  id: string,
  body: { strategy: "blend" | "select"; branch_index?: number },
): Promise<Job> {
  return request<Job>(`/jobs/${id}/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function jobDownloadUrl(id: string): string {
  return `${API_BASE}/jobs/${id}/download`;
}

// ── Models ────────────────────────────────────────────────────────────────────

export async function fetchModels(): Promise<RestorerInfo[]> {
  const data = await request<{ restorers: RestorerInfo[] }>(`/models`);
  return data.restorers;
}

// ── DAG pipelines ─────────────────────────────────────────────────────────────

export function createDag(payload: {
  id: string;
  name: string;
  description?: string;
  config: DAGConfig;
}): Promise<DAGResponse> {
  return request<DAGResponse>(`/pipelines/dag`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchDag(id: string): Promise<DAGResponse> {
  return request<DAGResponse>(`/pipelines/dag/${id}`);
}

// ── Health ────────────────────────────────────────────────────────────────────

export function fetchCeleryHealth(): Promise<CeleryHealth> {
  return request<CeleryHealth>(`/health/celery`);
}

export { API_BASE };
