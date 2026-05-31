// Mirrors restorax/api/schemas — keep in sync with the backend.

export type JobStatus =
  | "pending"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface Job {
  id: string;
  status: JobStatus;
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

export interface ParamSpec {
  name: string;
  kind: "int" | "float" | "bool" | "enum" | "multiselect";
  default: unknown;
  label: string;
  /** "param" = top-level RestorerParams field; "extra" = nested under params.extra. */
  target: "param" | "extra";
  minimum?: number | null;
  maximum?: number | null;
  step?: number | null;
  choices?: unknown[] | null;
  help?: string | null;
}

export interface RestorerInfo {
  name: string;
  category: string;
  input_color_space?: string | null;
  output_color_space?: string | null;
  requires_temporal?: boolean | null;
  min_vram_gb?: number | null;
  scale_factor?: number | null;
  min_ram_gb?: number | null;
  supports_stereo?: boolean | null;
  sample_rates?: number[] | null;
  tags: string[];
  loaded: boolean;
  param_schema?: ParamSpec[];
}

export interface BranchInfo {
  branch_index: number;
  name: string;
  status: string;
  progress: number;
  output_path: string | null;
}

export interface BranchList {
  job_id: string;
  branches: BranchInfo[];
}

export interface DAGResponse {
  id: string;
  name: string;
  description: string;
  config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface CeleryHealth {
  status: string;
  workers: number;
  active_tasks: number;
  queued_tasks: number;
}

/** Progress event streamed over /ws/jobs/{id}/progress. DAG jobs add node_id/branch_index. */
export interface ProgressEvent {
  job_id: string;
  progress: number;
  status?: string;
  node_id?: string;
  branch_index?: number;
}

// ── DAG serialization (mirror of restorax/dag/serializer.py) ──────────────────

export interface DAGNode {
  type: string;
  id: string;
  name: string;
  [key: string]: unknown;
}

export interface DAGEdge {
  source_node_id: string;
  source_port: string;
  target_node_id: string;
  target_port: string;
}

export interface DAGConfig {
  schema_type: "dag";
  id: string;
  name: string;
  nodes: DAGNode[];
  edges: DAGEdge[];
}
