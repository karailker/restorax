import type { BranchInfo, ProgressEvent } from "@/types";

/** A branch row combining fetched static info with live progress events. */
export interface CombinedBranch {
  branch_index: number;
  /** Display label — fetched name, else live node_id, else fallback. */
  name: string;
  node_id?: string;
  status: string;
  progress: number;
  output_path: string | null;
}

/**
 * Merge fetched branch list (static, has names + output paths) with live
 * per-branch progress events (fresher status/progress). Returns rows sorted by
 * branch_index. Live values win for progress/status; fetched values supply
 * name + output_path.
 */
export function combineBranches(
  fetched: BranchInfo[],
  live: Record<number, ProgressEvent>,
): CombinedBranch[] {
  const byIndex = new Map<number, CombinedBranch>();

  for (const b of fetched) {
    byIndex.set(b.branch_index, {
      branch_index: b.branch_index,
      name: b.name,
      status: b.status,
      progress: b.progress,
      output_path: b.output_path,
    });
  }

  for (const [idx, evt] of Object.entries(live)) {
    const index = Number(idx);
    const existing = byIndex.get(index);
    if (existing) {
      existing.progress = typeof evt.progress === "number" ? evt.progress : existing.progress;
      existing.status = evt.status ?? existing.status;
      if (evt.node_id) existing.node_id = evt.node_id;
    } else {
      byIndex.set(index, {
        branch_index: index,
        name: evt.node_id ?? `branch ${index}`,
        node_id: evt.node_id,
        status: evt.status ?? "running",
        progress: typeof evt.progress === "number" ? evt.progress : 0,
        output_path: null,
      });
    }
  }

  return [...byIndex.values()].sort((a, b) => a.branch_index - b.branch_index);
}
