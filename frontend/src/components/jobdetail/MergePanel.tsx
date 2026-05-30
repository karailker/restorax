import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { mergeJobBranches } from "@/lib/api";
import type { Job } from "@/types";
import type { CombinedBranch } from "./branches";

interface MergePanelProps {
  jobId: string;
  branches: CombinedBranch[];
  onMerged: (job: Job) => void;
}

type Strategy = "blend" | "select";

/** Merge DAG branches via blend (combine all) or select (pick one). */
export function MergePanel({ jobId, branches, onMerged }: MergePanelProps) {
  const [strategy, setStrategy] = useState<Strategy>("blend");
  const [branchIndex, setBranchIndex] = useState<number>(branches[0]?.branch_index ?? 0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  async function handleMerge() {
    setBusy(true);
    setError(null);
    setSuccess(false);
    try {
      const body =
        strategy === "select"
          ? { strategy, branch_index: branchIndex }
          : { strategy };
      const job = await mergeJobBranches(jobId, body);
      setSuccess(true);
      onMerged(job);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Merge failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Merge branches</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-[var(--color-muted-foreground)]">Strategy</span>
            <select
              className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as Strategy)}
            >
              <option value="blend">Blend (all)</option>
              <option value="select">Select one</option>
            </select>
          </label>

          {strategy === "select" && (
            <label className="flex items-center gap-2 text-sm">
              <span className="text-[var(--color-muted-foreground)]">Branch</span>
              <select
                className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
                value={branchIndex}
                onChange={(e) => setBranchIndex(Number(e.target.value))}
              >
                {branches.map((b) => (
                  <option key={b.branch_index} value={b.branch_index}>
                    {b.name || b.node_id || `branch ${b.branch_index}`}
                  </option>
                ))}
              </select>
            </label>
          )}

          <Button onClick={handleMerge} disabled={busy}>
            {busy ? "Merging…" : "Merge branches"}
          </Button>
        </div>

        {error && (
          <p className="text-sm text-[var(--color-destructive)]">{error}</p>
        )}
        {success && (
          <p className="text-sm text-[var(--color-success)]">Branches merged.</p>
        )}
      </CardContent>
    </Card>
  );
}
