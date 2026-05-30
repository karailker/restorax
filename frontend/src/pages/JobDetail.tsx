import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge, statusVariant } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { fetchJob, fetchJobBranches } from "@/lib/api";
import { formatRelativeTime } from "@/lib/utils";
import { useJobProgress } from "@/hooks/useJobProgress";
import type { BranchInfo, Job } from "@/types";
import { combineBranches } from "@/components/jobdetail/branches";
import { BranchList } from "@/components/jobdetail/BranchList";
import { BranchCompare } from "@/components/jobdetail/BranchCompare";
import { MergePanel } from "@/components/jobdetail/MergePanel";
import { ResultPanel } from "@/components/jobdetail/ResultPanel";

/**
 * Job Detail — live progress + branch comparison/merge for a single job.
 * Combines a one-shot fetch (job + branch list) with a live WebSocket feed.
 */
export default function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [fetchedBranches, setFetchedBranches] = useState<BranchInfo[]>([]);
  const live = useJobProgress(jobId);

  const reload = useCallback(() => {
    if (!jobId) return;
    fetchJob(jobId).then(setJob).catch(() => setJob(null));
    fetchJobBranches(jobId)
      .then((bl) => setFetchedBranches(bl.branches))
      .catch(() => setFetchedBranches([]));
  }, [jobId]);

  useEffect(() => {
    reload();
  }, [reload]);

  // Refresh once the live feed reports a terminal status (output_path, metrics).
  const liveStatus = live.status;
  useEffect(() => {
    if (liveStatus === "completed" || liveStatus === "failed") reload();
  }, [liveStatus, reload]);

  const status = live.status ?? job?.status ?? "pending";
  const progress = live.progress || job?.progress || 0;
  const branches = combineBranches(fetchedBranches, live.branches);

  const isCompleted = status === "completed";
  const hasOutput = Boolean(job?.output_path) || isCompleted;
  const canMerge = branches.length >= 2 && !isCompleted;

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-8">
      <header className="flex items-start justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">Job</h1>
          <p className="font-mono text-sm text-[var(--color-muted-foreground)]">{jobId}</p>
          {job && (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Pipeline <span className="font-mono">{job.pipeline_id}</span>
              {" · created "}
              {formatRelativeTime(job.created_at)}
              {job.completed_at && ` · finished ${formatRelativeTime(job.completed_at)}`}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {live.connected && (
            <span className="flex items-center gap-1.5 text-xs text-[var(--color-muted-foreground)]">
              <span className="size-2 rounded-full bg-[var(--color-success)]" />
              live
            </span>
          )}
          <Badge variant={statusVariant(status)}>{status}</Badge>
        </div>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Overall progress</CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={progress} />
          <p className="mt-2 text-right text-sm tabular-nums text-[var(--color-muted-foreground)]">
            {Math.round(progress * 100)}%
          </p>
        </CardContent>
      </Card>

      {branches.length > 0 && <BranchList branches={branches} />}

      <BranchCompare branches={branches} />

      {canMerge && jobId && (
        <MergePanel jobId={jobId} branches={branches} onMerged={setJob} />
      )}

      {hasOutput && job && <ResultPanel job={job} />}

      {job?.error && (
        <Card className="border-[var(--color-destructive)]/40">
          <CardHeader>
            <CardTitle className="text-[var(--color-destructive)]">Error</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-[var(--color-destructive)]">
            {job.error}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
