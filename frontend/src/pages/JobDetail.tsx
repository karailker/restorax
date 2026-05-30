import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge, statusVariant } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { fetchJob } from "@/lib/api";
import { useJobProgress } from "@/hooks/useJobProgress";
import type { Job } from "@/types";

/**
 * Job Detail — live progress for a single job.
 * FOUNDATION STUB: overall + per-branch progress via WebSocket. Side-by-side
 * branch comparison (CompareSlider) and the blend/select merge UI are layered on
 * by the Job Detail view work.
 */
export default function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const live = useJobProgress(jobId);

  useEffect(() => {
    if (jobId) fetchJob(jobId).then(setJob).catch(() => setJob(null));
  }, [jobId]);

  const status = live.status ?? job?.status ?? "pending";
  const progress = live.progress || job?.progress || 0;
  const branches = Object.values(live.branches).sort((a, b) => (a.branch_index ?? 0) - (b.branch_index ?? 0));

  return (
    <div className="mx-auto max-w-4xl p-8">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Job</h1>
          <p className="font-mono text-sm text-[var(--color-muted-foreground)]">{jobId}</p>
        </div>
        <div className="flex items-center gap-2">
          {live.connected && <span className="text-xs text-[var(--color-muted-foreground)]">live</span>}
          <Badge variant={statusVariant(status)}>{status}</Badge>
        </div>
      </header>

      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Overall progress</CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={progress} />
          <p className="mt-2 text-right text-sm text-[var(--color-muted-foreground)]">
            {Math.round(progress * 100)}%
          </p>
        </CardContent>
      </Card>

      {branches.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Branches</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {branches.map((b) => (
              <div key={b.branch_index} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="font-mono">{b.node_id ?? `branch ${b.branch_index}`}</span>
                  <span className="text-[var(--color-muted-foreground)]">{Math.round((b.progress ?? 0) * 100)}%</span>
                </div>
                <Progress value={b.progress ?? 0} />
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {job?.error && (
        <Card className="mt-4 border-[var(--color-destructive)]/40">
          <CardContent className="pt-5 text-sm text-[var(--color-destructive)]">{job.error}</CardContent>
        </Card>
      )}
    </div>
  );
}
