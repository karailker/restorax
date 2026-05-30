import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge, statusVariant } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { fetchJobs } from "@/lib/api";
import { formatRelativeTime } from "@/lib/utils";
import type { Job } from "@/types";

/**
 * Dashboard — recent jobs + entry points.
 * FOUNDATION STUB: lists jobs with status. Quick-launch preset cards and richer
 * widgets are layered on by the Dashboard view work.
 */
export default function Dashboard() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJobs().then(setJobs).catch((e) => setError(String(e.message ?? e)));
  }, []);

  return (
    <div className="mx-auto max-w-5xl p-8">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-[var(--color-muted-foreground)]">Recent restoration jobs</p>
        </div>
        <Link
          to="/builder"
          className="rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-[var(--color-primary-foreground)] hover:opacity-90"
        >
          New pipeline
        </Link>
      </header>

      {error && (
        <Card className="mb-4 border-[var(--color-destructive)]/40">
          <CardContent className="pt-5 text-sm text-[var(--color-destructive)]">
            Could not load jobs: {error}
          </CardContent>
        </Card>
      )}

      {jobs.length === 0 && !error ? (
        <Card>
          <CardContent className="py-12 text-center text-sm text-[var(--color-muted-foreground)]">
            No jobs yet. Build a pipeline to get started.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3">
          {jobs.map((job) => (
            <Link key={job.id} to={`/jobs/${job.id}`}>
              <Card className="transition-colors hover:border-[var(--color-primary)]/40">
                <CardHeader className="flex-row items-center justify-between">
                  <CardTitle className="font-mono text-sm">{job.pipeline_id}</CardTitle>
                  <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Progress value={job.progress} />
                  <div className="flex justify-between text-xs text-[var(--color-muted-foreground)]">
                    <span className="font-mono">{job.id.slice(0, 8)}</span>
                    <span>{formatRelativeTime(job.created_at)}</span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
