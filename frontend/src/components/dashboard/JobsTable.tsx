import { Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge, statusVariant } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { formatRelativeTime } from "@/lib/utils";
import type { Job } from "@/types";

interface JobsTableProps {
  jobs: Job[];
}

export function JobsTable({ jobs }: JobsTableProps) {
  if (jobs.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-[var(--color-muted-foreground)]">
          No jobs yet. Pick a preset above or build a pipeline to get started.
        </CardContent>
      </Card>
    );
  }

  return (
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
  );
}
