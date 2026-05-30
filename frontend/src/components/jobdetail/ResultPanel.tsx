import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { jobDownloadUrl } from "@/lib/api";
import type { Job } from "@/types";

interface ResultPanelProps {
  job: Job;
}

/** Download link + metrics, shown once the job has produced output. */
export function ResultPanel({ job }: ResultPanelProps) {
  const metrics = Object.entries(job.metrics);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Result</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button asChild>
          <a href={jobDownloadUrl(job.id)} download>
            Download output
          </a>
        </Button>

        {metrics.length > 0 && (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {metrics.map(([key, value]) => (
              <div key={key} className="flex flex-col">
                <dt className="text-[var(--color-muted-foreground)]">{key}</dt>
                <dd className="font-mono tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        )}
      </CardContent>
    </Card>
  );
}
