import { Card, CardContent } from "@/components/ui/card";
import { Layers, Loader2, CheckCircle2, type LucideIcon } from "lucide-react";
import type { Job } from "@/types";
import { cn } from "@/lib/utils";

interface StatsStripProps {
  jobs: Job[];
}

interface Stat {
  label: string;
  value: number;
  icon: LucideIcon;
  accent: string;
}

export function StatsStrip({ jobs }: StatsStripProps) {
  const running = jobs.filter(
    (j) => j.status === "running" || j.status === "queued" || j.status === "pending",
  ).length;
  const completed = jobs.filter((j) => j.status === "completed").length;

  const stats: Stat[] = [
    { label: "Total jobs", value: jobs.length, icon: Layers, accent: "text-[var(--color-primary)]" },
    { label: "Running", value: running, icon: Loader2, accent: "text-[var(--color-warning)]" },
    { label: "Completed", value: completed, icon: CheckCircle2, accent: "text-[var(--color-success)]" },
  ];

  return (
    <div className="grid grid-cols-3 gap-3">
      {stats.map((s) => {
        const Icon = s.icon;
        return (
          <Card key={s.label}>
            <CardContent className="flex items-center gap-3 pt-5">
              <span
                className={cn(
                  "flex size-9 items-center justify-center rounded-lg bg-[var(--color-secondary)]",
                  s.accent,
                )}
              >
                <Icon className="size-4" />
              </span>
              <div>
                <p className="text-xl font-semibold tabular-nums leading-none">{s.value}</p>
                <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">{s.label}</p>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
