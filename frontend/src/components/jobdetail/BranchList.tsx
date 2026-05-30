import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge, statusVariant } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { CombinedBranch } from "./branches";

interface BranchListProps {
  branches: CombinedBranch[];
}

/** Per-branch progress + status for DAG jobs. */
export function BranchList({ branches }: BranchListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Branches</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {branches.map((b) => (
          <div key={b.branch_index} className="space-y-1.5">
            <div className="flex items-center justify-between gap-2 text-sm">
              <span className="font-mono text-[var(--color-foreground)]">
                {b.name || b.node_id || `branch ${b.branch_index}`}
              </span>
              <div className="flex items-center gap-2">
                <Badge variant={statusVariant(b.status)}>{b.status}</Badge>
                <span className="tabular-nums text-[var(--color-muted-foreground)]">
                  {Math.round(b.progress * 100)}%
                </span>
              </div>
            </div>
            <Progress value={b.progress} />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
