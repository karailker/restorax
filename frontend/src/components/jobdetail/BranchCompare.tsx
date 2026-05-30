import { useState } from "react";
import { ReactCompareSlider, ReactCompareSliderImage } from "react-compare-slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { API_BASE } from "@/lib/api";
import type { CombinedBranch } from "./branches";

interface BranchCompareProps {
  branches: CombinedBranch[];
}

/** Build an image URL for a branch output served under the API base. */
function branchImageUrl(outputPath: string): string {
  const clean = outputPath.replace(/^\/+/, "");
  return `${API_BASE}/${clean}`;
}

interface SelectorProps {
  label: string;
  value: number;
  options: CombinedBranch[];
  onChange: (index: number) => void;
}

function BranchSelector({ label, value, options, onChange }: SelectorProps) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="text-[var(--color-muted-foreground)]">{label}</span>
      <select
        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {options.map((b) => (
          <option key={b.branch_index} value={b.branch_index}>
            {b.name || b.node_id || `branch ${b.branch_index}`}
          </option>
        ))}
      </select>
    </label>
  );
}

/**
 * Side-by-side comparison of two branch outputs. Hidden by the parent when
 * fewer than two branches have outputs.
 */
export function BranchCompare({ branches }: BranchCompareProps) {
  const withOutput = branches.filter(
    (b): b is CombinedBranch & { output_path: string } => Boolean(b.output_path),
  );

  const [leftIdx, setLeftIdx] = useState(withOutput[0]?.branch_index ?? 0);
  const [rightIdx, setRightIdx] = useState(
    withOutput[1]?.branch_index ?? withOutput[0]?.branch_index ?? 0,
  );

  if (withOutput.length < 2) return null;

  const left = withOutput.find((b) => b.branch_index === leftIdx) ?? withOutput[0];
  const right = withOutput.find((b) => b.branch_index === rightIdx) ?? withOutput[1];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Compare branches</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-4">
          <BranchSelector label="Left" value={left.branch_index} options={withOutput} onChange={setLeftIdx} />
          <BranchSelector label="Right" value={right.branch_index} options={withOutput} onChange={setRightIdx} />
        </div>
        <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
          <ReactCompareSlider
            itemOne={
              <ReactCompareSliderImage
                src={branchImageUrl(left.output_path)}
                alt={left.name || `branch ${left.branch_index}`}
              />
            }
            itemTwo={
              <ReactCompareSliderImage
                src={branchImageUrl(right.output_path)}
                alt={right.name || `branch ${right.branch_index}`}
              />
            }
          />
        </div>
      </CardContent>
    </Card>
  );
}
