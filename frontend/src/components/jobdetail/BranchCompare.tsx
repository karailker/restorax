import { useState } from "react";
import { ReactCompareSlider, ReactCompareSliderImage } from "react-compare-slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { API_BASE } from "@/lib/api";
import type { CombinedBranch } from "./branches";

interface BranchCompareProps {
  branches: CombinedBranch[];
}

/** Build an image URL for a branch preview served under the API base. */
function previewImageUrl(previewUrl: string): string {
  const clean = previewUrl.replace(/^\/+/, "");
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
    <div className="flex flex-col gap-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <Select value={String(value)} onValueChange={(v) => onChange(Number(v))}>
        <SelectTrigger className="w-52">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((b) => (
            <SelectItem key={b.branch_index} value={String(b.branch_index)}>
              {b.name || b.node_id || `branch ${b.branch_index}`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

/**
 * Side-by-side comparison of two branch previews. Renders only when at least two
 * branches actually recorded a preview image (the parallel node writes one PNG per
 * branch); nothing here fakes a comparison when no artifact exists.
 */
export function BranchCompare({ branches }: BranchCompareProps) {
  const withPreview = branches.filter(
    (b): b is CombinedBranch & { preview_url: string } => Boolean(b.preview_url),
  );

  const [leftIdx, setLeftIdx] = useState(withPreview[0]?.branch_index ?? 0);
  const [rightIdx, setRightIdx] = useState(
    withPreview[1]?.branch_index ?? withPreview[0]?.branch_index ?? 0,
  );

  if (withPreview.length < 2) return null;

  const left = withPreview.find((b) => b.branch_index === leftIdx) ?? withPreview[0];
  const right = withPreview.find((b) => b.branch_index === rightIdx) ?? withPreview[1];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Compare branches</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-4">
          <BranchSelector label="Left" value={left.branch_index} options={withPreview} onChange={setLeftIdx} />
          <BranchSelector label="Right" value={right.branch_index} options={withPreview} onChange={setRightIdx} />
        </div>
        <div className="overflow-hidden rounded-lg border border-border">
          <ReactCompareSlider
            itemOne={
              <ReactCompareSliderImage
                src={previewImageUrl(left.preview_url)}
                alt={left.name || `branch ${left.branch_index}`}
              />
            }
            itemTwo={
              <ReactCompareSliderImage
                src={previewImageUrl(right.preview_url)}
                alt={right.name || `branch ${right.branch_index}`}
              />
            }
          />
        </div>
        <p className="text-xs text-muted-foreground">
          First frame of each branch. The merge strategy is decided in the pipeline DAG
          (Merge node) before the job runs.
        </p>
      </CardContent>
    </Card>
  );
}
