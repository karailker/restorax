import { Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { Preset } from "./presets";

interface PresetCardProps {
  preset: Preset;
  onClick: (preset: Preset) => void;
  uploading?: boolean;
  disabled?: boolean;
}

export function PresetCard({ preset, onClick, uploading = false, disabled = false }: PresetCardProps) {
  const Icon = preset.icon;
  return (
    <button
      type="button"
      onClick={() => onClick(preset)}
      disabled={disabled}
      className="group text-left focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-60"
    >
      <Card className="h-full transition-colors group-hover:border-[var(--color-primary)]/50 group-focus-visible:ring-2 group-focus-visible:ring-[var(--color-ring)]">
        <CardContent className="flex flex-col gap-3 pt-5">
          <span
            className={cn(
              "flex size-10 items-center justify-center rounded-lg bg-[var(--color-secondary)]",
              preset.accent,
            )}
          >
            {uploading ? (
              <Loader2 className="size-5 animate-spin" />
            ) : (
              <Icon className="size-5" />
            )}
          </span>
          <div className="space-y-1">
            <p className="text-sm font-medium leading-tight">{preset.title}</p>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              {uploading ? "Uploading…" : preset.description}
            </p>
          </div>
        </CardContent>
      </Card>
    </button>
  );
}
