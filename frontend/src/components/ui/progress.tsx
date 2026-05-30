import { cn } from "@/lib/utils";

interface ProgressProps {
  /** 0..1 */
  value: number;
  className?: string;
  indicatorClassName?: string;
}

export function Progress({ value, className, indicatorClassName }: ProgressProps) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className={cn("h-2 w-full overflow-hidden rounded-full bg-[var(--color-secondary)]", className)}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={cn("h-full rounded-full bg-[var(--color-primary)] transition-[width] duration-300", indicatorClassName)}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
