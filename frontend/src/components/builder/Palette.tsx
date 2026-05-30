import { useEffect, useState } from "react";
import type { RestorerInfo } from "@/types";
import { fetchModels } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { DRAG_MIME, type PaletteDragPayload } from "./types";

/** Minimal static fallback if /models is unavailable. */
const FALLBACK_MODELS: RestorerInfo[] = [
  { name: "real-esrgan", category: "upscale", tags: [], loaded: false },
  { name: "gfpgan", category: "face", tags: [], loaded: false },
  { name: "deoldify", category: "colorize", tags: [], loaded: false },
];

const STRUCTURAL: PaletteDragPayload[] = [
  { type: "parallel", label: "Parallel" },
  { type: "merge", label: "Merge" },
  { type: "pass", label: "Pass" },
];

function DraggableItem({
  payload,
  children,
}: {
  payload: PaletteDragPayload;
  children: React.ReactNode;
}) {
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload));
        e.dataTransfer.effectAllowed = "move";
      }}
      className={cn(
        "cursor-grab rounded-md border border-[var(--color-border)] bg-[var(--color-card)] px-3 py-2 text-sm",
        "hover:border-[var(--color-ring)] active:cursor-grabbing",
      )}
    >
      {children}
    </div>
  );
}

export function Palette() {
  const [models, setModels] = useState<RestorerInfo[]>([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((m) => alive && setModels(m))
      .catch(() => {
        if (!alive) return;
        setModels(FALLBACK_MODELS);
        setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, []);

  const byCategory = models.reduce<Record<string, RestorerInfo[]>>((acc, m) => {
    (acc[m.category] ??= []).push(m);
    return acc;
  }, {});

  return (
    <aside className="flex w-64 shrink-0 flex-col gap-4 overflow-y-auto border-r border-[var(--color-border)] p-4">
      <div>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
          Structure
        </h2>
        <div className="flex flex-col gap-2">
          {STRUCTURAL.map((s) => (
            <DraggableItem key={s.type} payload={s}>
              {s.label}
            </DraggableItem>
          ))}
        </div>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
            Restorers
          </h2>
          {failed && (
            <Badge variant="warning" className="text-[10px]">
              offline
            </Badge>
          )}
        </div>
        <div className="flex flex-col gap-3">
          {Object.entries(byCategory).map(([category, items]) => (
            <div key={category}>
              <p className="mb-1 text-[11px] font-medium capitalize text-[var(--color-muted-foreground)]">
                {category}
              </p>
              <div className="flex flex-col gap-2">
                {items.map((m) => (
                  <DraggableItem
                    key={m.name}
                    payload={{ type: "restore", label: m.name, restorer_name: m.name }}
                  >
                    <span className="flex items-center justify-between gap-2">
                      <span className="truncate">{m.name}</span>
                      {m.loaded && (
                        <Badge variant="success" className="text-[10px]">
                          loaded
                        </Badge>
                      )}
                    </span>
                  </DraggableItem>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
