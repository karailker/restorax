import { useEffect, useState } from "react";
import type { RestorerInfo } from "@/types";
import { fetchModels } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  BuilderNode,
  BuilderNodeData,
  MergeNodeData,
  RestoreNodeData,
} from "./types";

const fieldCls =
  "h-9 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]";
const labelCls =
  "mb-1 block text-xs font-medium text-[var(--color-muted-foreground)]";

interface Props {
  node: BuilderNode | null;
  /** Patch the selected node's data (shallow merge). */
  onChange: (id: string, patch: Partial<BuilderNodeData>) => void;
}

export function ConfigPanel({ node, onChange }: Props) {
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((m: RestorerInfo[]) => alive && setModels(m.map((r) => r.name)))
      .catch(() => alive && setModels([]));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <aside className="flex w-72 shrink-0 flex-col overflow-y-auto border-l border-[var(--color-border)] p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
        Node Config
      </h2>
      {!node ? (
        <p className="text-sm text-[var(--color-muted-foreground)]">
          Select a node to edit its settings.
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          <div>
            <label className={labelCls}>Label</label>
            <input
              className={fieldCls}
              value={node.data.label}
              onChange={(e) => onChange(node.id, { label: e.target.value })}
            />
          </div>

          {node.data.kind === "restore" && (
            <RestoreFields node={node} models={models} onChange={onChange} />
          )}
          {node.data.kind === "merge" && (
            <MergeFields node={node} onChange={onChange} />
          )}
          {(node.data.kind === "parallel" ||
            node.data.kind === "pass" ||
            node.data.kind === "video_input" ||
            node.data.kind === "video_output") && (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Structural node — no extra settings.
            </p>
          )}
        </div>
      )}
    </aside>
  );
}

function RestoreFields({
  node,
  models,
  onChange,
}: {
  node: BuilderNode;
  models: string[];
  onChange: Props["onChange"];
}) {
  const data = node.data as RestoreNodeData;
  // Edit params as raw JSON; keep it simple and forgiving.
  const [paramsText, setParamsText] = useState(() =>
    JSON.stringify(data.params, null, 2),
  );
  const [paramsErr, setParamsErr] = useState(false);

  useEffect(() => {
    setParamsText(JSON.stringify(data.params, null, 2));
    setParamsErr(false);
    // Re-sync when a different node is selected.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node.id]);

  const options =
    data.restorer_name && !models.includes(data.restorer_name)
      ? [data.restorer_name, ...models]
      : models;

  return (
    <>
      <div>
        <label className={labelCls}>Restorer</label>
        <select
          className={fieldCls}
          value={data.restorer_name}
          onChange={(e) => onChange(node.id, { restorer_name: e.target.value })}
        >
          <option value="">— select —</option>
          {options.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label className={labelCls}>Params (JSON)</label>
        <textarea
          className={cn(
            fieldCls,
            "h-28 resize-y py-2 font-mono text-xs",
            paramsErr && "border-[var(--color-destructive)]",
          )}
          value={paramsText}
          onChange={(e) => {
            const text = e.target.value;
            setParamsText(text);
            try {
              const parsed = JSON.parse(text || "{}");
              setParamsErr(false);
              onChange(node.id, { params: parsed });
            } catch {
              setParamsErr(true);
            }
          }}
        />
        {paramsErr && (
          <p className="mt-1 text-xs text-[var(--color-destructive)]">
            Invalid JSON — not saved.
          </p>
        )}
      </div>
    </>
  );
}

function MergeFields({
  node,
  onChange,
}: {
  node: BuilderNode;
  onChange: Props["onChange"];
}) {
  const data = node.data as MergeNodeData;
  return (
    <>
      <div>
        <label className={labelCls}>Strategy</label>
        <select
          className={fieldCls}
          value={data.strategy}
          onChange={(e) =>
            onChange(node.id, {
              strategy: e.target.value as MergeNodeData["strategy"],
            })
          }
        >
          <option value="blend">blend</option>
          <option value="select">select</option>
        </select>
      </div>
      {data.strategy === "select" && (
        <div>
          <label className={labelCls}>Select index</label>
          <input
            type="number"
            min={0}
            className={fieldCls}
            value={data.select_index}
            onChange={(e) =>
              onChange(node.id, {
                select_index: Math.max(0, Number(e.target.value) || 0),
              })
            }
          />
        </div>
      )}
    </>
  );
}
