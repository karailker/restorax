import { useEffect, useRef, useState } from "react";
import type { ParamSpec, RestorerInfo } from "@/types";
import { fetchModels } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  BuilderNode,
  BuilderNodeData,
  MergeNodeData,
  RestoreNodeData,
} from "./types";

const labelCls = "mb-1.5 text-xs font-medium text-muted-foreground";

interface Props {
  node: BuilderNode | null;
  /** Patch the selected node's data (shallow merge). */
  onChange: (id: string, patch: Partial<BuilderNodeData>) => void;
}

export function ConfigPanel({ node, onChange }: Props) {
  const [models, setModels] = useState<RestorerInfo[]>([]);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((m: RestorerInfo[]) => alive && setModels(m))
      .catch(() => alive && setModels([]));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <aside className="flex w-72 shrink-0 flex-col overflow-y-auto border-l border-border p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Node Config
      </h2>
      {!node ? (
        <p className="text-sm text-muted-foreground">
          Select a node to edit its settings.
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          <div>
            <Label className={labelCls}>Label</Label>
            <Input
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
            <p className="text-sm text-muted-foreground">
              Structural node — no extra settings.
            </p>
          )}
        </div>
      )}
    </aside>
  );
}

type Params = Record<string, unknown>;

/** Read a spec's current value from the params dict, falling back to its default. */
function readParam(params: Params, spec: ParamSpec): unknown {
  if (spec.target === "param") {
    return params[spec.name] ?? spec.default;
  }
  const extra = (params.extra as Params | undefined) ?? {};
  return extra[spec.name] ?? spec.default;
}

/** Return a new params dict with `spec` set to `value`, placed per its target. */
function writeParam(params: Params, spec: ParamSpec, value: unknown): Params {
  if (spec.target === "param") {
    return { ...params, [spec.name]: value };
  }
  const extra = (params.extra as Params | undefined) ?? {};
  return { ...params, extra: { ...extra, [spec.name]: value } };
}

function RestoreFields({
  node,
  models,
  onChange,
}: {
  node: BuilderNode;
  models: RestorerInfo[];
  onChange: Props["onChange"];
}) {
  const data = node.data as RestoreNodeData;
  const params = data.params as Params;
  const schema = models.find((m) => m.name === data.restorer_name)?.param_schema ?? [];

  const setParams = (next: Params) => onChange(node.id, { params: next });
  const setSpec = (spec: ParamSpec, value: unknown) =>
    setParams(writeParam(params, spec, value));

  const names = models.map((m) => m.name);
  const options =
    data.restorer_name && !names.includes(data.restorer_name)
      ? [data.restorer_name, ...names]
      : names;

  return (
    <>
      <div>
        <Label className={labelCls}>Restorer</Label>
        <Select
          value={data.restorer_name || undefined}
          onValueChange={(v) => onChange(node.id, { restorer_name: v })}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="— select —" />
          </SelectTrigger>
          <SelectContent>
            {options.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {data.restorer_name && schema.length === 0 && (
        <p className="text-xs text-muted-foreground">
          This restorer has no tunable parameters.
        </p>
      )}

      {schema.map((spec) => (
        <ParamField
          key={spec.name}
          spec={spec}
          value={readParam(params, spec)}
          onChange={(v) => setSpec(spec, v)}
        />
      ))}

      <AdvancedExtra params={params} onChange={setParams} nodeId={node.id} />
    </>
  );
}

function ParamField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <Label className="text-xs font-medium text-muted-foreground">
          {spec.label}
        </Label>
        {spec.kind === "bool" && (
          <Switch
            checked={Boolean(value)}
            onCheckedChange={(checked) => onChange(checked)}
          />
        )}
      </div>

      {(spec.kind === "int" || spec.kind === "float") && (
        <Input
          type="number"
          value={value as number}
          min={spec.minimum ?? undefined}
          max={spec.maximum ?? undefined}
          step={spec.step ?? (spec.kind === "int" ? 1 : "any")}
          onChange={(e) => {
            const n = spec.kind === "int" ? parseInt(e.target.value, 10) : Number(e.target.value);
            if (!Number.isNaN(n)) onChange(n);
          }}
        />
      )}

      {spec.kind === "enum" && spec.choices && (
        <Select
          value={String(value)}
          onValueChange={(v) => {
            const choice = spec.choices!.find((c) => String(c) === v);
            onChange(choice ?? v);
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {spec.choices.map((c) => (
              <SelectItem key={String(c)} value={String(c)}>
                {String(c)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {spec.kind === "multiselect" && spec.choices && (
        <MultiSelect
          choices={spec.choices}
          value={Array.isArray(value) ? value : []}
          onChange={onChange}
        />
      )}

      {spec.help && (
        <p className="mt-1 text-[11px] text-muted-foreground">{spec.help}</p>
      )}
    </div>
  );
}

function MultiSelect({
  choices,
  value,
  onChange,
}: {
  choices: unknown[];
  value: unknown[];
  onChange: (value: unknown[]) => void;
}) {
  const toggle = (c: unknown) => {
    const has = value.some((v) => String(v) === String(c));
    onChange(has ? value.filter((v) => String(v) !== String(c)) : [...value, c]);
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {choices.map((c) => {
        const active = value.some((v) => String(v) === String(c));
        return (
          <Button
            key={String(c)}
            type="button"
            size="sm"
            variant={active ? "default" : "outline"}
            className="h-7 px-2 text-xs"
            onClick={() => toggle(c)}
          >
            {String(c)}
          </Button>
        );
      })}
    </div>
  );
}

/**
 * Collapsible escape hatch for restorer-specific keys, scoped to `params.extra`
 * only — so the typed widgets keep sole control of the top-level RestorerParams
 * fields and a stray key here can never crash `RestorerParams(**params_dict)`.
 * Re-syncs from upstream (typed-widget edits, node switch) while not being typed in.
 */
function AdvancedExtra({
  params,
  onChange,
  nodeId,
}: {
  params: Params;
  onChange: (params: Params) => void;
  nodeId: string;
}) {
  const extra = (params.extra as Params | undefined) ?? {};
  const [text, setText] = useState(() => JSON.stringify(extra, null, 2));
  const [err, setErr] = useState(false);
  const editing = useRef(false);

  useEffect(() => {
    if (editing.current) return; // don't clobber active typing
    setText(JSON.stringify(((params.extra as Params | undefined) ?? {}), null, 2));
    setErr(false);
  }, [params, nodeId]);

  return (
    <details className="group">
      <summary className="cursor-pointer text-xs font-medium text-muted-foreground hover:text-foreground">
        Advanced — extra params (JSON)
      </summary>
      <Textarea
        className={cn(
          "mt-2 h-28 resize-y font-mono text-xs",
          err && "border-destructive",
        )}
        value={text}
        onFocus={() => {
          editing.current = true;
        }}
        onBlur={() => {
          editing.current = false;
        }}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          try {
            const parsed = JSON.parse(next || "{}");
            if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
              setErr(true);
              return;
            }
            setErr(false);
            onChange({ ...params, extra: parsed });
          } catch {
            setErr(true);
          }
        }}
      />
      <p className="mt-1 text-[11px] text-muted-foreground">
        Restorer-specific keys, merged into <code>params.extra</code>.
      </p>
      {err && (
        <p className="mt-1 text-xs text-destructive">
          Must be a JSON object — not saved.
        </p>
      )}
    </details>
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
        <Label className={labelCls}>Strategy</Label>
        <Select
          value={data.strategy}
          onValueChange={(v) =>
            onChange(node.id, { strategy: v as MergeNodeData["strategy"] })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="blend">blend</SelectItem>
            <SelectItem value="select">select</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {data.strategy === "select" && (
        <div>
          <Label className={labelCls}>Select index</Label>
          <Input
            type="number"
            min={0}
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
