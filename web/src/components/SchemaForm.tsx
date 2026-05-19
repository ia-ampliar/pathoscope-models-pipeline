import { useMemo } from "react";
import { clsx } from "clsx";
import { ParamCard } from "./ParamCard";
import { isEssential } from "../paramHelp";

type JsonSchema = {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: (string | number)[];
  minimum?: number;
  maximum?: number;
  properties?: Record<string, JsonSchema>;
  anyOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
};

function deref(prop: JsonSchema | undefined, root: JsonSchema): JsonSchema | undefined {
  if (!prop) return undefined;
  if (prop.$ref) {
    const name = prop.$ref.replace(/^#\/\$defs\//, "");
    return deref(root.$defs?.[name], root);
  }
  return prop;
}

function resolveSchema(prop: JsonSchema | undefined, root: JsonSchema): JsonSchema | undefined {
  const p = deref(prop, root);
  if (!p) return undefined;
  if (p.anyOf?.length) {
    const nonNull = p.anyOf.find((s) => s.type !== "null" && s.type !== undefined);
    return nonNull ? resolveSchema(nonNull, root) : p;
  }
  return p;
}

type Group = { label: string; keys: string[] };

export function SchemaForm({
  schemaRoot,
  values,
  onChange,
  lrRiskCheck,
  groups,
  inspectorMode = true,
}: {
  schemaRoot: JsonSchema;
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  lrRiskCheck?: (key: string, v: number) => boolean;
  groups?: Group[];
  inspectorMode?: boolean;
}) {
  const allKeys = useMemo(() => {
    const p = schemaRoot.properties || {};
    return Object.keys(p).sort();
  }, [schemaRoot]);

  const setField = (k: string, v: unknown) => onChange({ ...values, [k]: v });

  const shouldShow = (key: string) => inspectorMode || isEssential(key);

  const renderField = (key: string) => {
    const raw = schemaRoot.properties?.[key];
    const p = resolveSchema(raw, schemaRoot);
    if (!p) return null;
    if (!shouldShow(key)) return null;

    const desc = p.description || "";
    const label = (p.title || key).replace(/_/g, " ");
    const val = values[key];
    const risky = lrRiskCheck?.(key, typeof val === "number" ? val : 0) ?? false;

    if (p.type === "boolean") {
      return (
        <ParamCard key={key} paramKey={key} label={label} description={desc} inspectorMode={inspectorMode}>
          <label className="flex cursor-pointer items-center gap-2 py-0.5">
            <input
              type="checkbox"
              className="h-3.5 w-3.5 rounded border-border-strong bg-elevated accent-accent"
              checked={Boolean(val)}
              onChange={(e) => setField(key, e.target.checked)}
            />
            <span className="text-sm text-text-primary">{Boolean(val) ? "Ligado" : "Desligado"}</span>
          </label>
        </ParamCard>
      );
    }

    if (p.enum) {
      return (
        <ParamCard key={key} paramKey={key} label={label} description={desc} inspectorMode={inspectorMode}>
          <select
            className="w-full rounded border border-border-strong bg-elevated px-2 py-1.5 font-mono text-sm text-text-primary focus:border-accent/60 focus:outline-none"
            value={String(val ?? "")}
            onChange={(e) => setField(key, e.target.value)}
          >
            {p.enum.map((opt) => (
              <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
            ))}
          </select>
        </ParamCard>
      );
    }

    if (p.type === "integer" || p.type === "number") {
      const num = typeof val === "number" ? val : Number(val ?? 0);
      const min = p.minimum;
      const max = p.maximum;
      return (
        <ParamCard key={key} paramKey={key} label={label} description={desc} risky={risky} inspectorMode={inspectorMode}>
          <input
            type="number"
            min={min}
            max={max}
            step={p.type === "integer" ? 1 : "any"}
            className={clsx(
              "w-full rounded border bg-elevated px-2 py-1.5 font-mono text-sm text-text-primary focus:outline-none focus:border-accent/60",
              risky ? "border-warn/60 ring-1 ring-warn/20" : "border-border-strong"
            )}
            value={Number.isFinite(num) ? num : 0}
            onChange={(e) => {
              const v = p.type === "integer" ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
              setField(key, v);
            }}
          />
          {min != null && max != null && (
            <input
              type="range"
              min={min}
              max={max}
              step={p.type === "integer" ? 1 : (max - min) / 200}
              className="mt-1.5 w-full accent-accent"
              value={Number.isFinite(num) ? num : min}
              onChange={(e) => {
                const v = p.type === "integer" ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
                setField(key, v);
              }}
            />
          )}
        </ParamCard>
      );
    }

    if (p.type === "string" || (!p.type && !p.enum)) {
      return (
        <ParamCard key={key} paramKey={key} label={label} description={desc} inspectorMode={inspectorMode}>
          <input
            type="text"
            className="w-full rounded border border-border-strong bg-elevated px-2 py-1.5 font-mono text-sm text-text-primary placeholder-text-muted focus:border-accent/60 focus:outline-none"
            value={val != null ? String(val) : ""}
            onChange={(e) => {
              const t = e.target.value;
              setField(key, t === "" ? null : t);
            }}
            placeholder={desc}
          />
        </ParamCard>
      );
    }

    return null;
  };

  if (groups) {
    return (
      <div className="space-y-4">
        {groups.map((g) => {
          const visibleKeys = g.keys.filter((k) => shouldShow(k) && schemaRoot.properties?.[k]);
          if (visibleKeys.length === 0) return null;
          return (
            <div key={g.label}>
              <div className="mb-2 flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">{g.label}</span>
                <div className="h-px flex-1 bg-border-subtle" />
              </div>
              <div className="space-y-2">
                {visibleKeys.map(renderField)}
              </div>
            </div>
          );
        })}
        {/* Ungrouped keys */}
        {allKeys
          .filter((k) => !groups.some((g) => g.keys.includes(k)))
          .filter(shouldShow)
          .filter((k) => schemaRoot.properties?.[k])
          .map(renderField)
        }
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {allKeys.map(renderField)}
    </div>
  );
}
