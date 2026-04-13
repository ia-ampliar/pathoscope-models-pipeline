import { useMemo } from "react";

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
    const inner = root.$defs?.[name];
    return deref(inner, root);
  }
  return prop;
}

function resolveSchema(prop: JsonSchema | undefined, root: JsonSchema): JsonSchema | undefined {
  const p = deref(prop, root);
  if (!p) return undefined;
  if (p.anyOf && p.anyOf.length) {
    const nonNull = p.anyOf.find((s) => s.type !== "null" && s.type !== undefined);
    return nonNull ? resolveSchema(nonNull, root) : p;
  }
  return p;
}

export function SchemaForm({
  schemaRoot,
  values,
  onChange,
  lrRiskCheck,
}: {
  schemaRoot: JsonSchema;
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  lrRiskCheck?: (key: string, v: number) => boolean;
}) {
  const props = useMemo(() => {
    const s = schemaRoot;
    const p = s.properties || {};
    return Object.keys(p).sort();
  }, [schemaRoot]);

  const setField = (k: string, v: unknown) => {
    onChange({ ...values, [k]: v });
  };

  return (
    <div className="space-y-3 text-sm">
      {props.map((key) => {
        const raw = schemaRoot.properties?.[key];
        const p = resolveSchema(raw, schemaRoot);
        if (!p) return null;
        const desc = p.description || "";
        const label = p.title || key;
        const val = values[key];

        if (p.type === "boolean") {
          return (
            <label key={key} className="flex cursor-pointer items-center gap-2 text-slate-300">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-surface"
                checked={Boolean(val)}
                onChange={(e) => setField(key, e.target.checked)}
              />
              <span>{label}</span>
              {desc && <span className="text-xs text-slate-500">— {desc}</span>}
            </label>
          );
        }

        if (p.enum) {
          return (
            <div key={key}>
              <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">{label}</label>
              <select
                className="w-full rounded border border-slate-700 bg-surface px-2 py-1.5 text-slate-200"
                value={String(val ?? "")}
                onChange={(e) => {
                  const s = e.target.value;
                  setField(key, s);
                }}
              >
                {p.enum.map((opt) => (
                  <option key={String(opt)} value={String(opt)}>
                    {String(opt)}
                  </option>
                ))}
              </select>
              {desc && <p className="mt-0.5 text-xs text-slate-500">{desc}</p>}
            </div>
          );
        }

        if (p.type === "integer" || p.type === "number") {
          const num = typeof val === "number" ? val : Number(val ?? 0);
          const min = p.minimum;
          const max = p.maximum;
          const risky = lrRiskCheck?.(key, num) ?? false;
          return (
            <div key={key}>
              <label className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wide text-slate-500">
                {label}
                {risky && (
                  <span
                    className="cursor-help rounded border border-warn/60 bg-warn/10 px-1.5 text-[10px] font-normal normal-case text-warn"
                    title="Potencial instabilidade — LR alto para fine-tune."
                  >
                    risco
                  </span>
                )}
              </label>
              <input
                type="number"
                min={min}
                max={max}
                step={p.type === "integer" ? 1 : "any"}
                className={`w-full rounded border bg-surface px-2 py-1.5 text-slate-200 ${
                  risky ? "border-warn/70 ring-1 ring-warn/30" : "border-slate-700"
                }`}
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
                  className="mt-1 w-full accent-accent"
                  value={Number.isFinite(num) ? num : min}
                  onChange={(e) => {
                    const v = p.type === "integer" ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
                    setField(key, v);
                  }}
                />
              )}
              {desc && <p className="mt-0.5 text-xs text-slate-500">{desc}</p>}
            </div>
          );
        }

        if (p.type === "string" || (p.type === undefined && !p.enum)) {
          return (
            <div key={key}>
              <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">{label}</label>
              <input
                type="text"
                className="w-full rounded border border-slate-700 bg-surface px-2 py-1.5 text-slate-200"
                value={val != null ? String(val) : ""}
                onChange={(e) => {
                  const t = e.target.value;
                  setField(key, t === "" ? null : t);
                }}
                placeholder={desc}
              />
            </div>
          );
        }

        return null;
      })}
    </div>
  );
}
