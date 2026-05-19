import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";
import { clsx } from "clsx";

export type MetricPoint = {
  idx: number;
  stage: string;
  epoch: number;
  loss?: number;
  val_loss?: number;
  acc?: number;
  val_acc?: number;
  warn?: boolean;
};

type Phase = "baseline" | "fine_tune" | "qat";

const PHASE_COLORS: Record<Phase, { train: string; val: string }> = {
  baseline:  { train: "#2DD4D4", val: "#818cf8" },
  fine_tune: { train: "#60A5FA", val: "#f472b6" },
  qat:       { train: "#4ADE80", val: "#facc15" },
};

const PHASE_LABELS: Record<Phase, string> = {
  baseline: "Baseline", fine_tune: "Fine-tune", qat: "QAT",
};

type Props = {
  data: MetricPoint[];
  pulse?: boolean;
};

function getPhase(stage: string): Phase {
  if (stage.startsWith("fine") || stage === "fine_tune") return "fine_tune";
  if (stage === "qat") return "qat";
  return "baseline";
}

export function MetricsPanel({ data, pulse }: Props) {
  const phases = useMemo(() => {
    const seen = new Set<Phase>();
    data.forEach((d) => seen.add(getPhase(d.stage)));
    return Array.from(seen);
  }, [data]);

  const [visiblePhases, setVisiblePhases] = useState<Set<Phase>>(new Set(["baseline", "fine_tune", "qat"]));

  const toggle = (p: Phase) => {
    setVisiblePhases((prev) => {
      const next = new Set(prev);
      next.has(p) ? next.delete(p) : next.add(p);
      return next;
    });
  };

  const chartData = useMemo(
    () => data.filter((d) => visiblePhases.has(getPhase(d.stage))),
    [data, visiblePhases]
  );

  const tooltipStyle = {
    backgroundColor: "#1A242B",
    border: "1px solid #1F2F38",
    borderRadius: 6,
    fontSize: 11,
    fontFamily: "JetBrains Mono, monospace",
  };

  const axisStyle = { fontSize: 10, fontFamily: "JetBrains Mono, monospace", fill: "#5E7378" };

  return (
    <div className={clsx("rounded-lg border border-border-subtle bg-surface/60 p-3", pulse && "pulse-glow-green")}>
      {/* Phase toggles */}
      {phases.length > 0 && (
        <div className="mb-3 flex items-center gap-2 flex-wrap">
          <span className="font-mono text-[10px] uppercase tracking-wider text-text-muted mr-1">Fase</span>
          {(["baseline", "fine_tune", "qat"] as Phase[]).filter(p => phases.includes(p)).map((p) => {
            const c = PHASE_COLORS[p];
            const on = visiblePhases.has(p);
            return (
              <button
                key={p}
                type="button"
                onClick={() => toggle(p)}
                className={clsx(
                  "flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-[10px] transition-colors",
                  on ? "border-border-strong bg-elevated/60 text-text-secondary" : "border-border-subtle text-text-muted opacity-50"
                )}
              >
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: c.train }} />
                {PHASE_LABELS[p]}
              </button>
            );
          })}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {/* Loss */}
        <div>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-muted">Loss</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2F38" />
              <XAxis dataKey="idx" tick={axisStyle} />
              <YAxis tick={axisStyle} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#93B3B7" }} />
              {(["baseline", "fine_tune", "qat"] as Phase[]).map((p) => {
                const c = PHASE_COLORS[p];
                return [
                  <Line key={`${p}-loss`} type="monotone" dataKey="loss"
                    data={data.filter(d => getPhase(d.stage) === p && visiblePhases.has(p))}
                    stroke={c.train} strokeWidth={1.5} dot={false} name={`${PHASE_LABELS[p]} train`} />,
                  <Line key={`${p}-val_loss`} type="monotone" dataKey="val_loss"
                    data={data.filter(d => getPhase(d.stage) === p && visiblePhases.has(p))}
                    stroke={c.val} strokeWidth={1.5} dot={false} strokeDasharray="4 2" name={`${PHASE_LABELS[p]} val`} />,
                ];
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Accuracy */}
        <div>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-muted">Accuracy</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2F38" />
              <XAxis dataKey="idx" tick={axisStyle} />
              <YAxis domain={[0, 1]} tick={axisStyle} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#93B3B7" }} />
              {(["baseline", "fine_tune", "qat"] as Phase[]).map((p) => {
                const c = PHASE_COLORS[p];
                return [
                  <Line key={`${p}-acc`} type="monotone" dataKey="acc"
                    data={data.filter(d => getPhase(d.stage) === p && visiblePhases.has(p))}
                    stroke={c.train} strokeWidth={1.5} dot={false} name={`${PHASE_LABELS[p]} train`} />,
                  <Line key={`${p}-val_acc`} type="monotone" dataKey="val_acc"
                    data={data.filter(d => getPhase(d.stage) === p && visiblePhases.has(p))}
                    stroke={c.val} strokeWidth={1.5} dot={false} strokeDasharray="4 2" name={`${PHASE_LABELS[p]} val`} />,
                ];
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {data.length === 0 && (
        <div className="flex h-32 items-center justify-center">
          <p className="font-mono text-[11px] text-text-muted italic">Aguardando dados de treino…</p>
        </div>
      )}
    </div>
  );
}
