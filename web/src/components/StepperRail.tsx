import { clsx } from "clsx";
import { Lock, ArrowRight, Loader2, CheckCircle2, AlertCircle, Cpu, Server } from "lucide-react";
import type { StepId, StepStatus } from "../types";

export type StepConfig = {
  id: StepId;
  num: number;
  label: string;
  shortDesc: string;
  requires: StepId | null;
};

export const PIPELINE_STEPS: StepConfig[] = [
  { id: "tcga",   num: 1, label: "Download TCGA", shortDesc: "Baixar WSIs do GDC",    requires: null },
  { id: "label",  num: 2, label: "Labels",         shortDesc: "Gerar label_file.csv", requires: "tcga" },
  { id: "tiling", num: 3, label: "Tiling",          shortDesc: "Tiles por classe",     requires: "label" },
  { id: "split",  num: 4, label: "Split",           shortDesc: "Train/val/test CSVs",  requires: "tiling" },
  { id: "train",  num: 5, label: "Treino",          shortDesc: "MobileNetV2 + QAT",    requires: "split" },
  { id: "infer",  num: 6, label: "Inferência",      shortDesc: "Heatmap WSI",          requires: "train" },
];

type Props = {
  statuses: Record<StepId, StepStatus>;
  active: StepId;
  onSelect: (id: StepId) => void;
  inspectorMode: boolean;
  onToggleInspector: () => void;
};

const STATUS_ICON: Record<StepStatus, React.ReactNode> = {
  locked:    <Lock size={11} className="text-text-muted" />,
  available: <ArrowRight size={11} className="text-accent" />,
  running:   <Loader2 size={11} className="text-running animate-spin" />,
  done:      <CheckCircle2 size={11} className="text-ok" />,
  error:     <AlertCircle size={11} className="text-danger" />,
};

const STEP_BORDER: Record<StepStatus, string> = {
  locked:    "border-border-subtle",
  available: "border-teal-deep",
  running:   "border-running/60",
  done:      "border-ok/40",
  error:     "border-danger/40",
};

const STEP_NUM: Record<StepStatus, string> = {
  locked:    "bg-elevated text-text-muted",
  available: "bg-teal-deep/40 text-accent",
  running:   "bg-running/20 text-running",
  done:      "bg-ok/20 text-ok",
  error:     "bg-danger/20 text-danger",
};

export function StepperRail({ statuses, active, onSelect, inspectorMode, onToggleInspector }: Props) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border-subtle bg-surface">
      {/* Wordmark */}
      <div className="border-b border-border-subtle px-4 py-4">
        <div className="font-mono text-sm font-semibold tracking-tight text-accent">Pathoscope</div>
        <div className="mt-0.5 font-mono text-[10px] text-text-muted">Control Room</div>
      </div>

      {/* System status */}
      <div className="border-b border-border-subtle px-4 py-2 flex items-center gap-3 flex-wrap">
        <span className="flex items-center gap-1 font-mono text-[10px] text-text-muted">
          <Server size={10} /> FastAPI
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-ok ml-1" />
        </span>
        <span className="flex items-center gap-1 font-mono text-[10px] text-text-muted">
          <Cpu size={10} /> GPU
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-text-muted ml-1" />
        </span>
      </div>

      {/* Pipeline steps */}
      <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-1">
        {PIPELINE_STEPS.map((step) => {
          const status = statuses[step.id];
          const isActive = step.id === active;

          return (
            <button
              key={step.id}
              type="button"
              onClick={() => onSelect(step.id)}
              className={clsx(
                "w-full rounded-lg border px-3 py-2.5 text-left transition-all cursor-pointer",
                STEP_BORDER[status],
                isActive ? "bg-elevated/80" : "hover:bg-elevated/40",
              )}
            >
              <div className="flex items-center gap-2">
                <span
                  className={clsx(
                    "flex h-5 w-5 shrink-0 items-center justify-center rounded font-mono text-[10px] font-semibold",
                    STEP_NUM[status]
                  )}
                >
                  {step.num}
                </span>
                <span className="flex-1 text-xs font-medium text-text-primary">
                  {step.label}
                </span>
                <span className="shrink-0">{STATUS_ICON[status]}</span>
              </div>

              <p className="mt-0.5 pl-7 font-mono text-[10px] text-text-muted">
                {step.shortDesc}
              </p>

              {step.requires && status !== "done" && status !== "running" && (
                <p className="mt-0.5 pl-7 font-mono text-[10px] text-text-muted opacity-60">
                  ↑ depende de {PIPELINE_STEPS.find(s => s.id === step.requires)?.label}
                </p>
              )}

              {status === "running" && (
                <p className="mt-0.5 pl-7 font-mono text-[10px] text-running animate-step-pulse">
                  em execução…
                </p>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer: Inspector toggle */}
      <div className="border-t border-border-subtle px-4 py-3">
        <button
          type="button"
          onClick={onToggleInspector}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 hover:bg-elevated/40 transition-colors"
        >
          <span className="font-mono text-[10px] text-text-muted uppercase tracking-wider">
            Modo
          </span>
          <span
            className={clsx(
              "rounded border px-2 py-0.5 font-mono text-[10px]",
              inspectorMode
                ? "border-accent/40 bg-accent/10 text-accent"
                : "border-border-subtle text-text-muted"
            )}
          >
            {inspectorMode ? "Inspetor" : "Operador"}
          </span>
        </button>
        <p className="mt-1 px-2 font-mono text-[10px] text-text-muted">
          {inspectorMode ? "Todos os parâmetros visíveis" : "Apenas parâmetros essenciais"}
        </p>
      </div>
    </aside>
  );
}
