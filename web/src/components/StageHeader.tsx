import { clsx } from "clsx";
import { Play, Square, RotateCcw } from "lucide-react";

type Props = {
  title: string;
  description: string;
  status: string;
  jobId?: string | null;
  phase?: string;
  onRun: () => void;
  onAbort?: () => void;
  onReset?: () => void;
  runDisabled?: boolean;
  abortDisabled?: boolean;
};

const STATUS_BADGE: Record<string, { label: string; cls: string }> = {
  idle:      { label: "aguardando",   cls: "border-border-subtle text-text-muted" },
  loading:   { label: "iniciando…",   cls: "border-running/40 text-running animate-step-pulse" },
  running:   { label: "executando",   cls: "border-running/50 bg-running/10 text-running animate-step-pulse" },
  stopping:  { label: "parando…",     cls: "border-warn/40 text-warn" },
  completed: { label: "concluído",    cls: "border-ok/40 bg-ok/10 text-ok" },
  done:      { label: "concluído",    cls: "border-ok/40 bg-ok/10 text-ok" },
  error:     { label: "erro",         cls: "border-danger/40 bg-danger/10 text-danger" },
  stopped:   { label: "interrompido", cls: "border-warn/40 text-warn" },
};

export function StageHeader({
  title,
  description,
  status,
  jobId,
  phase,
  onRun,
  onAbort,
  onReset,
  runDisabled,
  abortDisabled,
}: Props) {
  const badge = STATUS_BADGE[status] ?? STATUS_BADGE.idle;
  const isActive = status === "running" || status === "loading";

  return (
    <div className="mb-5 flex items-start justify-between gap-4 border-b border-border-subtle pb-4">
      <div className="min-w-0">
        <div className="flex items-center gap-2.5 flex-wrap">
          <h2 className="text-base font-semibold text-text-primary">{title}</h2>
          <span
            className={clsx(
              "rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
              badge.cls
            )}
          >
            {badge.label}
          </span>
          {phase && (
            <span className="rounded border border-border-subtle px-2 py-0.5 font-mono text-[10px] text-text-muted">
              {phase}
            </span>
          )}
        </div>
        <p className="mt-1 text-sm text-text-secondary">{description}</p>
        {jobId && (
          <p className="mt-1 font-mono text-[10px] text-text-muted">
            job <span className="text-accent">{jobId}</span>
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        {onReset && !isActive && (
          <button
            type="button"
            onClick={onReset}
            className="flex items-center gap-1.5 rounded-lg border border-border-subtle px-3 py-1.5 text-sm text-text-muted hover:border-border-strong hover:text-text-secondary transition-colors"
          >
            <RotateCcw size={13} />
            Reset
          </button>
        )}
        {onAbort && isActive && (
          <button
            type="button"
            onClick={onAbort}
            disabled={abortDisabled}
            className="flex items-center gap-1.5 rounded-lg border border-danger/50 px-3 py-1.5 text-sm text-danger hover:bg-danger/10 disabled:opacity-40 transition-colors"
          >
            <Square size={13} />
            Abortar
          </button>
        )}
        <button
          type="button"
          onClick={onRun}
          disabled={runDisabled || isActive}
          className="flex items-center gap-1.5 rounded-lg bg-accent/90 px-4 py-1.5 text-sm font-medium text-base hover:bg-accent disabled:opacity-40 transition-colors"
        >
          <Play size={13} />
          {isActive ? "Executando…" : "Executar"}
        </button>
      </div>
    </div>
  );
}
