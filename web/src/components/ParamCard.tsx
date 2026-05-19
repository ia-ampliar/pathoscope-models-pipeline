import { useState } from "react";
import { Info, ChevronDown, ChevronUp, AlertTriangle } from "lucide-react";
import { clsx } from "clsx";
import { getParamHelp } from "../paramHelp";

type Props = {
  paramKey: string;
  label: string;
  description?: string;
  risky?: boolean;
  inspectorMode?: boolean;
  children: React.ReactNode;
};

export function ParamCard({ paramKey, label, description, risky, children }: Props) {
  const [open, setOpen] = useState(false);
  const help = getParamHelp(paramKey);

  return (
    <div
      className={clsx(
        "rounded-lg border bg-elevated/40 px-3 py-2.5 transition-colors",
        risky ? "border-warn/40" : "border-border-subtle hover:border-border-strong"
      )}
    >
      {/* Label row */}
      <div className="mb-1.5 flex items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
          {label}
        </span>
        {risky && (
          <span className="flex items-center gap-1 rounded border border-warn/50 bg-warn/10 px-1.5 py-0.5 font-mono text-[10px] text-warn">
            <AlertTriangle size={10} />
            risco
          </span>
        )}
        {help && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? "Fechar ajuda" : "Ver ajuda"}
            className="ml-auto rounded p-0.5 text-text-muted hover:text-accent transition-colors"
          >
            <Info size={13} />
          </button>
        )}
      </div>

      {/* Input slot */}
      {children}

      {/* Short description */}
      {description && (
        <p className="mt-1 text-[11px] text-text-muted leading-relaxed">{description}</p>
      )}

      {/* Rich help panel */}
      {help && open && (
        <div className="mt-2.5 space-y-2 rounded-md border border-border-subtle bg-surface/80 p-3 text-[11px] leading-relaxed text-text-secondary">
          <Section title="O que é" body={help.what} />
          <Section title="Como escolher" body={help.how} />
          <Section title="Trade-offs" body={help.tradeoffs} />
          <Section title="De onde vem" body={help.source} />
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="mt-1 flex items-center gap-1 text-text-muted hover:text-accent transition-colors"
          >
            <ChevronUp size={11} /> fechar
          </button>
        </div>
      )}

      {/* Expand prompt when help exists but panel is closed */}
      {help && !open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="mt-1 flex items-center gap-1 text-[10px] text-text-muted hover:text-accent transition-colors"
        >
          <ChevronDown size={10} /> ver detalhes
        </button>
      )}
    </div>
  );
}

function Section({ title, body }: { title: string; body: string }) {
  return (
    <div>
      <span className="font-mono text-[10px] uppercase tracking-wider text-accent">{title}</span>
      <p className="mt-0.5 text-text-secondary">{body}</p>
    </div>
  );
}
