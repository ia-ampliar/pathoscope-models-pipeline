import { useEffect, useRef, useState } from "react";
import { Search, Lock, Unlock } from "lucide-react";
import { clsx } from "clsx";
import type { LogEntry, LogLevel } from "../types";

type Props = {
  entries: LogEntry[];
  maxHeight?: string;
};

const LEVEL_COLORS: Record<LogLevel, string> = {
  INFO:  "text-text-secondary",
  DEBUG: "text-text-muted",
  WARN:  "text-warn",
  ERROR: "text-danger",
};

const LEVEL_LABELS: Record<LogLevel, string> = {
  INFO: "INFO ", DEBUG: "DBG  ", WARN: "WARN ", ERROR: "ERR  ",
};

export function LogStream({ entries, maxHeight = "220px" }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [locked, setLocked] = useState(true);
  const [filter, setFilter] = useState<LogLevel | "ALL">("ALL");
  const [search, setSearch] = useState("");

  const filtered = entries.filter((e) => {
    if (filter !== "ALL" && e.level !== filter) return false;
    if (search && !e.message.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  useEffect(() => {
    if (locked && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [filtered.length, locked]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    if (!atBottom && locked) setLocked(false);
    if (atBottom && !locked) setLocked(true);
  };

  return (
    <div className="rounded-lg border border-border-subtle bg-surface/60 overflow-hidden">
      {/* toolbar */}
      <div className="flex items-center gap-2 border-b border-border-subtle bg-elevated/40 px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-text-muted mr-1">Log</span>
        {(["ALL", "INFO", "WARN", "ERROR"] as const).map((l) => (
          <button
            key={l}
            type="button"
            onClick={() => setFilter(l)}
            className={clsx(
              "rounded px-1.5 py-0.5 font-mono text-[10px] transition-colors",
              filter === l
                ? "bg-accent/20 text-accent"
                : "text-text-muted hover:text-text-secondary"
            )}
          >
            {l}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-1.5">
          <div className="relative">
            <Search size={10} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="buscar..."
              className="h-5 w-28 rounded border border-border-subtle bg-base pl-6 pr-2 font-mono text-[10px] text-text-secondary focus:outline-none focus:border-accent/50"
            />
          </div>
          <button
            type="button"
            onClick={() => setLocked((v) => !v)}
            title={locked ? "Auto-scroll ligado" : "Auto-scroll desligado"}
            className={clsx(
              "rounded p-0.5 transition-colors",
              locked ? "text-accent" : "text-text-muted hover:text-text-secondary"
            )}
          >
            {locked ? <Lock size={11} /> : <Unlock size={11} />}
          </button>
        </div>
      </div>

      {/* log lines */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{ maxHeight }}
        className="overflow-y-auto px-3 py-2 space-y-0.5"
      >
        {filtered.length === 0 && (
          <p className="font-mono text-[11px] text-text-muted italic py-2">Nenhum log ainda.</p>
        )}
        {filtered.map((e) => (
          <div key={e.id} className="flex gap-2 font-mono text-[11px] leading-relaxed">
            <span className="shrink-0 text-text-muted tabular-nums">
              {e.time.toLocaleTimeString("pt-BR", { hour12: false })}
            </span>
            <span className={clsx("shrink-0 tabular-nums", LEVEL_COLORS[e.level])}>
              {LEVEL_LABELS[e.level]}
            </span>
            <span className="text-text-secondary break-all">{e.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

let _logId = 0;
export function makeLog(level: LogLevel, message: string): LogEntry {
  return { id: ++_logId, time: new Date(), level, message };
}
