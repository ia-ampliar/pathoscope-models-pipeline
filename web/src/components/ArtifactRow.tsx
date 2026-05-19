import { File, Copy, CheckCircle2 } from "lucide-react";
import { useState } from "react";
import { clsx } from "clsx";

type Artifact = {
  name: string;
  path: string;
  type?: "model" | "csv" | "tflite" | "json" | "generic";
  size?: string;
  timestamp?: string;
};

const TYPE_ICON: Record<string, React.ReactNode> = {
  model:   <File size={13} className="text-accent" />,
  tflite:  <File size={13} className="text-ok" />,
  csv:     <File size={13} className="text-text-secondary" />,
  json:    <File size={13} className="text-text-secondary" />,
  generic: <File size={13} className="text-text-muted" />,
};

export function ArtifactRow({ artifact }: { artifact: Artifact }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(artifact.path).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-elevated/60 transition-colors group">
      <span className="shrink-0">{TYPE_ICON[artifact.type ?? "generic"]}</span>
      <div className="min-w-0 flex-1">
        <p className="font-mono text-[11px] text-text-primary truncate">{artifact.name}</p>
        <p className="font-mono text-[10px] text-text-muted truncate">{artifact.path}</p>
      </div>
      {artifact.size && (
        <span className="shrink-0 font-mono text-[10px] text-text-muted">{artifact.size}</span>
      )}
      {artifact.timestamp && (
        <span className="shrink-0 font-mono text-[10px] text-text-muted hidden group-hover:inline">
          {artifact.timestamp}
        </span>
      )}
      <button
        type="button"
        onClick={copy}
        title="Copiar path"
        className={clsx(
          "shrink-0 rounded p-0.5 transition-colors",
          copied ? "text-ok" : "text-text-muted opacity-0 group-hover:opacity-100 hover:text-accent"
        )}
      >
        {copied ? <CheckCircle2 size={12} /> : <Copy size={12} />}
      </button>
    </div>
  );
}

export function ArtifactList({
  title,
  artifacts,
}: {
  title?: string;
  artifacts: Artifact[];
}) {
  if (artifacts.length === 0) return null;
  return (
    <div className="rounded-lg border border-border-subtle bg-surface/60">
      {title && (
        <div className="border-b border-border-subtle px-3 py-1.5">
          <span className="font-mono text-[10px] uppercase tracking-wider text-text-muted">{title}</span>
        </div>
      )}
      <div className="p-1">
        {artifacts.map((a) => (
          <ArtifactRow key={a.path} artifact={a} />
        ))}
      </div>
    </div>
  );
}

export type { Artifact };
