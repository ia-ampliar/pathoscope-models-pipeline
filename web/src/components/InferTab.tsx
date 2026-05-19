import { useEffect, useMemo, useState } from "react";
import { Upload } from "lucide-react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload, StepStatus, LogEntry } from "../types";
import { StageHeader } from "./StageHeader";
import { SchemaForm } from "./SchemaForm";
import { LogStream, makeLog } from "./LogStream";
import { ArtifactList } from "./ArtifactRow";

const GROUPS = [
  { label: "Modelo", keys: ["tflite_path", "threshold"] },
  { label: "Patches", keys: ["patch_multiplier", "image_path", "output_dir"] },
];

type Props = { inspectorMode: boolean; onStatusChange: (s: StepStatus) => void };

export function InferTab({ inspectorMode, onStatusChange }: Props) {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [infer, setInfer] = useState<Record<string, unknown>>({});
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [beam, setBeam] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = (level: LogEntry["level"], msg: string) =>
    setLogs((p) => [...p.slice(-200), makeLog(level, msg)]);

  useEffect(() => {
    fetchSchema()
      .then((p) => { setSchemaPayload(p); setInfer({ ...p.defaults.InferenceJobConfig }); })
      .catch((e) => setErr(String(e)));
  }, []);

  const sch = useMemo(() => schemaPayload?.schemas?.InferenceJobConfig || {}, [schemaPayload]);

  const run = async () => {
    setErr(null); setResult(null); setLogs([]); setBeam(true);
    setStatus("loading");
    onStatusChange("running");
    addLog("INFO", "Iniciando inferência WSI…");
    const fd = new FormData();
    fd.append("inference_json", JSON.stringify(infer));
    if (file) fd.append("file", file);
    const r = await fetch("/api/inference/jobs", { method: "POST", body: fd });
    if (!r.ok) {
      const msg = await r.text();
      setStatus("error"); setErr(msg); setBeam(false);
      onStatusChange("error"); addLog("ERROR", msg);
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id); setStatus("running");
    addLog("INFO", `Job criado: ${j.job_id}`);
    const poll = setInterval(async () => {
      const s = await fetch(`/api/inference/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as {
        status: string;
        error_message?: string;
        result?: Record<string, unknown>;
      };
      setStatus(d.status);
      if (d.error_message) { setErr(d.error_message); addLog("ERROR", d.error_message); }
      if (d.result) { setResult(d.result); addLog("INFO", "Inferência concluída."); }
      if (d.status === "completed") {
        onStatusChange("done"); setBeam(false); clearInterval(poll);
      }
      if (d.status === "error") { onStatusChange("error"); setBeam(false); clearInterval(poll); }
    }, 1500);
  };

  const metrics = result?.metrics as Record<string, unknown> | undefined;
  const rawArtifacts = result?.artifacts as { files?: { path: string; suffix: string }[] } | undefined;
  const jpgFiles = rawArtifacts?.files?.filter((f) => f.suffix === ".jpg") || [];
  const artifactList = rawArtifacts?.files?.map((f) => ({
    name: f.path.split("/").pop() || f.path,
    path: f.path,
    type: "generic" as const,
  })) || [];

  return (
    <div className="p-6">
      <StageHeader
        title="Inferência"
        description="Inferência por patch em WSI .svs; gera heatmap de confiança e arquivo NPZ com resultados por patch."
        status={status}
        jobId={jobId}
        onRun={run}
        runDisabled={status === "running" || status === "loading"}
      />

      <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
        {/* Params + upload */}
        <div className="space-y-3">
          {schemaPayload && (
            <SchemaForm
              schemaRoot={sch as never}
              values={infer}
              onChange={setInfer}
              groups={GROUPS}
              inspectorMode={inspectorMode}
            />
          )}

          <div
            className="cursor-pointer rounded-lg border-2 border-dashed border-border-strong bg-elevated/20 p-4 text-center transition-colors hover:border-accent/40"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}
          >
            <input
              type="file"
              accept=".svs"
              className="hidden"
              id="infer-svs"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <label htmlFor="infer-svs" className="flex flex-col items-center gap-1 cursor-pointer">
              <Upload size={16} className="text-text-muted" />
              <span className="text-sm text-text-secondary">Arquivo WSI (.svs)</span>
              <span className="text-[11px] text-text-muted">arraste ou clique para escolher</span>
            </label>
            {file && <p className="mt-2 font-mono text-[11px] text-accent">{file.name}</p>}
          </div>
        </div>

        {/* Results */}
        <div className="space-y-3">
          {beam && (
            <div className="relative h-1 overflow-hidden rounded-full bg-elevated">
              <div className="absolute h-full w-1/3 rounded-full bg-gradient-to-r from-transparent via-accent to-transparent animate-beam" />
            </div>
          )}
          <LogStream entries={logs} />
          <ArtifactList title="Artefatos" artifacts={artifactList} />
          {metrics && (
            <pre className="rounded-md border border-border-subtle bg-surface/60 p-3 font-mono text-[10px] text-text-secondary overflow-auto max-h-40">
              {JSON.stringify(metrics, null, 2)}
            </pre>
          )}
          {jpgFiles.map((f) => (
            <img
              key={f.path}
              src={`/api/artifacts/${jobId}/${f.path.split("/").map(encodeURIComponent).join("/")}`}
              alt={f.path}
              className="max-h-64 w-full rounded border border-border-subtle object-contain"
            />
          ))}
          {err && (
            <p className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 font-mono text-[11px] text-danger">
              {err}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
