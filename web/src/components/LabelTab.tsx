import { useEffect, useMemo, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload, StepStatus, LogEntry } from "../types";
import { StageHeader } from "./StageHeader";
import { SchemaForm } from "./SchemaForm";
import { LogStream, makeLog } from "./LogStream";
import { ArtifactList } from "./ArtifactRow";

const GROUPS = [
  { label: "Fontes", keys: ["manifest_csv", "sheets_ref"] },
  { label: "Colunas", keys: ["patient_id_column", "subtype_column", "manifest_path_column", "output_csv"] },
];

type Props = { inspectorMode: boolean; onStatusChange: (s: StepStatus) => void };

export function LabelTab({ inspectorMode, onStatusChange }: Props) {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [label, setLabel] = useState<Record<string, unknown>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = (level: LogEntry["level"], msg: string) =>
    setLogs((p) => [...p.slice(-200), makeLog(level, msg)]);

  useEffect(() => {
    fetchSchema()
      .then((p) => { setSchemaPayload(p); setLabel({ ...p.defaults.LabelJobConfig }); })
      .catch((e) => setErr(String(e)));
  }, []);

  const sch = useMemo(() => schemaPayload?.schemas?.LabelJobConfig || {}, [schemaPayload]);

  const run = async () => {
    setErr(null); setLogs([]); setStatus("loading");
    onStatusChange("running");
    addLog("INFO", "Gerando label_file.csv…");
    const r = await fetch("/api/label/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!r.ok) {
      const msg = await r.text();
      setStatus("error"); setErr(msg);
      onStatusChange("error"); addLog("ERROR", msg);
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id); setStatus("running");
    addLog("INFO", `Job criado: ${j.job_id}`);
    const poll = setInterval(async () => {
      const s = await fetch(`/api/label/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as { status: string; error_message?: string };
      setStatus(d.status);
      if (d.error_message) { setErr(d.error_message); addLog("ERROR", d.error_message); }
      if (d.status === "completed") {
        onStatusChange("done"); addLog("INFO", "label_file.csv gerado com sucesso.");
        clearInterval(poll);
      }
      if (d.status === "error") { onStatusChange("error"); clearInterval(poll); }
    }, 800);
  };

  const outputPath = typeof label.output_csv === "string" && label.output_csv
    ? label.output_csv : "split/label_file.csv";

  const artifacts = status === "completed"
    ? [{ name: "label_file.csv", path: outputPath, type: "csv" as const }]
    : [];

  return (
    <div className="p-6">
      <StageHeader
        title="Labels"
        description="Cruza o manifest WSI com a planilha de rótulos para gerar split/label_file.csv (Image_path, Label)."
        status={status}
        jobId={jobId}
        onRun={run}
        runDisabled={status === "running" || status === "loading"}
      />
      <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
        <div>
          {schemaPayload && (
            <SchemaForm
              schemaRoot={sch as never}
              values={label}
              onChange={setLabel}
              groups={GROUPS}
              inspectorMode={inspectorMode}
            />
          )}
        </div>
        <div className="space-y-3">
          <LogStream entries={logs} />
          <ArtifactList title="Artefatos" artifacts={artifacts} />
          {err && <p className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 font-mono text-[11px] text-danger">{err}</p>}
        </div>
      </div>
    </div>
  );
}
