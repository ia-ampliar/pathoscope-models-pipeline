import { useEffect, useState } from "react";
import { Upload } from "lucide-react";
import { fetchSchema } from "../api";
import { StageHeader } from "./StageHeader";
import { LogStream, makeLog } from "./LogStream";
import { ArtifactList } from "./ArtifactRow";
import { ParamCard } from "./ParamCard";
import type { StepStatus, LogEntry } from "../types";

type TcgaProgress = {
  phase?: string; message?: string;
  current?: number; total?: number;
  file_name?: string; case_submitter_id?: string;
};

type Props = { inspectorMode: boolean; onStatusChange: (s: StepStatus) => void };

export function TcgaDownloadTab({ inspectorMode, onStatusChange }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState<TcgaProgress | null>(null);
  const [caseIdColumn, setCaseIdColumn] = useState("case_submitter_id");
  const [onlyOpenAccess, setOnlyOpenAccess] = useState(true);
  const [gdcToken, setGdcToken] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = (level: LogEntry["level"], msg: string) =>
    setLogs((p) => [...p.slice(-200), makeLog(level, msg)]);

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        const d = p.defaults.TcgaDownloadJobConfig || {};
        setCaseIdColumn(String(d.case_id_column ?? "case_submitter_id"));
        setOnlyOpenAccess(Boolean(d.only_open_access ?? true));
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const run = async () => {
    setErr(null); setResult(null); setProgress(null);
    setLogs([]); setStatus("loading");
    onStatusChange("running");
    addLog("INFO", "Iniciando download TCGA…");

    const payload = {
      case_id_column: caseIdColumn || "case_submitter_id",
      only_open_access: onlyOpenAccess,
      gdc_token: gdcToken.trim() || null,
      data_root: "data",
      wsi_manifest_csv: "wsi_manifest.csv",
    };
    const fd = new FormData();
    fd.append("tcga_download_json", JSON.stringify(payload));
    if (file) fd.append("file", file);

    const r = await fetch("/api/tcga/download/jobs", { method: "POST", body: fd });
    if (!r.ok) {
      const msg = await r.text();
      setStatus("error"); setErr(msg);
      onStatusChange("error");
      addLog("ERROR", msg);
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id); setStatus("running");
    addLog("INFO", `Job criado: ${j.job_id}`);

    const poll = setInterval(async () => {
      const s = await fetch(`/api/tcga/download/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as {
        status: string; error_message?: string;
        result?: Record<string, unknown>; progress?: TcgaProgress | null;
      };
      setStatus(d.status);
      if (d.progress) {
        setProgress(d.progress);
        if (d.progress.message) addLog("INFO", d.progress.message);
      }
      if (d.error_message) { setErr(d.error_message); addLog("ERROR", d.error_message); }
      if (d.result) { setResult(d.result); addLog("INFO", "Download concluído."); }
      if (d.status === "completed") { onStatusChange("done"); clearInterval(poll); }
      if (d.status === "error") { onStatusChange("error"); clearInterval(poll); }
    }, 1200);
  };

  const manifestPath = typeof result?.manifest_path === "string" ? result.manifest_path : null;
  const rowCount = typeof result?.row_count === "number" ? result.row_count : null;
  const total = typeof progress?.total === "number" ? progress.total : 0;
  const current = typeof progress?.current === "number" ? progress.current : 0;
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
  const isListing = progress?.phase === "listing";
  const isDownloading = progress?.phase === "downloading";

  const artifacts = manifestPath
    ? [{ name: "wsi_manifest.csv", path: manifestPath, type: "csv" as const,
         size: rowCount != null ? `${rowCount} ficheiros` : undefined }]
    : [];

  return (
    <div className="p-6">
      <StageHeader
        title="Download TCGA"
        description="Consulta o GDC, descarrega .svs para data/ e gera wsi_manifest.csv com os caminhos e IDs."
        status={status}
        jobId={jobId}
        onRun={run}
        runDisabled={status === "running" || status === "loading"}
      />

      <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
        {/* Params */}
        <div className="space-y-2">
          <ParamCard paramKey="case_id_column" label="case id column" description="Nome da coluna com os IDs de caso no CSV." inspectorMode={inspectorMode}>
            <input type="text" className="w-full rounded border border-border-strong bg-elevated px-2 py-1.5 font-mono text-sm text-text-primary focus:border-accent/60 focus:outline-none"
              value={caseIdColumn} onChange={(e) => setCaseIdColumn(e.target.value)} placeholder="case_submitter_id" />
          </ParamCard>

          <ParamCard paramKey="only_open_access" label="only open access" description="Filtra apenas dados de acesso aberto no GDC." inspectorMode={inspectorMode}>
            <label className="flex cursor-pointer items-center gap-2 py-0.5">
              <input type="checkbox" className="h-3.5 w-3.5 rounded accent-accent"
                checked={onlyOpenAccess} onChange={(e) => setOnlyOpenAccess(e.target.checked)} />
              <span className="text-sm text-text-primary">{onlyOpenAccess ? "Ligado" : "Desligado"}</span>
            </label>
          </ParamCard>

          {(inspectorMode || !onlyOpenAccess) && (
            <ParamCard paramKey="gdc_token" label="gdc token" description="Token para dados controlados. Opcional se GDC_TOKEN estiver no servidor." inspectorMode={inspectorMode}>
              <input type="password" className="w-full rounded border border-border-strong bg-elevated px-2 py-1.5 font-mono text-sm text-text-primary focus:border-accent/60 focus:outline-none"
                value={gdcToken} onChange={(e) => setGdcToken(e.target.value)} placeholder="deixe vazio para usar GDC_TOKEN do servidor" />
            </ParamCard>
          )}

          {/* File upload */}
          <div
            className="cursor-pointer rounded-lg border-2 border-dashed border-border-strong bg-elevated/20 p-4 text-center transition-colors hover:border-accent/40"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}
          >
            <input type="file" accept=".csv,text/csv" className="hidden" id="tcga-ids-csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <label htmlFor="tcga-ids-csv" className="flex flex-col items-center gap-1 cursor-pointer">
              <Upload size={16} className="text-text-muted" />
              <span className="text-sm text-text-secondary">CSV de IDs de caso</span>
              <span className="text-[11px] text-text-muted">opcional se ids_csv já existir no servidor</span>
            </label>
            {file && <p className="mt-2 font-mono text-[11px] text-accent">{file.name}</p>}
          </div>
        </div>

        {/* Right: Progress + Logs + Artifacts */}
        <div className="space-y-3">
          {/* Progress */}
          {progress && (
            <div className="rounded-lg border border-border-subtle bg-surface/60 p-3">
              <p className="font-mono text-[11px] text-text-secondary mb-2">{progress.message || "Processando…"}</p>
              {isListing && (
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div className="h-full w-1/3 rounded-full bg-accent/70 animate-step-pulse" />
                </div>
              )}
              {isDownloading && (
                <>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                    <div className="h-full rounded-full bg-accent/80 transition-all duration-300" style={{ width: `${pct}%` }} />
                  </div>
                  <p className="mt-1 font-mono text-[10px] text-text-muted">
                    {current}/{total} ({pct}%)
                    {progress.file_name ? ` · ${progress.file_name}` : ""}
                    {progress.case_submitter_id ? ` · ${progress.case_submitter_id}` : ""}
                  </p>
                </>
              )}
            </div>
          )}

          <LogStream entries={logs} />
          <ArtifactList title="Artefatos" artifacts={artifacts} />
          {err && <p className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 font-mono text-[11px] text-danger">{err}</p>}
        </div>
      </div>
    </div>
  );
}
