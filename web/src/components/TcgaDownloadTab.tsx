import { useEffect, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload } from "../types";

type TcgaProgress = {
  phase?: string;
  message?: string;
  current?: number;
  total?: number;
  file_name?: string;
  case_submitter_id?: string;
};

export function TcgaDownloadTab() {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState<TcgaProgress | null>(null);
  const [caseIdColumn, setCaseIdColumn] = useState("case_submitter_id");
  const [onlyOpenAccess, setOnlyOpenAccess] = useState(true);
  const [gdcToken, setGdcToken] = useState("");

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        setSchemaPayload(p);
        const d = p.defaults.TcgaDownloadJobConfig || {};
        setCaseIdColumn(String(d.case_id_column ?? "case_submitter_id"));
        setOnlyOpenAccess(Boolean(d.only_open_access ?? true));
        setGdcToken(String(d.gdc_token ?? ""));
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const run = async () => {
    setErr(null);
    setResult(null);
    setProgress(null);
    setStatus("loading");
    const defaults = schemaPayload?.defaults?.TcgaDownloadJobConfig || {};
    const payload = {
      ...defaults,
      case_id_column: caseIdColumn || "case_submitter_id",
      only_open_access: onlyOpenAccess,
      gdc_token: gdcToken.trim() || null,
      data_root: "datas",
    };
    const fd = new FormData();
    fd.append("tcga_download_json", JSON.stringify(payload));
    if (file) fd.append("file", file);
    const r = await fetch("/api/tcga/download/jobs", { method: "POST", body: fd });
    if (!r.ok) {
      setStatus("error");
      setErr(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id);
    setStatus("running");
    const poll = setInterval(async () => {
      const s = await fetch(`/api/tcga/download/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as {
        status: string;
        error_message?: string;
        result?: Record<string, unknown>;
        progress?: TcgaProgress | null;
      };
      setStatus(d.status);
      if (d.error_message) setErr(d.error_message);
      if (d.result) setResult(d.result);
      if (d.progress) setProgress(d.progress);
      if (d.status === "completed" || d.status === "error") clearInterval(poll);
    }, 1200);
  };

  const manifestPath = typeof result?.manifest_path === "string" ? result.manifest_path : null;
  const rowCount = typeof result?.row_count === "number" ? result.row_count : null;
  const message = typeof result?.message === "string" ? result.message : null;
  const total = typeof progress?.total === "number" ? progress.total : 0;
  const current = typeof progress?.current === "number" ? progress.current : 0;
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
  const isListing = progress?.phase === "listing";
  const isDownloading = progress?.phase === "downloading";

  return (
    <div className="max-w-xl rounded-xl border border-slate-800 bg-surface/80 p-4">
      <h2 className="mb-3 text-sm font-semibold text-accent">Download WSIs (TCGA / GDC)</h2>
      <p className="mb-3 text-xs text-slate-500">
        Consulta o GDC, descarrega .svs para data_root e gera o manifest. Token opcional: campo abaixo ou
        GDC_TOKEN no servidor. CSV de casos: envie ficheiro ou use o caminho em ids_csv (relativo à raiz do
        projeto).
      </p>
      <div className="space-y-3 text-sm">
        <div>
          <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">case_id_column</label>
          <input
            type="text"
            className="w-full rounded border border-slate-700 bg-surface px-2 py-1.5 text-slate-200"
            value={caseIdColumn}
            onChange={(e) => setCaseIdColumn(e.target.value)}
            placeholder="case_submitter_id"
          />
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-slate-300">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-surface"
            checked={onlyOpenAccess}
            onChange={(e) => setOnlyOpenAccess(e.target.checked)}
          />
          <span>only_open_access</span>
        </label>
        <div>
          <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">gdc_token (opcional)</label>
          <input
            type="password"
            className="w-full rounded border border-slate-700 bg-surface px-2 py-1.5 text-slate-200"
            value={gdcToken}
            onChange={(e) => setGdcToken(e.target.value)}
            placeholder="se vazio, usa GDC_TOKEN do servidor"
          />
        </div>
      </div>
      <div
        className="mt-4 cursor-pointer rounded-lg border-2 border-dashed border-slate-600 bg-panel/50 p-4 text-center text-sm text-slate-400 hover:border-accent/50"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files[0];
          if (f) setFile(f);
        }}
      >
        <input
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          id="tcga-ids-csv"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <label htmlFor="tcga-ids-csv" className="cursor-pointer">
          CSV de IDs (opcional se ids_csv já existir no servidor)
        </label>
        {file && <p className="mt-2 text-xs text-accent">{file.name}</p>}
      </div>
      <button
        type="button"
        onClick={run}
        disabled={status === "running" || status === "loading"}
        className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
      >
        Iniciar download
      </button>
      <p className="mt-2 text-xs text-slate-500">
        Job: {jobId || "—"} · {status}
      </p>
      {progress && (
        <div className="mt-3 rounded-lg border border-slate-800 bg-panel/60 p-3">
          <p className="text-xs text-slate-300">{progress.message || "A processar..."}</p>
          {isListing && (
            <div className="mt-2 h-2 w-full overflow-hidden rounded bg-slate-800">
              <div className="h-full w-1/3 animate-pulse rounded bg-accent/80" />
            </div>
          )}
          {isDownloading && (
            <>
              <div className="mt-2 h-2 w-full overflow-hidden rounded bg-slate-800">
                <div className="h-full rounded bg-accent/80 transition-all" style={{ width: `${pct}%` }} />
              </div>
              <p className="mt-1 text-[11px] text-slate-400">
                {current}/{total} ({pct}%)
                {progress.file_name ? ` · ${progress.file_name}` : ""}
                {progress.case_submitter_id ? ` · ${progress.case_submitter_id}` : ""}
              </p>
            </>
          )}
        </div>
      )}
      {manifestPath && (
        <p className="mt-2 text-xs text-slate-300">
          Manifest: <code className="text-accent">{manifestPath}</code>
          {rowCount !== null && ` (${rowCount} ficheiros)`}
        </p>
      )}
      {status === "completed" && rowCount === 0 && (
        <p className="mt-2 text-xs text-warn">
          {message ||
            "Nenhum SVS encontrado. O sistema aceita IDs curtos (TCGA-XX-XXXX) e barcodes longos (convertidos automaticamente para case_id). Se necessário, desative only_open_access e informe token."}
        </p>
      )}
      {err && <p className="mt-2 text-xs text-danger">{err}</p>}
    </div>
  );
}
