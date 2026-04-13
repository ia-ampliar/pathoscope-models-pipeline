import { type ReactNode, useEffect, useMemo, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload } from "../types";
import { SchemaForm } from "./SchemaForm";

type JobState = "idle" | "loading" | "running" | "completed" | "error";

function Card({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-slate-800 bg-surface/80 p-4">
      <h2 className="mb-3 text-sm font-semibold text-accent">{title}</h2>
      {children}
    </section>
  );
}

export function TcgaTab() {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [downloadCfg, setDownloadCfg] = useState<Record<string, unknown>>({});
  const [manifestCfg, setManifestCfg] = useState<Record<string, unknown>>({});
  const [labelsCfg, setLabelsCfg] = useState<Record<string, unknown>>({});
  const [idsFile, setIdsFile] = useState<File | null>(null);
  const [sheetsFile, setSheetsFile] = useState<File | null>(null);
  const [manifestFile, setManifestFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobState>("idle");
  const [lastOp, setLastOp] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        setSchemaPayload(p);
        setDownloadCfg({ ...(p.defaults.TcgaDownloadJobConfig as object) });
        setManifestCfg({ ...(p.defaults.TcgaManifestJobConfig as object) });
        setLabelsCfg({ ...(p.defaults.TcgaLabelsJobConfig as object) });
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const schDl = useMemo(
    () => (schemaPayload?.schemas?.TcgaDownloadJobConfig || {}) as Record<string, unknown>,
    [schemaPayload]
  );
  const schMan = useMemo(
    () => (schemaPayload?.schemas?.TcgaManifestJobConfig || {}) as Record<string, unknown>,
    [schemaPayload]
  );
  const schLab = useMemo(
    () => (schemaPayload?.schemas?.TcgaLabelsJobConfig || {}) as Record<string, unknown>,
    [schemaPayload]
  );

  const poll = async (job: string) => {
    const r = await fetch(`/api/tcga/jobs/${job}`);
    if (!r.ok) return;
    const d = (await r.json()) as {
      status: string;
      error_message?: string;
      result?: Record<string, unknown>;
    };
    setStatus(d.status as JobState);
    if (d.error_message) setErr(d.error_message);
    if (d.result) setResult(d.result);
  };

  const runTcga = async (
    operation: "download" | "manifest" | "labels",
    cfg: Record<string, unknown>,
    files: { ids?: File | null; sheets?: File | null; manifest?: File | null }
  ) => {
    setErr(null);
    setResult(null);
    setStatus("loading");
    setLastOp(operation);
    const fd = new FormData();
    fd.append("operation", operation);
    fd.append("config_json", JSON.stringify(cfg));
    if (files.ids) fd.append("ids_csv", files.ids);
    if (files.sheets) fd.append("sheets_csv", files.sheets);
    if (files.manifest) fd.append("manifest_csv", files.manifest);
    const r = await fetch("/api/tcga/jobs", { method: "POST", body: fd });
    if (!r.ok) {
      setStatus("error");
      setErr(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id);
    setStatus("running");
    const t = setInterval(async () => {
      const s = await fetch(`/api/tcga/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as {
        status: string;
        error_message?: string;
        result?: Record<string, unknown>;
      };
      setStatus(d.status as JobState);
      if (d.error_message) setErr(d.error_message);
      if (d.result) setResult(d.result);
      if (d.status === "completed" || d.status === "error") {
        clearInterval(t);
      }
    }, 1200);
  };

  return (
    <div className="space-y-6">
      <p className="text-sm text-slate-500">
        Download via <strong className="text-slate-400">API REST do GDC</strong> (não o browser). Estrutura{" "}
        <code className="text-accent">data/&lt;caso&gt;/*.svs</code>. O label file usa colunas{" "}
        <code className="text-accent">Image_path</code> e <code className="text-accent">Label</code> (compatível com
        Split).
      </p>

      <Card title="1. Download GDC (TCGA)">
        <p className="mb-3 text-xs text-slate-500">
          CSV com uma coluna de IDs de caso (padrão <code>case_submitter_id</code>). Dados controlados: defina{" "}
          <code>GDC_TOKEN</code> no ambiente do servidor.
        </p>
        {schemaPayload && (
          <SchemaForm schemaRoot={schDl as never} values={downloadCfg} onChange={setDownloadCfg} />
        )}
        <label className="mt-3 block text-xs text-slate-500">
          Ficheiro CSV de IDs
          <input
            type="file"
            accept=".csv"
            className="mt-1 block w-full text-sm text-slate-400"
            onChange={(e) => setIdsFile(e.target.files?.[0] || null)}
          />
        </label>
        <button
          type="button"
          disabled={status === "running" || status === "loading" || !idsFile}
          className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
          onClick={() => runTcga("download", downloadCfg, { ids: idsFile })}
        >
          Iniciar download
        </button>
      </Card>

      <Card title="2. Manifest (apenas disco)">
        <p className="mb-3 text-xs text-slate-500">
          Se já existir <code>data/</code> com .svs, gera <code>wsi_manifest.csv</code> sem chamar o GDC.
        </p>
        {schemaPayload && (
          <SchemaForm schemaRoot={schMan as never} values={manifestCfg} onChange={setManifestCfg} />
        )}
        <button
          type="button"
          disabled={status === "running" || status === "loading"}
          className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
          onClick={() => runTcga("manifest", manifestCfg, {})}
        >
          Gerar manifest
        </button>
      </Card>

      <Card title="3. Label file (planilha + manifest)">
        <p className="mb-3 text-xs text-slate-500">
          URL pública do Google Sheets (export CSV) <strong>ou</strong> upload de CSV local. O manifest pode ser o
          gerado acima ou um ficheiro enviado. <code>Patient ID</code> na planilha deve coincidir com a pasta sob{" "}
          <code>data/</code> (ex. TCGA-BR-4191).
        </p>
        {schemaPayload && (
          <SchemaForm schemaRoot={schLab as never} values={labelsCfg} onChange={setLabelsCfg} />
        )}
        <div className="mt-3 grid gap-2 text-xs text-slate-500">
          <label>
            Planilha (CSV) opcional se já preencheu Sheets URL
            <input
              type="file"
              accept=".csv"
              className="mt-1 block w-full text-sm"
              onChange={(e) => setSheetsFile(e.target.files?.[0] || null)}
            />
          </label>
          <label>
            Manifest CSV opcional (override do caminho em disco)
            <input
              type="file"
              accept=".csv"
              className="mt-1 block w-full text-sm"
              onChange={(e) => setManifestFile(e.target.files?.[0] || null)}
            />
          </label>
        </div>
        <button
          type="button"
          disabled={status === "running" || status === "loading"}
          className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
          onClick={() => {
            const url = (labelsCfg.sheets_url as string | undefined)?.trim();
            if (!url && !sheetsFile) {
              setErr("Indique sheets_url ou envie um CSV de planilha.");
              return;
            }
            runTcga("labels", labelsCfg, { sheets: sheetsFile, manifest: manifestFile });
          }}
        >
          Gerar label_file.csv
        </button>
      </Card>

      <div className="rounded-lg border border-slate-800 bg-panel/40 p-3 font-mono text-xs text-slate-400">
        <div>Job: {jobId ?? "—"} · op: {lastOp ?? "—"} · {status}</div>
        {err && <div className="mt-2 text-danger">{err}</div>}
        {result && (
          <pre className="mt-2 max-h-48 overflow-auto text-[11px]">{JSON.stringify(result, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
