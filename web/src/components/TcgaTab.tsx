import { useEffect, useMemo, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload } from "../types";
import { SchemaForm } from "./SchemaForm";

function repoFileUrl(rel: string) {
  return `/api/files/repo/${rel.split("/").map(encodeURIComponent).join("/")}`;
}

export function TcgaTab() {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [downloadCfg, setDownloadCfg] = useState<Record<string, unknown>>({});
  const [manifestCfg, setManifestCfg] = useState<Record<string, unknown>>({});
  const [labelsCfg, setLabelsCfg] = useState<Record<string, unknown>>({});
  const [idsFile, setIdsFile] = useState<File | null>(null);
  const [manifestUpload, setManifestUpload] = useState<File | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, { kind: string; status: string; result?: Record<string, unknown> }>>(
    {}
  );

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        setSchemaPayload(p);
        const d = p.defaults as Record<string, Record<string, unknown>>;
        setDownloadCfg({ ...(d.TcgaDownloadJobConfig || {}) });
        setManifestCfg({ ...(d.TcgaManifestDiskJobConfig || {}) });
        setLabelsCfg({ ...(d.TcgaLabelsJobConfig || {}) });
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const schDl = useMemo(() => schemaPayload?.schemas?.TcgaDownloadJobConfig || {}, [schemaPayload]);
  const schMan = useMemo(() => schemaPayload?.schemas?.TcgaManifestDiskJobConfig || {}, [schemaPayload]);
  const schLab = useMemo(() => schemaPayload?.schemas?.TcgaLabelsJobConfig || {}, [schemaPayload]);

  const pollJob = (jobId: string) => {
    const t = setInterval(async () => {
      const r = await fetch(`/api/tcga/jobs/${jobId}`);
      if (!r.ok) return;
      const d = (await r.json()) as {
        status: string;
        result?: Record<string, unknown>;
        error_message?: string;
      };
      setJobs((prev) => ({
        ...prev,
        [jobId]: { ...prev[jobId], status: d.status, result: d.result },
      }));
      if (d.error_message) setErr(d.error_message);
      if (d.status === "completed" || d.status === "error") clearInterval(t);
    }, 1200);
  };

  const runDownload = async () => {
    setErr(null);
    if (!idsFile) {
      setErr("Envie o CSV com os case IDs (coluna case_submitter_id ou equivalente).");
      return;
    }
    const fd = new FormData();
    fd.append("ids_csv", idsFile);
    fd.append("config_json", JSON.stringify(downloadCfg));
    const r = await fetch("/api/tcga/download", { method: "POST", body: fd });
    if (!r.ok) {
      setErr(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string; kind: string };
    setJobs((prev) => ({ ...prev, [j.job_id]: { kind: j.kind, status: "running" } }));
    pollJob(j.job_id);
  };

  const runManifest = async () => {
    setErr(null);
    const r = await fetch("/api/tcga/manifest-from-disk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(manifestCfg),
    });
    if (!r.ok) {
      setErr(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string; kind: string };
    setJobs((prev) => ({ ...prev, [j.job_id]: { kind: j.kind, status: "running" } }));
    pollJob(j.job_id);
  };

  const runLabels = async () => {
    setErr(null);
    const fd = new FormData();
    fd.append("config_json", JSON.stringify(labelsCfg));
    if (manifestUpload) fd.append("manifest_file", manifestUpload);
    const r = await fetch("/api/tcga/labels", { method: "POST", body: fd });
    if (!r.ok) {
      setErr(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string; kind: string };
    setJobs((prev) => ({ ...prev, [j.job_id]: { kind: j.kind, status: "running" } }));
    pollJob(j.job_id);
  };

  return (
    <div className="space-y-10">
      <p className="text-sm text-slate-500">
        Usa a API GDC (mesmos dados que o portal). Defina <code className="text-accent">GDC_TOKEN</code> no
        servidor para dados controlados.
      </p>

      <section className="rounded-xl border border-slate-800 bg-surface/80 p-4">
        <h2 className="mb-2 text-sm font-semibold text-accent">1. Download WSI (TCGA / GDC)</h2>
        <p className="mb-3 text-xs text-slate-500">
          CSV com IDs de caso (ex.: coluna <code>case_submitter_id</code>). Grava em{" "}
          <code>data/&lt;caso&gt;/…*.svs</code> e gera o manifest.
        </p>
        {schemaPayload && (
          <SchemaForm schemaRoot={schDl as never} values={downloadCfg} onChange={setDownloadCfg} />
        )}
        <div className="mt-3">
          <label className="cursor-pointer text-xs text-slate-400">
            <input type="file" accept=".csv" className="sr-only" onChange={(e) => setIdsFile(e.target.files?.[0] || null)} />
            <span className="rounded border border-slate-600 px-2 py-1 hover:border-accent/50">
              {idsFile ? idsFile.name : "Escolher CSV de IDs"}
            </span>
          </label>
        </div>
        <button
          type="button"
          onClick={runDownload}
          className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950"
        >
          Iniciar download
        </button>
      </section>

      <section className="rounded-xl border border-slate-800 bg-surface/80 p-4">
        <h2 className="mb-2 text-sm font-semibold text-accent">2. Manifest a partir de disco</h2>
        <p className="mb-3 text-xs text-slate-500">
          Se a pasta <code>data/</code> já existir com <code>.svs</code>, gera o CSV de caminhos relativos (sem
          chamar o GDC).
        </p>
        {schemaPayload && (
          <SchemaForm schemaRoot={schMan as never} values={manifestCfg} onChange={setManifestCfg} />
        )}
        <button
          type="button"
          onClick={runManifest}
          className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950"
        >
          Gerar manifest
        </button>
      </section>

      <section className="rounded-xl border border-slate-800 bg-surface/80 p-4">
        <h2 className="mb-2 text-sm font-semibold text-accent">3. Label file (planilha + manifest)</h2>
        <p className="mb-3 text-xs text-slate-500">
          URL da planilha Google (export CSV público) ou caminho no servidor; cruza com{" "}
          <code>Patient ID</code> e <code>Subtype</code>. Opcionalmente envie um manifest CSV aqui; senão usa o
          caminho indicado no formulário (ex.: <code>wsi_manifest.csv</code>).
        </p>
        {schemaPayload && (
          <SchemaForm schemaRoot={schLab as never} values={labelsCfg} onChange={setLabelsCfg} />
        )}
        <div className="mt-3">
          <label className="cursor-pointer text-xs text-slate-400">
            <input
              type="file"
              accept=".csv"
              className="sr-only"
              onChange={(e) => setManifestUpload(e.target.files?.[0] || null)}
            />
            <span className="rounded border border-slate-600 px-2 py-1 hover:border-accent/50">
              {manifestUpload ? manifestUpload.name : "Opcional: upload manifest.csv"}
            </span>
          </label>
        </div>
        <button
          type="button"
          onClick={runLabels}
          className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950"
        >
          Gerar label_file.csv
        </button>
      </section>

      {err && <p className="text-sm text-danger">{err}</p>}

      <section className="rounded-xl border border-slate-800 bg-panel/50 p-4">
        <h3 className="text-xs font-medium uppercase text-slate-500">Jobs recentes</h3>
        <ul className="mt-2 space-y-3 font-mono text-sm">
          {Object.entries(jobs).map(([id, j]) => (
            <li key={id} className="border-b border-slate-800 pb-2">
              <div className="text-slate-400">
                {id.slice(0, 8)}… · {j.kind} · {j.status}
              </div>
              {j.result && (
                <div className="mt-1 text-xs text-slate-500">
                  {"manifest_csv" in j.result && (
                    <div>
                      <a href={repoFileUrl(String(j.result.manifest_csv))} className="text-accent hover:underline">
                        {String(j.result.manifest_csv)}
                      </a>
                    </div>
                  )}
                  {"label_file" in j.result && (
                    <div>
                      <a href={repoFileUrl(String(j.result.label_file))} className="text-accent hover:underline">
                        {String(j.result.label_file)}
                      </a>
                    </div>
                  )}
                  {"data_root" in j.result && <div>data_root: {String(j.result.data_root)}</div>}
                </div>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
