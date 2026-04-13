import { useEffect, useMemo, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload } from "../types";
import { SchemaForm } from "./SchemaForm";

export function SplitTab() {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [split, setSplit] = useState<Record<string, unknown>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        setSchemaPayload(p);
        setSplit({ ...p.defaults.SplitJobConfig });
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const sch = useMemo(() => schemaPayload?.schemas?.SplitJobConfig || {}, [schemaPayload]);

  const run = async () => {
    setErr(null);
    setStatus("loading");
    const r = await fetch("/api/split/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ split }),
    });
    if (!r.ok) {
      setStatus("error");
      setErr(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id);
    setStatus("running");
    const poll = setInterval(async () => {
      const s = await fetch(`/api/split/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as { status: string; error_message?: string };
      setStatus(d.status);
      if (d.error_message) setErr(d.error_message);
      if (d.status === "completed" || d.status === "error") clearInterval(poll);
    }, 800);
  };

  return (
    <div className="max-w-xl rounded-xl border border-slate-800 bg-surface/80 p-4">
      <h2 className="mb-3 text-sm font-semibold text-accent">Split train/val/test</h2>
      <p className="mb-3 text-xs text-slate-500">
        Para lâminas TCGA brutas, use a aba <strong className="text-slate-400">TCGA / WSI</strong> (download,
        manifest, label_file) e depois aponte o split para o <code className="text-accent">label_file.csv</code> em{" "}
        <code className="text-accent">split/</code> ou faça tiling para pastas por classe em <code>datas/</code>.
      </p>
      {schemaPayload && (
        <SchemaForm schemaRoot={sch as never} values={split} onChange={setSplit} />
      )}
      <button
        type="button"
        onClick={run}
        disabled={status === "running" || status === "loading"}
        className="mt-4 rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
      >
        Gerar CSVs
      </button>
      <p className="mt-2 text-xs text-slate-500">
        Job: {jobId || "—"} · {status}
      </p>
      {err && <p className="mt-2 text-xs text-danger">{err}</p>}
    </div>
  );
}
