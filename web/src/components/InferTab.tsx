import { useEffect, useMemo, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload } from "../types";
import { SchemaForm } from "./SchemaForm";

export function InferTab() {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [infer, setInfer] = useState<Record<string, unknown>>({});
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [beam, setBeam] = useState(false);

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        setSchemaPayload(p);
        setInfer({ ...p.defaults.InferenceJobConfig });
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const sch = useMemo(() => schemaPayload?.schemas?.InferenceJobConfig || {}, [schemaPayload]);

  const run = async () => {
    setErr(null);
    setResult(null);
    setStatus("loading");
    setBeam(true);
    const fd = new FormData();
    fd.append("inference_json", JSON.stringify(infer));
    if (file) fd.append("file", file);
    const r = await fetch("/api/inference/jobs", { method: "POST", body: fd });
    if (!r.ok) {
      setStatus("error");
      setErr(await r.text());
      setBeam(false);
      return;
    }
    const j = (await r.json()) as { job_id: string };
    setJobId(j.job_id);
    setStatus("running");
    const poll = setInterval(async () => {
      const s = await fetch(`/api/inference/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as {
        status: string;
        error_message?: string;
        result?: Record<string, unknown>;
      };
      setStatus(d.status);
      if (d.error_message) setErr(d.error_message);
      if (d.result) setResult(d.result);
      if (d.status === "completed" || d.status === "error") {
        clearInterval(poll);
        setBeam(false);
      }
    }, 1500);
  };

  const metrics = result?.metrics as Record<string, unknown> | undefined;
  const artifacts = result?.artifacts as { files?: { path: string; suffix: string }[] } | undefined;
  const jpg = artifacts?.files?.filter((f) => f.suffix === ".jpg") || [];

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(280px,380px)_1fr]">
      <aside className="rounded-xl border border-slate-800 bg-surface/80 p-4">
        <h2 className="mb-3 text-sm font-semibold text-accent">Inferência WSI</h2>
        {schemaPayload && (
          <SchemaForm schemaRoot={sch as never} values={infer} onChange={setInfer} />
        )}
        <div
          className="mt-4 cursor-pointer rounded-lg border-2 border-dashed border-slate-600 bg-panel/50 p-6 text-center text-sm text-slate-400 hover:border-accent/50"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) setFile(f);
          }}
        >
          <input
            type="file"
            accept=".svs"
            className="hidden"
            id="svs"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <label htmlFor="svs" className="cursor-pointer">
            Arraste .svs ou clique para escolher
          </label>
          {file && <p className="mt-2 text-xs text-accent">{file.name}</p>}
        </div>
        <button
          type="button"
          onClick={run}
          disabled={status === "running" || status === "loading"}
          className="mt-4 w-full rounded-lg bg-accent/90 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
        >
          Executar inferência
        </button>
        <p className="mt-2 text-xs text-slate-500">
          Job: {jobId || "—"} · {status}
        </p>
        {err && <p className="mt-2 text-xs text-danger">{err}</p>}
      </aside>
      <section className="relative min-h-[320px] rounded-xl border border-slate-800 bg-surface/40 p-4">
        <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 overflow-hidden rounded bg-slate-800">
          {beam && <div className="h-full w-1/3 animate-beam bg-gradient-to-r from-transparent via-accent to-transparent" />}
        </div>
        <div className="relative grid gap-4 md:grid-cols-2">
          <div>
            <h3 className="text-xs uppercase text-slate-500">Entrada</h3>
            <p className="mt-1 text-sm text-slate-300">
              {file?.name || (infer.image_path as string) || "Nenhum arquivo"}
            </p>
            {metrics && (
              <pre className="mt-2 max-h-48 overflow-auto rounded bg-panel p-2 text-[10px] text-slate-400">
                {JSON.stringify(metrics, null, 2)}
              </pre>
            )}
          </div>
          <div>
            <h3 className="text-xs uppercase text-slate-500">Saída (heatmap / métricas)</h3>
            <p className="mt-1 text-xs text-slate-500">
              Inferência por patch na lâmina; confiança por patch reflete o limiar THRESHOLD e a prob. da classe
              positiva.
            </p>
            <div className="mt-2 grid gap-2">
              {jpg.map((f) => (
                <img
                  key={f.path}
                  src={`/api/artifacts/${jobId}/${f.path.split("/").map(encodeURIComponent).join("/")}`}
                  alt={f.path}
                  className="max-h-64 rounded border border-slate-700 object-contain"
                />
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
