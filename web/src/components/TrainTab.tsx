import { useEffect, useMemo, useState } from "react";
import { fetchSchema, wsTrainUrl } from "../api";
import type { ApiSchemaPayload, StreamEvent } from "../types";
import { SchemaForm } from "./SchemaForm";
import { MetricPoint, TrainCharts } from "./TrainCharts";

function lrRisk(key: string, v: number): boolean {
  if ((key === "lr_fine" || key === "lr_baseline" || key === "lr_qat") && v > 1e-3) return true;
  return false;
}

export function TrainTab() {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [training, setTraining] = useState<Record<string, unknown>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("idle");
  const [points, setPoints] = useState<MetricPoint[]>([]);
  const [overfitBadge, setOverfitBadge] = useState(false);
  const [pulse, setPulse] = useState(false);
  const [completedInfo, setCompletedInfo] = useState<Record<string, string> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => {
    fetchSchema()
      .then((p) => {
        setSchemaPayload(p);
        setTraining({ ...p.defaults.TrainingJobConfig });
      })
      .catch((e) => setError(String(e)));
  }, []);

  const trainSchema = useMemo(() => {
    const s = schemaPayload?.schemas?.TrainingJobConfig as Record<string, unknown> | undefined;
    return s || {};
  }, [schemaPayload]);

  useEffect(() => {
    if (!jobId) return;
    const url = wsTrainUrl(jobId);
    const ws = new WebSocket(url);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as StreamEvent;
        if (msg.type === "epoch") {
          const w = Boolean(msg.warning);
          setOverfitBadge(w);
          setPoints((prev) => {
            const m = msg.metrics;
            const next: MetricPoint = {
              idx: prev.length,
              stage: msg.stage,
              epoch: msg.epoch,
              loss: m.loss,
              val_loss: m.val_loss,
              acc: m.accuracy,
              val_acc: m.val_accuracy,
              warn: w,
            };
            return [...prev, next];
          });
        } else if (msg.type === "target_val_loss_check" && msg.hit) {
          setPulse(true);
          setLog((prev) => [
            ...prev.slice(-80),
            `Meta val_loss atingida (melhor ${msg.best_val_loss.toFixed(4)} ≤ ${msg.target_val_loss})`,
          ]);
        } else if (msg.type === "completed") {
          setCompletedInfo({
            best_fp32: msg.best_fp32_model || "",
            tflite: msg.tflite || "",
            baseline: msg.baseline_final || "",
          });
          setJobStatus("completed");
        } else if (msg.type === "error") {
          setError(msg.message);
          setJobStatus("error");
        } else if (msg.type === "stream_end") {
          setJobStatus(msg.status);
        } else if (msg.type === "test_eval") {
          setLog((prev) => [
            ...prev.slice(-80),
            `Teste: loss=${msg.test_loss.toFixed(4)} acc=${msg.test_accuracy.toFixed(4)}`,
          ]);
        }
      } catch {
        /* ignore */
      }
    };
    ws.onerror = () => setLog((prev) => [...prev.slice(-80), "WebSocket erro"]);
    return () => ws.close();
  }, [jobId]);

  const startTrain = async () => {
    setError(null);
    setPoints([]);
    setOverfitBadge(false);
    setPulse(false);
    setCompletedInfo(null);
    setJobStatus("loading");
    const body = { training };
    const r = await fetch("/api/train/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      setJobStatus("error");
      setError(await r.text());
      return;
    }
    const j = (await r.json()) as { job_id: string; status: string };
    setJobId(j.job_id);
    setJobStatus("running");
  };

  const stopTrain = async () => {
    if (!jobId) return;
    setJobStatus("stopping");
    const r = await fetch(`/api/train/jobs/${jobId}/stop`, { method: "POST" });
    if (r.ok) {
      const j = (await r.json()) as { status: string };
      setJobStatus(j.status);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-6 lg:grid-cols-[minmax(280px,360px)_1fr]">
        <aside className="rounded-xl border border-slate-800 bg-surface/80 p-4">
          <h2 className="mb-3 text-sm font-semibold text-accent">Configuração de treino</h2>
          {schemaPayload && (
            <SchemaForm
              schemaRoot={trainSchema as never}
              values={training}
              onChange={setTraining}
              lrRiskCheck={lrRisk}
            />
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              disabled={jobStatus === "running" || jobStatus === "loading"}
              onClick={startTrain}
              className="rounded-lg bg-accent/90 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-accent disabled:opacity-40"
            >
              {jobStatus === "loading" ? "Iniciando…" : "Iniciar treinamento"}
            </button>
            <button
              type="button"
              disabled={jobStatus !== "running"}
              onClick={stopTrain}
              className="rounded-lg border border-danger/60 px-4 py-2 text-sm text-danger hover:bg-danger/10 disabled:opacity-40"
            >
              Interromper
            </button>
          </div>
          <p className="mt-2 text-xs text-slate-500">Estado: {jobStatus}</p>
          {error && <p className="mt-2 text-xs text-danger">{error}</p>}
          {overfitBadge && (
            <p className="mt-2 rounded border border-warn/50 bg-warn/10 px-2 py-1 text-xs text-warn">
              Atenção: val_loss muito acima de train (possível overfitting).
            </p>
          )}
        </aside>
        <section className="min-w-0 space-y-3">
          <TrainCharts data={points} pulse={pulse} />
          {completedInfo && (
            <div className="rounded-lg border border-ok/40 bg-ok/5 p-3 text-sm text-slate-300">
              <div className="font-medium text-ok">Melhores pesos / artefatos</div>
              <ul className="mt-2 list-inside list-disc text-xs text-slate-400">
                {completedInfo.best_fp32 && <li>FP32: {completedInfo.best_fp32}</li>}
                {completedInfo.baseline && <li>Baseline: {completedInfo.baseline}</li>}
                {completedInfo.tflite && <li>TFLite: {completedInfo.tflite}</li>}
              </ul>
            </div>
          )}
          <div className="rounded-lg border border-slate-800 bg-surface/40 p-3 font-mono text-[11px] text-slate-500">
            {log.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
