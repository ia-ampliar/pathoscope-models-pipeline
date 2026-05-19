import { useEffect, useMemo, useState } from "react";
import { fetchSchema, wsTrainUrl } from "../api";
import type { ApiSchemaPayload, StepStatus, LogEntry, StreamEvent } from "../types";
import { StageHeader } from "./StageHeader";
import { SchemaForm } from "./SchemaForm";
import { LogStream, makeLog } from "./LogStream";
import { ArtifactList } from "./ArtifactRow";
import { MetricsPanel } from "./MetricsPanel";
import type { MetricPoint } from "./MetricsPanel";

const GROUPS = [
  { label: "Dados", keys: ["train_csv", "val_csv", "test_csv", "dataset_path", "label_csv"] },
  { label: "Otimização", keys: ["lr_baseline", "lr_fine", "lr_qat", "batch_size", "fine_tune_percent"] },
  { label: "Épocas", keys: ["epochs_baseline", "epochs_fine", "epochs_qat"] },
  { label: "QAT / Augment", keys: ["use_qat", "augment", "use_weighted_loss", "target_val_loss"] },
  { label: "Saída", keys: ["model_dir", "output_dir", "experiment_name"] },
];

function lrRisk(key: string, v: number): boolean {
  return (key === "lr_fine" || key === "lr_baseline" || key === "lr_qat") && v > 1e-3;
}

type Props = { inspectorMode: boolean; onStatusChange: (s: StepStatus) => void };

export function TrainTab({ inspectorMode, onStatusChange }: Props) {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [training, setTraining] = useState<Record<string, unknown>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [epochInfo, setEpochInfo] = useState<string | undefined>(undefined);
  const [points, setPoints] = useState<MetricPoint[]>([]);
  const [pulse, setPulse] = useState(false);
  const [completedInfo, setCompletedInfo] = useState<Record<string, string> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = (level: LogEntry["level"], msg: string) =>
    setLogs((p) => [...p.slice(-200), makeLog(level, msg)]);

  useEffect(() => {
    fetchSchema()
      .then((p) => { setSchemaPayload(p); setTraining({ ...p.defaults.TrainingJobConfig }); })
      .catch((e) => setErr(String(e)));
  }, []);

  const trainSchema = useMemo(
    () => (schemaPayload?.schemas?.TrainingJobConfig as Record<string, unknown> | undefined) || {},
    [schemaPayload]
  );

  useEffect(() => {
    if (!jobId) return;
    const ws = new WebSocket(wsTrainUrl(jobId));
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as StreamEvent;
        if (msg.type === "stage_start") {
          setEpochInfo(`${msg.stage}`);
          addLog("INFO", `Fase iniciada: ${msg.stage}`);
        } else if (msg.type === "epoch") {
          const w = Boolean(msg.warning);
          const m = msg.metrics;
          setPoints((prev) => [
            ...prev,
            {
              idx: prev.length,
              stage: msg.stage,
              epoch: msg.epoch,
              loss: m.loss,
              val_loss: m.val_loss,
              acc: m.accuracy,
              val_acc: m.val_accuracy,
              warn: w,
            },
          ]);
          setEpochInfo(`${msg.stage} · epoch ${msg.epoch}`);
          if (w) addLog("WARN", `Overfitting detectado (epoch ${msg.epoch}) · val_loss: ${m.val_loss?.toFixed(4)}`);
        } else if (msg.type === "target_val_loss_check" && msg.hit) {
          setPulse(true);
          addLog("INFO", `Meta val_loss atingida: ${msg.best_val_loss.toFixed(4)} ≤ ${msg.target_val_loss}`);
        } else if (msg.type === "test_eval") {
          addLog("INFO", `Teste: loss=${msg.test_loss.toFixed(4)} · acc=${msg.test_accuracy.toFixed(4)}`);
        } else if (msg.type === "completed") {
          const info: Record<string, string> = {};
          if (msg.best_fp32_model) info.best_fp32 = msg.best_fp32_model;
          if (msg.baseline_final) info.baseline = msg.baseline_final;
          if (msg.tflite) info.tflite = msg.tflite;
          setCompletedInfo(info);
          setStatus("completed");
          onStatusChange("done");
          addLog("INFO", "Treinamento concluído.");
        } else if (msg.type === "error") {
          setErr(msg.message);
          setStatus("error");
          onStatusChange("error");
          addLog("ERROR", msg.message);
        } else if (msg.type === "stream_end") {
          setStatus(msg.status);
          if (msg.status !== "completed") onStatusChange("error");
        }
      } catch { /* ignore */ }
    };
    ws.onerror = () => addLog("ERROR", "WebSocket: erro de conexão");
    return () => ws.close();
  }, [jobId]);

  const startTrain = async () => {
    setErr(null); setPoints([]); setLogs([]); setPulse(false);
    setCompletedInfo(null); setEpochInfo(undefined);
    setStatus("loading");
    onStatusChange("running");
    addLog("INFO", "Iniciando treinamento…");
    const r = await fetch("/api/train/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ training }),
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
  };

  const stopTrain = async () => {
    if (!jobId) return;
    setStatus("stopping");
    addLog("WARN", "Interrompendo treinamento…");
    const r = await fetch(`/api/train/jobs/${jobId}/stop`, { method: "POST" });
    if (r.ok) {
      const j = (await r.json()) as { status: string };
      setStatus(j.status);
    }
  };

  const artifacts = completedInfo
    ? [
        ...(completedInfo.baseline
          ? [{ name: "baseline_checkpoint_final.keras", path: completedInfo.baseline, type: "model" as const }]
          : []),
        ...(completedInfo.best_fp32
          ? [{ name: "fine_tuned_checkpoint_final.keras", path: completedInfo.best_fp32, type: "model" as const }]
          : []),
        ...(completedInfo.tflite
          ? [{ name: "qat_baseline_final.tflite", path: completedInfo.tflite, type: "tflite" as const }]
          : []),
      ]
    : [];

  return (
    <div className="p-6">
      <StageHeader
        title="Treino"
        description="MobileNetV2 com baseline → fine-tune → QAT, overfitting detection e target val_loss automático."
        status={status}
        jobId={jobId}
        phase={epochInfo}
        onRun={startTrain}
        onAbort={stopTrain}
        runDisabled={status === "running" || status === "loading"}
        abortDisabled={status !== "running"}
      />

      <div className="grid gap-5 xl:grid-cols-[300px_1fr_300px] lg:grid-cols-[300px_1fr]">
        {/* Params */}
        <div>
          {schemaPayload && (
            <SchemaForm
              schemaRoot={trainSchema as never}
              values={training}
              onChange={setTraining}
              lrRiskCheck={lrRisk}
              groups={GROUPS}
              inspectorMode={inspectorMode}
            />
          )}
        </div>

        {/* Metrics */}
        <div>
          <MetricsPanel data={points} pulse={pulse} />
        </div>

        {/* Logs + Artifacts */}
        <div className="space-y-3">
          <LogStream entries={logs} />
          <ArtifactList title="Modelos gerados" artifacts={artifacts} />
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
