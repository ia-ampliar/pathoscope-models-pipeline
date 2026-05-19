import { useEffect, useMemo, useState } from "react";
import { fetchSchema } from "../api";
import type { ApiSchemaPayload, StepStatus, LogEntry } from "../types";
import { StageHeader } from "./StageHeader";
import { SchemaForm } from "./SchemaForm";
import { LogStream, makeLog } from "./LogStream";
import { ArtifactList } from "./ArtifactRow";

const GROUPS = [
  { label: "Entrada / Saída", keys: ["wsi_csv", "processed_dataset_dir", "wsi_path_column", "label_column"] },
  { label: "Extração", keys: ["tile_size", "target_magnification", "overlap"] },
  { label: "Qualidade", keys: ["max_white_background_fraction", "blur_laplacian_threshold", "jpeg_quality"] },
  { label: "Performance", keys: ["workers", "keep_staging"] },
];

type Props = { inspectorMode: boolean; onStatusChange: (s: StepStatus) => void };

export function TilingTab({ inspectorMode, onStatusChange }: Props) {
  const [schemaPayload, setSchemaPayload] = useState<ApiSchemaPayload | null>(null);
  const [tiling, setTiling] = useState<Record<string, unknown>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [existingDir, setExistingDir] = useState<string | null>(null);

  const addLog = (level: LogEntry["level"], msg: string) =>
    setLogs((p) => [...p.slice(-200), makeLog(level, msg)]);

  useEffect(() => {
    fetchSchema()
      .then((p) => { setSchemaPayload(p); setTiling({ ...p.defaults.TilingJobConfig }); })
      .catch((e) => setErr(String(e)));
  }, []);

  // Auto-detect existing output directory on the server
  useEffect(() => {
    const dir = typeof tiling.processed_dataset_dir === "string" && tiling.processed_dataset_dir
      ? tiling.processed_dataset_dir : "datas";
    fetch(`/api/filesystem/check?path=${encodeURIComponent(dir)}`)
      .then((r) => r.json())
      .then((d: { exists: boolean; is_dir: boolean }) => {
        setExistingDir(d.exists && d.is_dir ? dir : null);
      })
      .catch(() => {});
  }, [tiling.processed_dataset_dir]);

  const sch = useMemo(() => schemaPayload?.schemas?.TilingJobConfig || {}, [schemaPayload]);

  const run = async () => {
    setErr(null); setLogs([]); setStatus("loading");
    onStatusChange("running");
    addLog("INFO", "Iniciando tiling WSI…");
    const r = await fetch("/api/tiling/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tiling }),
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
      const s = await fetch(`/api/tiling/jobs/${j.job_id}`);
      if (!s.ok) return;
      const d = (await s.json()) as { status: string; error_message?: string };
      setStatus(d.status);
      if (d.error_message) { setErr(d.error_message); addLog("ERROR", d.error_message); }
      if (d.status === "completed") {
        onStatusChange("done");
        addLog("INFO", `Tiles gravados em: ${tiling.processed_dataset_dir ?? "datas/"}`);
        clearInterval(poll);
      }
      if (d.status === "error") { onStatusChange("error"); clearInterval(poll); }
    }, 800);
  };

  const outDir = typeof tiling.processed_dataset_dir === "string" && tiling.processed_dataset_dir
    ? tiling.processed_dataset_dir : "datas";

  const artifacts = [
    ...(existingDir ? [{ name: existingDir + "/", path: existingDir, type: "generic" as const }] : []),
    ...(status === "completed" && outDir !== existingDir
      ? [{ name: outDir + "/", path: outDir, type: "generic" as const }]
      : []),
  ];

  return (
    <div className="p-6">
      <StageHeader
        title="Tiling WSI"
        description="Extrai patches de WSIs (.svs) via OpenSlide + DeepZoom, aplica Macenko e organiza por classe em subpastas."
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
              values={tiling}
              onChange={setTiling}
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
