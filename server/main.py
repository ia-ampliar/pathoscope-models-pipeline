"""
API FastAPI: schema, jobs de treino/split/inferência, WebSocket de métricas.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.jobs import (
    JobRegistry,
    JobStatus,
    spawn_infer_worker,
    spawn_split_worker,
    spawn_tcga_download_worker,
    spawn_train_worker,
    stop_train_job,
    wait_infer_process,
    wait_split_process,
    wait_tcga_download_process,
    wait_train_process,
)
from server.schemas import (
    InferenceJobConfig,
    SplitJobConfig,
    TcgaDownloadJobConfig,
    TrainingJobConfig,
    build_api_schema_payload,
)

ROOT = Path(__file__).resolve().parents[1]
registry = JobRegistry(ROOT)

app = FastAPI(title="Pathoscope Control Room", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_web_dist = ROOT / "web" / "dist"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/schema")
def api_schema() -> dict[str, Any]:
    return build_api_schema_payload()


class TrainJobRequest(BaseModel):
    training: TrainingJobConfig


class SplitJobRequest(BaseModel):
    split: SplitJobConfig


@app.post("/api/train/jobs")
async def create_train_job(body: TrainJobRequest) -> JSONResponse:
    training = body.training
    job = registry.new_train_job()
    payload = {"training": training.model_dump(mode="json", exclude_none=True)}
    spawn_train_worker(ROOT, job, payload)

    async def _watch() -> None:
        await wait_train_process(job, ROOT)

    asyncio.create_task(_watch())
    return JSONResponse(
        {
            "job_id": job.id,
            "status": job.status.value,
            "metrics_stream_path": str(job.metrics_path) if job.metrics_path else None,
        }
    )


@app.get("/api/train/jobs/{job_id}")
def get_train_job(job_id: str) -> dict[str, Any]:
    job = registry.train_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "error_message": job.error_message,
        "metrics_path": str(job.metrics_path) if job.metrics_path else None,
    }


@app.post("/api/train/jobs/{job_id}/stop")
def stop_train(job_id: str) -> dict[str, str]:
    job = registry.train_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    stop_train_job(job)
    return {"status": job.status.value}


@app.websocket("/api/train/jobs/{job_id}/stream")
async def train_metrics_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = registry.train_jobs.get(job_id)
    if not job or not job.metrics_path:
        await websocket.send_json({"type": "error", "message": "Job inválido ou sem stream"})
        await websocket.close()
        return

    path = job.metrics_path
    last_pos = 0
    try:
        while True:
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(last_pos)
                    chunk = f.read()
                    last_pos = f.tell()
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            await websocket.send_json(json.loads(line))
                        except Exception:
                            await websocket.send_text(line)
            st = registry.train_jobs.get(job_id)
            if st and st.status in (JobStatus.completed, JobStatus.error, JobStatus.stopped):
                # drena o restante
                if path.exists():
                    with open(path, "rb") as f:
                        f.seek(last_pos)
                        chunk = f.read()
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        for line in text.splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    await websocket.send_json(json.loads(line))
                                except Exception:
                                    await websocket.send_text(line)
                await websocket.send_json({"type": "stream_end", "status": st.status.value})
                break
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


@app.post("/api/split/jobs")
async def create_split_job(body: SplitJobRequest) -> JSONResponse:
    cfg = body.split
    job = registry.new_split_job()
    spawn_split_worker(ROOT, job, cfg.model_dump(mode="json"))

    async def _watch() -> None:
        await wait_split_process(job, ROOT)

    asyncio.create_task(_watch())
    return JSONResponse({"job_id": job.id, "status": job.status.value})


@app.get("/api/split/jobs/{job_id}")
def get_split_job(job_id: str) -> dict[str, Any]:
    job = registry.split_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "error_message": job.error_message,
    }


@app.post("/api/tcga/download/jobs")
async def create_tcga_download_job(
    tcga_download_json: str = Form(...),
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    tc = TcgaDownloadJobConfig.model_validate(json.loads(tcga_download_json))
    job = registry.new_tcga_download_job()

    uploads = ROOT / "server_state" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    data = tc.model_dump(mode="json")
    if file and file.filename:
        dest = uploads / f"{job.id}_{file.filename}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        data["ids_csv"] = str(dest.resolve())

    result_json = ROOT / "server_state" / f"tcga_result_{job.id}.json"
    if result_json.exists():
        result_json.unlink()

    payload = {
        "tcga_download": data,
        "result_json_path": str(result_json.resolve()),
        "server_root": str(ROOT.resolve()),
    }
    spawn_tcga_download_worker(ROOT, job, payload)

    async def _watch() -> None:
        await wait_tcga_download_process(job, ROOT)

    asyncio.create_task(_watch())
    return JSONResponse({"job_id": job.id, "status": job.status.value})


@app.get("/api/tcga/download/jobs/{job_id}")
def get_tcga_download_job(job_id: str) -> dict[str, Any]:
    job = registry.tcga_download_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    progress: Optional[dict[str, Any]] = None
    if job.progress_path and job.progress_path.exists():
        try:
            with open(job.progress_path, encoding="utf-8") as pf:
                progress = json.load(pf)
        except Exception:
            progress = None

    return {
        "job_id": job.id,
        "status": job.status.value,
        "error_message": job.error_message,
        "result": job.result,
        "progress": progress,
    }


@app.post("/api/inference/jobs")
async def create_infer_job(
    inference_json: str = Form(...),
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    inv = InferenceJobConfig.model_validate(json.loads(inference_json))
    job = registry.new_infer_job()
    work_out = ROOT / "server_state" / f"infer_out_{job.id}"
    work_out.mkdir(parents=True, exist_ok=True)

    uploads = ROOT / "server_state" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    image_path: Optional[str] = inv.image_path
    if file and file.filename:
        dest = uploads / f"{job.id}_{file.filename}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        image_path = str(dest.resolve())

    if not image_path:
        raise HTTPException(400, "Forneça image_path na config ou envie um arquivo .svs")

    result_json = ROOT / "server_state" / f"infer_result_{job.id}.json"
    payload = {
        "inference": inv.model_dump(mode="json"),
        "image_path": image_path,
        "work_output_dir": str(work_out.resolve()),
        "result_json_path": str(result_json.resolve()),
    }
    spawn_infer_worker(ROOT, job, payload)

    async def _watch() -> None:
        await wait_infer_process(job, ROOT)

    asyncio.create_task(_watch())
    return JSONResponse({"job_id": job.id, "status": job.status.value})


@app.get("/api/inference/jobs/{job_id}")
def get_infer_job(job_id: str) -> dict[str, Any]:
    job = registry.infer_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "error_message": job.error_message,
        "result": job.result,
    }


@app.get("/api/artifacts/{job_id}/{rel_path:path}")
def serve_artifact(job_id: str, rel_path: str) -> FileResponse:
    """Serve arquivos gerados pela inferência (imagens NPZ dentro de server_state)."""
    base = ROOT / "server_state" / f"infer_out_{job_id}"
    target = (base / rel_path).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise HTTPException(403, "Caminho inválido")
    if not target.is_file():
        raise HTTPException(404, "Arquivo não encontrado")
    return FileResponse(target)


if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dist), html=True), name="spa")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)
