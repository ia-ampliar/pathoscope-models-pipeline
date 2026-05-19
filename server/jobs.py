"""
Estado de jobs em memória + utilitários para subprocessos.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    error = "error"
    stopped = "stopped"


@dataclass
class TrainJob:
    id: str
    status: JobStatus = JobStatus.pending
    process: Optional[subprocess.Popen] = None
    metrics_path: Optional[Path] = None
    config_path: Optional[Path] = None
    error_message: Optional[str] = None


@dataclass
class SplitJob:
    id: str
    status: JobStatus = JobStatus.pending
    process: Optional[subprocess.Popen] = None
    error_message: Optional[str] = None


@dataclass
class InferJob:
    id: str
    status: JobStatus = JobStatus.pending
    process: Optional[subprocess.Popen] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class TcgaDownloadJob:
    id: str
    status: JobStatus = JobStatus.pending
    process: Optional[subprocess.Popen] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress_path: Optional[Path] = None


@dataclass
class LabelJob:
    id: str
    status: JobStatus = JobStatus.pending
    process: Optional[subprocess.Popen] = None
    error_message: Optional[str] = None


@dataclass
class TilingJob:
    id: str
    status: JobStatus = JobStatus.pending
    process: Optional[subprocess.Popen] = None
    error_message: Optional[str] = None


class JobRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.train_jobs: Dict[str, TrainJob] = {}
        self.split_jobs: Dict[str, SplitJob] = {}
        self.infer_jobs: Dict[str, InferJob] = {}
        self.tcga_download_jobs: Dict[str, TcgaDownloadJob] = {}
        self.label_jobs: Dict[str, LabelJob] = {}
        self.tiling_jobs: Dict[str, TilingJob] = {}
        self.jobs_dir = root / "server_state"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def new_train_job(self) -> TrainJob:
        jid = str(uuid.uuid4())
        job = TrainJob(id=jid)
        self.train_jobs[jid] = job
        return job

    def new_split_job(self) -> SplitJob:
        jid = str(uuid.uuid4())
        job = SplitJob(id=jid)
        self.split_jobs[jid] = job
        return job

    def new_infer_job(self) -> InferJob:
        jid = str(uuid.uuid4())
        job = InferJob(id=jid)
        self.infer_jobs[jid] = job
        return job

    def new_tcga_download_job(self) -> TcgaDownloadJob:
        jid = str(uuid.uuid4())
        job = TcgaDownloadJob(id=jid)
        self.tcga_download_jobs[jid] = job
        return job

    def new_label_job(self) -> LabelJob:
        jid = str(uuid.uuid4())
        job = LabelJob(id=jid)
        self.label_jobs[jid] = job
        return job

    def new_tiling_job(self) -> TilingJob:
        jid = str(uuid.uuid4())
        job = TilingJob(id=jid)
        self.tiling_jobs[jid] = job
        return job


def _python_executable() -> str:
    return sys.executable


def spawn_train_worker(root: Path, job: TrainJob, config_payload: dict) -> None:
    cfg_file = root / "server_state" / f"train_{job.id}.json"
    job.config_path = cfg_file
    job.metrics_path = root / "server_state" / f"metrics_{job.id}.jsonl"
    config_payload = {**config_payload, "metrics_stream_path": str(job.metrics_path)}
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    if job.metrics_path.exists():
        job.metrics_path.unlink()

    log_path = root / "server_state" / f"train_{job.id}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_python_executable(), "-m", "server.train_worker", str(cfg_file)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
    )
    job.process = proc
    job.status = JobStatus.running


async def wait_train_process(job: TrainJob, root: Path) -> None:
    if job.process is None:
        return
    loop = asyncio.get_event_loop()
    ret = await loop.run_in_executor(None, job.process.wait)
    if job.status == JobStatus.stopped:
        return
    if ret != 0:
        log_path = root / "server_state" / f"train_{job.id}.log"
        err = ""
        if log_path.exists():
            err = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        job.error_message = err or f"exit {ret}"
        job.status = JobStatus.error
    else:
        job.status = JobStatus.completed


def stop_train_job(job: TrainJob) -> None:
    if job.process and job.process.poll() is None:
        job.process.terminate()
        try:
            job.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            job.process.kill()
    if job.status == JobStatus.running:
        job.status = JobStatus.stopped


def spawn_split_worker(root: Path, job: SplitJob, payload: dict) -> None:
    cfg_file = root / "server_state" / f"split_{job.id}.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = root / "server_state" / f"split_{job.id}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_python_executable(), "-m", "server.split_worker", str(cfg_file)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
    )
    job.process = proc
    job.status = JobStatus.running


async def wait_split_process(job: SplitJob, root: Path) -> None:
    if job.process is None:
        return
    loop = asyncio.get_event_loop()
    ret = await loop.run_in_executor(None, job.process.wait)
    if ret != 0:
        log_path = root / "server_state" / f"split_{job.id}.log"
        err = ""
        if log_path.exists():
            err = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        job.error_message = err or f"exit {ret}"
        job.status = JobStatus.error
    else:
        job.status = JobStatus.completed


def spawn_infer_worker(root: Path, job: InferJob, payload: dict) -> None:
    cfg_file = root / "server_state" / f"infer_{job.id}.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = root / "server_state" / f"infer_{job.id}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_python_executable(), "-m", "server.infer_worker", str(cfg_file)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
    )
    job.process = proc
    job.status = JobStatus.running


def spawn_tcga_download_worker(root: Path, job: TcgaDownloadJob, payload: dict) -> None:
    cfg_file = root / "server_state" / f"tcga_download_{job.id}.json"
    progress_path = root / "server_state" / f"tcga_progress_{job.id}.json"
    job.progress_path = progress_path
    payload = {**payload, "progress_json_path": str(progress_path.resolve())}
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = root / "server_state" / f"tcga_download_{job.id}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_python_executable(), "-m", "server.tcga_download_worker", str(cfg_file)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
    )
    job.process = proc
    job.status = JobStatus.running


async def wait_tcga_download_process(job: TcgaDownloadJob, root: Path) -> None:
    if job.process is None:
        return
    loop = asyncio.get_event_loop()
    ret = await loop.run_in_executor(None, job.process.wait)
    out_path = root / "server_state" / f"tcga_result_{job.id}.json"
    if ret == 0 and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            job.result = json.load(f)
        job.status = JobStatus.completed
    else:
        log_path = root / "server_state" / f"tcga_download_{job.id}.log"
        err = ""
        if log_path.exists():
            err = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        job.error_message = err or f"exit {ret}"
        job.status = JobStatus.error


async def wait_infer_process(job: InferJob, root: Path) -> None:
    if job.process is None:
        return
    loop = asyncio.get_event_loop()
    ret = await loop.run_in_executor(None, job.process.wait)
    out_path = root / "server_state" / f"infer_result_{job.id}.json"
    if ret == 0 and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            job.result = json.load(f)
        job.status = JobStatus.completed
    else:
        log_path = root / "server_state" / f"infer_{job.id}.log"
        err = ""
        if log_path.exists():
            err = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        job.error_message = err or f"exit {ret}"
        job.status = JobStatus.error


def spawn_label_worker(root: Path, job: LabelJob, payload: dict) -> None:
    cfg_file = root / "server_state" / f"label_{job.id}.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = root / "server_state" / f"label_{job.id}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_python_executable(), "-m", "server.label_worker", str(cfg_file)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
    )
    job.process = proc
    job.status = JobStatus.running


async def wait_label_process(job: LabelJob, root: Path) -> None:
    if job.process is None:
        return
    loop = asyncio.get_event_loop()
    ret = await loop.run_in_executor(None, job.process.wait)
    if ret != 0:
        log_path = root / "server_state" / f"label_{job.id}.log"
        err = ""
        if log_path.exists():
            err = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        job.error_message = err or f"exit {ret}"
        job.status = JobStatus.error
    else:
        job.status = JobStatus.completed


def spawn_tiling_worker(root: Path, job: TilingJob, payload: dict) -> None:
    cfg_file = root / "server_state" / f"tiling_{job.id}.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = root / "server_state" / f"tiling_{job.id}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_python_executable(), "-m", "server.tiling_worker", str(cfg_file)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
    )
    job.process = proc
    job.status = JobStatus.running


async def wait_tiling_process(job: TilingJob, root: Path) -> None:
    if job.process is None:
        return
    loop = asyncio.get_event_loop()
    ret = await loop.run_in_executor(None, job.process.wait)
    if ret != 0:
        log_path = root / "server_state" / f"tiling_{job.id}.log"
        err = ""
        if log_path.exists():
            err = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        job.error_message = err or f"exit {ret}"
        job.status = JobStatus.error
    else:
        job.status = JobStatus.completed
