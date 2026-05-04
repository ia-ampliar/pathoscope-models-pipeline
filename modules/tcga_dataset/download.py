"""
Orquestra a consulta ao GDC e o download de .SVS para ``data/<case>/``.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import gdc_api
from .config import TcgaDatasetConfig

logger = logging.getLogger(__name__)

# String de campos para POST /files (evita 500 com lista JSON nesta API).
GDC_FILE_FIELDS = "file_id,file_name,file_size,md5sum,cases.submitter_id"
MANIFEST_COLUMNS = ["image_path", "case_submitter_id", "file_id"]


def _write_progress_json(progress_path: Path | None, payload: dict) -> None:
    """Escreve progresso de forma atómica para leitura pela API/UI."""
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(progress_path)


def _normalize_tcga_case_id(raw: str) -> str:
    """Aceita case_id curto ou barcode longo e devolve ``TCGA-XX-XXXX``."""
    v = raw.strip().upper()
    if not v:
        return ""
    if v.startswith("TCGA-") and len(v) >= 12:
        return v[:12]
    return v


def read_case_submitter_ids(ids_csv: Path, column: str) -> list[str]:
    """Lê o CSV de caso; se a coluna não existir, usa a primeira coluna."""
    df = pd.read_csv(ids_csv)
    if column not in df.columns:
        if len(df.columns) < 1:
            raise ValueError(f"CSV vazio ou sem colunas: {ids_csv}")
        col = df.columns[0]
        logger.warning("Coluna '%s' ausente; usando primeira coluna '%s'.", column, col)
    else:
        col = column
    vals = df[col].dropna().astype(str).tolist()
    out: list[str] = []
    seen: set[str] = set()
    for raw in vals:
        cid = _normalize_tcga_case_id(raw)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def should_skip_download(dest: Path, expected_size: int | None) -> bool:
    if not dest.is_file():
        return False
    if expected_size is None:
        return dest.stat().st_size > 0
    return dest.stat().st_size == expected_size


def download_ws_for_config(
    cfg: TcgaDatasetConfig,
    progress_path: Path | None = None,
) -> list[dict[str, str]]:
    """
    Descobre ficheiros SVS no GDC e desce-os para ``cfg.data_root/<case>/<file_name>``.

    Returns:
        Lista de dicts com chaves ``image_path`` (relativo a ``cfg.root_dir``), ``case_submitter_id``,
        ``file_id`` (para depuração).
    """
    case_ids = read_case_submitter_ids(cfg.ids_csv, cfg.case_id_column)
    if not case_ids:
        raise ValueError(f"Nenhum case ID encontrado em {cfg.ids_csv}")

    cfg.data_root.mkdir(parents=True, exist_ok=True)
    chunk = max(1, cfg.gdc_case_id_chunk_size)

    _write_progress_json(
        progress_path,
        {"phase": "listing", "message": "A consultar API GDC...", "current": 0, "total": 0, "pending_count": 0},
    )

    pending: list[dict[str, str | int | None]] = []
    seen_ids: set[str] = set()

    for i in range(0, len(case_ids), chunk):
        batch = case_ids[i : i + chunk]
        filters = gdc_api.build_svs_filters(batch, only_open_access=cfg.only_open_access)
        for hit in gdc_api.iter_file_hits(
            filters,
            GDC_FILE_FIELDS,
            page_size=cfg.gdc_files_page_size,
            token=cfg.gdc_token,
        ):
            row = gdc_api.normalize_hit(hit)
            if not row:
                continue
            fid = row["file_id"]
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            pending.append(row)

        _write_progress_json(
            progress_path,
            {
                "phase": "listing",
                "message": "A consultar API GDC...",
                "current": 0,
                "total": 0,
                "pending_count": len(pending),
            },
        )

    total = len(pending)
    _write_progress_json(
        progress_path,
        {
            "phase": "downloading",
            "message": "A descarregar ficheiros SVS...",
            "current": 0,
            "total": total,
            "pending_count": total,
        },
    )

    if total == 0:
        _write_progress_json(
            progress_path,
            {
                "phase": "done",
                "message": "Nenhum ficheiro SVS encontrado para os IDs informados.",
                "current": 0,
                "total": 0,
                "pending_count": 0,
            },
        )
        return []

    normalized: list[dict[str, str] | None] = [None] * total
    lock = threading.Lock()
    state = {"current": 0, "file_name": None, "case_submitter_id": None}

    def _download_one(idx: int, row: dict[str, str | int | None]) -> tuple[int, dict[str, str]]:
        case_submitter_id = str(row["case_submitter_id"])
        file_name = str(row["file_name"])
        file_id = str(row["file_id"])
        file_size = row.get("file_size")
        expected_size = int(file_size) if isinstance(file_size, int) else None

        case_dir = cfg.data_root / case_submitter_id
        dest = case_dir / file_name
        if should_skip_download(dest, expected_size):
            logger.info("Já existe (skip): %s", dest)
        else:
            logger.info("A descarregar %s -> %s", file_id, dest)
            gdc_api.download_with_retries(
                file_id,
                dest,
                token=cfg.gdc_token,
                timeout=cfg.download_timeout_sec,
                expected_size=expected_size,
                max_retries=cfg.max_download_retries,
            )

        rel = dest.resolve().relative_to(cfg.root_dir.resolve())
        item = {
            "image_path": str(rel).replace("\\", "/"),
            "case_submitter_id": case_submitter_id,
            "file_id": file_id,
        }
        return idx, item

    try:
        workers = max(1, cfg.download_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_download_one, idx, row) for idx, row in enumerate(pending)]
            for fut in as_completed(futures):
                idx, item = fut.result()
                normalized[idx] = item
                with lock:
                    state["current"] += 1
                    state["file_name"] = Path(item["image_path"]).name
                    state["case_submitter_id"] = item["case_submitter_id"]
                    _write_progress_json(
                        progress_path,
                        {
                            "phase": "downloading",
                            "message": "A descarregar ficheiros SVS...",
                            "current": state["current"],
                            "total": total,
                            "pending_count": total,
                            "file_name": state["file_name"],
                            "case_submitter_id": state["case_submitter_id"],
                        },
                    )
    except Exception as e:
        _write_progress_json(
            progress_path,
            {
                "phase": "error",
                "message": f"Erro durante download: {e}",
                "current": state["current"],
                "total": total,
                "pending_count": total,
                "file_name": state["file_name"],
                "case_submitter_id": state["case_submitter_id"],
            },
        )
        raise

    out = [x for x in normalized if x]
    _write_progress_json(
        progress_path,
        {
            "phase": "done",
            "message": "Download concluído.",
            "current": len(out),
            "total": total,
            "pending_count": total,
        },
    )
    return out


def download_and_write_manifest(
    cfg: TcgaDatasetConfig,
    progress_path: Path | None = None,
) -> Path:
    """Executa download e grava ``cfg.wsi_manifest_csv``."""
    rows = download_ws_for_config(cfg, progress_path=progress_path)
    out = cfg.wsi_manifest_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(out, index=False, encoding="utf-8")
    logger.info("Manifest escrito: %s (%d ficheiros)", out, len(rows))
    return out
