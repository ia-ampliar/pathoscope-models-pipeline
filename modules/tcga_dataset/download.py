"""
Orquestra a consulta ao GDC e o download de .SVS para ``data/<case>/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import gdc_api
from .config import TcgaDatasetConfig

logger = logging.getLogger(__name__)

# String de campos para POST /files (evita 500 com lista JSON nesta API).
GDC_FILE_FIELDS = "file_id,file_name,file_size,md5sum,cases.submitter_id"


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
    ids = df[col].dropna().astype(str).str.strip().unique().tolist()
    return [i for i in ids if i]


def should_skip_download(dest: Path, expected_size: int | None) -> bool:
    if not dest.is_file():
        return False
    if expected_size is None:
        return dest.stat().st_size > 0
    return dest.stat().st_size == expected_size


def download_ws_for_config(cfg: TcgaDatasetConfig) -> list[dict[str, str]]:
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

    normalized: list[dict[str, str]] = []
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

            case_dir = cfg.data_root / row["case_submitter_id"]
            dest = case_dir / row["file_name"]

            if should_skip_download(dest, row["file_size"]):
                logger.info("Já existe (skip): %s", dest)
            else:
                logger.info("A descarregar %s -> %s", fid, dest)
                gdc_api.download_with_retries(
                    fid,
                    dest,
                    token=cfg.gdc_token,
                    timeout=cfg.download_timeout_sec,
                    expected_size=row["file_size"],
                    max_retries=cfg.max_download_retries,
                )

            rel = dest.resolve().relative_to(cfg.root_dir.resolve())
            normalized.append(
                {
                    "image_path": str(rel).replace("\\", "/"),
                    "case_submitter_id": row["case_submitter_id"],
                    "file_id": fid,
                }
            )

    return normalized


def download_and_write_manifest(cfg: TcgaDatasetConfig) -> Path:
    """Executa download e grava ``cfg.wsi_manifest_csv``."""
    rows = download_ws_for_config(cfg)
    out = cfg.wsi_manifest_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    logger.info("Manifest escrito: %s (%d ficheiros)", out, len(rows))
    return out
