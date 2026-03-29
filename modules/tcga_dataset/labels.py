"""
Cruza manifest WSI com planilha (Google Sheets como CSV) para gerar ``label_file.csv``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from .config import TcgaDatasetConfig

logger = logging.getLogger(__name__)


def google_sheet_to_csv_export_url(url: str) -> str:
    """
    Converte URL de edição/partilha do Google Sheets em URL de exportação CSV.

    A planilha deve estar publicada na web ou partilhada como "anyone with the link"
    para leitura, caso contrário ``pandas.read_csv`` falhará sem credenciais OAuth.
    """
    url = url.strip()
    parsed = urlparse(url)
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError(f"URL não parece ser Google Sheets: {url}")
    spreadsheet_id = m.group(1)

    gid = "0"
    qs = parse_qs(parsed.query)
    if "gid" in qs and qs["gid"]:
        gid = qs["gid"][0]
    frag = parsed.fragment or ""
    if "gid=" in frag:
        gim = re.search(r"gid=(\d+)", frag)
        if gim:
            gid = gim.group(1)

    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    )


def load_sheet_dataframe(sheets_ref: str) -> pd.DataFrame:
    """Aceita caminho local ``.csv`` ou URL de Google Sheets."""
    from io import StringIO

    p = Path(sheets_ref)
    if p.is_file():
        return pd.read_csv(p, encoding="utf-8")
    export_url = google_sheet_to_csv_export_url(sheets_ref)
    r = requests.get(export_url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def patient_id_from_image_path(image_path: str, repo_root: Path) -> str | None:
    """
    Espera ``data/<PATIENT_CASE_ID>/arquivo.svs`` relativo ao repositório.
    """
    p = Path(image_path.replace("\\", "/").strip())
    parts = p.parts
    if len(parts) < 2:
        return None
    # parts[0] costuma ser "data"
    return parts[1]


def build_label_file(
    manifest_csv: Path,
    sheets_ref: str,
    output_csv: Path,
    repo_root: Path,
    manifest_path_column: str | None = None,
    patient_id_column: str = "Patient ID",
    subtype_column: str = "Subtype",
) -> Path:
    """
    Lê manifest, cruza com planilha (`Patient ID` + `Subtype`) e grava `Image_path`, `Label`.
    """
    man = pd.read_csv(manifest_csv)
    if man.empty:
        raise ValueError(f"Manifest vazio: {manifest_csv}")

    path_col = manifest_path_column
    if path_col is None or path_col not in man.columns:
        path_col = "image_path" if "image_path" in man.columns else man.columns[0]

    sheet = load_sheet_dataframe(sheets_ref)
    sheet.columns = sheet.columns.str.strip()
    pid = patient_id_column.strip()
    sub = subtype_column.strip()
    if pid not in sheet.columns:
        raise ValueError(f"Coluna '{pid}' não encontrada na planilha. Colunas: {list(sheet.columns)}")
    if sub not in sheet.columns:
        raise ValueError(f"Coluna '{sub}' não encontrada na planilha. Colunas: {list(sheet.columns)}")

    work = man[[path_col]].copy()
    work["_patient_id"] = work[path_col].map(lambda x: patient_id_from_image_path(str(x), repo_root))
    missing_path = work["_patient_id"].isna().sum()
    if missing_path:
        logger.warning("%d linhas sem Patient ID inferido do caminho.", int(missing_path))
    work = work.dropna(subset=["_patient_id"])
    work["_patient_id"] = work["_patient_id"].astype(str).str.strip()

    sheet_map = sheet[[pid, sub]].dropna(subset=[pid]).drop_duplicates(subset=[pid])
    sheet_map[pid] = sheet_map[pid].astype(str).str.strip()
    sheet_map = sheet_map.rename(columns={pid: "_patient_id", sub: "Label"})
    sheet_map["Label"] = sheet_map["Label"].astype(str).str.strip()

    merged = work.merge(sheet_map, on="_patient_id", how="inner")
    dropped = len(work) - len(merged)
    if dropped:
        logger.warning("%d linhas do manifest sem match em '%s'.", dropped, pid)

    out_df = merged[[path_col, "Label"]].rename(columns={path_col: "Image_path"})
    out_df["Label"] = out_df["Label"].astype(str).str.strip()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("label_file: %s (%d linhas)", output_csv, len(out_df))
    return output_csv


def build_label_file_from_config(cfg: TcgaDatasetConfig) -> Path:
    if not cfg.sheets_url:
        raise ValueError("sheets_url em TcgaDatasetConfig está vazio.")
    return build_label_file(
        manifest_csv=cfg.wsi_manifest_csv,
        sheets_ref=cfg.sheets_url,
        output_csv=cfg.label_file_csv,
        repo_root=cfg.root_dir,
        manifest_path_column=cfg.manifest_path_column,
        patient_id_column=cfg.patient_id_column,
        subtype_column=cfg.subtype_column,
    )
