"""Processo filho: consulta GDC e descarrega WSI; grava manifest CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def _resolve_paths(cfg, server_root: Path) -> tuple[Path, Path, Path, Path]:
    """Devolve (root_dir, data_root, ids_csv, wsi_manifest_csv) absolutos."""
    if cfg.root_dir and str(cfg.root_dir).strip():
        rd = Path(cfg.root_dir)
        root = rd.resolve() if rd.is_absolute() else (server_root / rd).resolve()
    else:
        root = server_root.resolve()

    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()

    ids_csv = Path(cfg.ids_csv)
    if not ids_csv.is_absolute():
        ids_csv = (root / ids_csv).resolve()

    wsi_manifest = Path(cfg.wsi_manifest_csv)
    if not wsi_manifest.is_absolute():
        wsi_manifest = (root / wsi_manifest).resolve()

    return root, data_root, ids_csv, wsi_manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python -m server.tcga_download_worker <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    result_path = Path(payload["result_json_path"])
    server_root = Path(payload["server_root"])
    progress_path = payload.get("progress_json_path")

    from modules.tcga_dataset.config import TcgaDatasetConfig
    from modules.tcga_dataset.download import download_and_write_manifest
    from server.schemas import TcgaDownloadJobConfig

    tc = TcgaDownloadJobConfig.model_validate(payload["tcga_download"])
    root_dir, data_root, ids_csv, wsi_manifest_csv = _resolve_paths(tc, server_root)

    cfg_kwargs: dict = {
        "root_dir": root_dir,
        "data_root": data_root,
        "ids_csv": ids_csv,
        "case_id_column": tc.case_id_column,
        "wsi_manifest_csv": wsi_manifest_csv,
        "only_open_access": tc.only_open_access,
        "gdc_files_page_size": tc.gdc_files_page_size,
        "gdc_case_id_chunk_size": tc.gdc_case_id_chunk_size,
        "download_timeout_sec": tc.download_timeout_sec,
        "max_download_retries": tc.max_download_retries,
        "download_concurrency": tc.download_concurrency,
    }
    if tc.gdc_token and tc.gdc_token.strip():
        cfg_kwargs["gdc_token"] = tc.gdc_token.strip()

    ds_cfg = TcgaDatasetConfig(**cfg_kwargs)
    out_path = download_and_write_manifest(
        ds_cfg,
        progress_path=Path(progress_path).resolve() if progress_path else None,
    )

    try:
        df = pd.read_csv(out_path, encoding="utf-8")
        row_count = len(df)
    except EmptyDataError:
        row_count = 0

    result = {
        "manifest_path": str(out_path.resolve()),
        "row_count": row_count,
        "root_dir": str(root_dir.resolve()),
    }
    if row_count == 0:
        result["message"] = "Nenhum ficheiro SVS encontrado para os IDs informados."
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(result, rf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
