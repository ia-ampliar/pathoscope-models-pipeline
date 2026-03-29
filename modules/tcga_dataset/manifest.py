"""
Constrói o CSV de caminhos WSI a partir de ficheiros já presentes em disco.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import TcgaDatasetConfig

logger = logging.getLogger(__name__)


def build_wsi_manifest_from_disk(
    data_root: Path,
    repo_root: Path,
    output_csv: Path,
    path_column: str = "image_path",
) -> Path:
    """
    Percorre ``data_root`` por ``*.svs`` e grava CSV com caminhos relativos a ``repo_root``.
    """
    data_root = data_root.resolve()
    repo_root = repo_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Directório de dados não existe: {data_root}")

    rows: list[dict[str, str]] = []
    for pattern in ("*.svs", "*.SVS"):
        for svs in sorted(data_root.rglob(pattern)):
            if not svs.is_file():
                continue
            try:
                rel = svs.resolve().relative_to(repo_root)
            except ValueError:
                rel = svs.resolve()
                logger.warning("Caminho fora de repo_root; usando absoluto: %s", rel)
            rows.append({path_column: str(rel).replace("\\", "/")})

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().sort_values(path_column).to_csv(
        output_csv, index=False, encoding="utf-8"
    )
    logger.info("Manifest a partir de disco: %s (%d linhas)", output_csv, len(rows))
    return output_csv


def manifest_from_config(cfg: TcgaDatasetConfig) -> Path:
    """Usa ``cfg.data_root``, ``cfg.root_dir`` e ``cfg.wsi_manifest_csv``."""
    return build_wsi_manifest_from_disk(
        cfg.data_root,
        cfg.root_dir,
        cfg.wsi_manifest_csv,
        path_column=cfg.manifest_path_column,
    )
