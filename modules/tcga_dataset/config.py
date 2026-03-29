"""
Configuração do módulo TCGA/GDC: pastas, CSVs e colunas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import modules.config as app_config


def _env_token() -> str | None:
    v = os.environ.get("GDC_TOKEN", "").strip()
    return v or None


@dataclass
class TcgaDatasetConfig:
    """Defaults alinhados ao plano: WSI brutos em ``data/``, manifest e labels configuráveis."""

    root_dir: Path = field(default_factory=lambda: app_config.ROOT_DIR)
    data_root: Path = field(default_factory=lambda: app_config.ROOT_DIR / "data")
    """Directório base onde cada caso tem uma subpasta com ficheiros .svs."""

    ids_csv: Path = field(default_factory=lambda: app_config.ROOT_DIR / "tcga_case_ids.csv")
    """CSV com identificadores de caso TCGA (submitter_id)."""

    case_id_column: str = "case_submitter_id"
    """Nome da coluna no ``ids_csv`` com IDs tipo ``TCGA-BR-4191``."""

    wsi_manifest_csv: Path = field(default_factory=lambda: app_config.ROOT_DIR / "wsi_manifest.csv")
    """CSV listando caminhos relativos ``data/<caso>/*.svs``."""

    label_file_csv: Path = field(default_factory=lambda: app_config.SPLIT_DIR / "label_file.csv")
    """Saída: ``Image_path``, ``Label`` (convenção de ``create_split``)."""

    sheets_url: str = ""
    """URL da planilha Google (edit ou publicada); ver ``labels.google_sheet_to_csv_url``."""

    patient_id_column: str = "Patient ID"
    subtype_column: str = "Subtype"

    gdc_token: str | None = field(default_factory=_env_token)
    """Token opcional para dados controlados (`X-Auth-Token`)."""

    gdc_files_page_size: int = 500
    download_timeout_sec: int = 3600
    max_download_retries: int = 3
    only_open_access: bool = True
    """Se True, filtra ``access=open`` (sem token). Se False, inclui controlados (requer ``GDC_TOKEN``)."""

    gdc_case_id_chunk_size: int = 40
    """Tamanho máximo do operador ``in`` em ``cases.submitter_id`` por pedido à API."""

    manifest_path_column: str = "image_path"
    """Coluna no manifest (etapa labels) com caminho relativo ao repositório."""
