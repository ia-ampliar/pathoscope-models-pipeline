"""
Configuração do pipeline WSI (tiling + pré-processamento).

Valores padrão podem ser sobrescritos por variáveis de ambiente (prefixo WSI_)
e/ou por um ficheiro JSON (merge por campo).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, field, replace
from pathlib import Path

import modules.config as config


def _env_path(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    return Path(v).expanduser() if v else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None and v != "" else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None and v != "" else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class WSIPipelineConfig:
    """Parâmetros do tiling DeepZoom/OpenSlide e do pré-processamento."""

    wsi_csv: Path = Path("manifest.csv")
    wsi_path_column: str = "wsi_path"
    label_column: str = "label"
    template_image: Path = field(
        default_factory=lambda: _env_path("WSI_TEMPLATE_IMAGE", config.ROOT_DIR / "img_template.png")
    )
    tiling_staging_dir: Path = field(
        default_factory=lambda: _env_path(
            "WSI_TILING_STAGING_DIR", config.ROOT_DIR / "workspace" / "wsi_tiling_raw"
        )
    )
    processed_dataset_dir: Path = field(
        default_factory=lambda: _env_path("WSI_PROCESSED_DATASET_DIR", config.DATA_DIR)
    )
    tile_size: int = field(default_factory=lambda: _env_int("WSI_TILE_SIZE", 1000))
    tile_size_subtract_legacy: int = field(
        default_factory=lambda: _env_int("WSI_TILE_SIZE_SUBTRACT", 2)
    )
    overlap: int = field(default_factory=lambda: _env_int("WSI_TILE_OVERLAP", 1))
    limit_bounds: bool = field(default_factory=lambda: _env_bool("WSI_LIMIT_BOUNDS", True))
    jpeg_quality: int = field(default_factory=lambda: _env_int("WSI_JPEG_QUALITY", 90))
    image_format: str = field(default_factory=lambda: os.environ.get("WSI_IMAGE_FORMAT", "jpeg"))
    workers: int = field(default_factory=lambda: _env_int("WSI_WORKERS", 8))
    target_magnification: int = field(
        default_factory=lambda: _env_int("WSI_TARGET_MAGNIFICATION", 20)
    )
    magnification_tolerance: float = field(
        default_factory=lambda: _env_float("WSI_MAG_TOLERANCE", 2.0)
    )
    max_white_background_fraction: float = field(
        default_factory=lambda: _env_float("WSI_MAX_WHITE_BG_FRACTION", 0.5)
    )
    white_pixel_gray_threshold: int = field(
        default_factory=lambda: _env_int("WSI_WHITE_GRAY_THRESHOLD", 220)
    )
    blur_laplacian_threshold: float = field(
        default_factory=lambda: _env_float("WSI_BLUR_LAPLACIAN_THRESHOLD", 1000.0)
    )
    keep_staging: bool = field(default_factory=lambda: _env_bool("WSI_KEEP_STAGING", False))

    def effective_tile_size(self) -> int:
        return max(16, self.tile_size - self.tile_size_subtract_legacy)


def load_wsi_pipeline_config(json_path: Path | None = None) -> WSIPipelineConfig:
    """Instancia configuração base e aplica overrides de um JSON opcional."""
    base = WSIPipelineConfig()
    if json_path is None or not Path(json_path).is_file():
        return base
    with open(json_path, encoding="utf-8") as fh:
        overrides = json.load(fh)
    if not isinstance(overrides, dict):
        raise ValueError("Ficheiro de configuração deve ser um objeto JSON (dict).")
    field_names = {f.name for f in fields(WSIPipelineConfig)}
    path_keys = {
        "wsi_csv",
        "template_image",
        "tiling_staging_dir",
        "processed_dataset_dir",
    }
    kwargs: dict = {}
    for k, v in overrides.items():
        if k not in field_names or v is None:
            continue
        if k in path_keys:
            kwargs[k] = Path(str(v)).expanduser()
        else:
            kwargs[k] = v
    return replace(base, **kwargs)


def read_wsi_manifest_rows(
    csv_path: Path,
    path_column: str,
    label_column: str,
) -> list[tuple[Path, str]]:
    """Lê CSV de WSI; aceita colunas nomeadas ou as duas primeiras colunas."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if path_column in df.columns and label_column in df.columns:
        paths = df[path_column]
        labels = df[label_column]
    elif len(df.columns) >= 2:
        paths = df.iloc[:, 0]
        labels = df.iloc[:, 1]
    else:
        raise ValueError(
            f"CSV deve ter colunas '{path_column}' e '{label_column}' ou pelo menos duas colunas."
        )
    rows: list[tuple[Path, str]] = []
    for p, lab in zip(paths, labels):
        if pd.isna(p) or pd.isna(lab):
            continue
        rows.append((Path(str(p)).expanduser(), str(lab).strip()))
    return rows
