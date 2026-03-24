"""
Pré-processamento de tiles: normalização Macenko (staintools), rejeição por desfoque
(Laplaciano) e filtro opcional de fundo branco.

Saída em ``datas/<classe>/`` (ou ``processed_dataset_dir``), estrutura esperada por
``create_split.py`` e pelo ``dataloader`` (caminhos relativos a ``config.DATA_DIR``).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2 as cv
import numpy as np
import staintools
from tqdm import tqdm

from . import config
from .wsi_pipeline_config import WSIPipelineConfig

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}


def _safe_label_component(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s[:200] if s else "class"


def variance_of_laplacian_bgr(image_bgr: np.ndarray) -> float:
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def white_background_fraction_bgr(image_bgr: np.ndarray, gray_threshold: int) -> float:
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    return float(np.mean(gray >= gray_threshold))


def _stem_from_files_dir(files_dir: Path) -> str:
    name = files_dir.name
    if name.endswith("_files"):
        return name[: -len("_files")]
    return files_dir.name


def _iter_staging_tiles(staging_root: Path) -> Iterable[tuple[Path, str, str, Path]]:
    """
    Percorre staging ``<staging>/<label>/<stem>_files/<mag>/*``.

    Yields:
        (ficheiro tile, label, stem da lâmina, pasta mag)
    """
    if not staging_root.is_dir():
        return
    for label_dir in sorted(staging_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for tile_root in sorted(label_dir.glob("*_files")):
            if not tile_root.is_dir():
                continue
            stem = _stem_from_files_dir(tile_root)
            for mag_dir in sorted(p for p in tile_root.iterdir() if p.is_dir()):
                for img_path in sorted(mag_dir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    yield img_path, label, stem, mag_dir


def _build_macenko_normalizer(template_path: Path) -> Any:
    template_path = template_path.resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template Macenko não encontrado: {template_path}")
    target = staintools.read_image(str(template_path))
    if target is None:
        raise ValueError(f"Falha ao ler template: {template_path}")
    normalizer = staintools.StainNormalizer(method="macenko")
    normalizer.fit(target)
    return normalizer


def process_tile_image(
    src_path: Path,
    dst_path: Path,
    normalizer: Any,
    cfg: WSIPipelineConfig,
) -> bool:
    """
    Aplica Macenko + filtros. Retorna True se o ficheiro foi gravado.
    """
    image_bgr = cv.imread(str(src_path))
    if image_bgr is None:
        logger.warning("Não foi possível ler: %s", src_path)
        return False

    if (
        white_background_fraction_bgr(image_bgr, cfg.white_pixel_gray_threshold)
        > cfg.max_white_background_fraction
    ):
        return False

    fm = variance_of_laplacian_bgr(image_bgr)
    if fm <= cfg.blur_laplacian_threshold:
        return False

    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    image_rgb = staintools.LuminosityStandardizer.standardize(image_rgb)
    try:
        out_rgb = normalizer.transform(image_rgb)
    except Exception as e:
        logger.warning("Macenko falhou em %s: %s", src_path, e)
        return False

    out_bgr = cv.cvtColor(out_rgb, cv.COLOR_RGB2BGR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(dst_path), out_bgr):
        logger.warning("Falha ao gravar: %s", dst_path)
        return False
    return True


def pre_process_staging_to_dataset(
    cfg: WSIPipelineConfig,
    progress: bool = True,
    normalizer_factory: Callable[[], Any] | None = None,
) -> int:
    """
    Lê tiles do staging, aplica pré-processamento e grava em ``processed_dataset_dir``.

    Cada imagem final fica em
    ``processed_dataset_dir/<label>/<stem>_m<mag>_c<col>_r<row>.<ext>``.

    Returns:
        Número de imagens gravadas.
    """
    staging = cfg.tiling_staging_dir.resolve()
    out_root = cfg.processed_dataset_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    factory = normalizer_factory or (lambda: _build_macenko_normalizer(cfg.template_image))
    normalizer = factory()

    count = 0
    tiles = list(_iter_staging_tiles(staging))
    if not tiles:
        logger.warning("Nenhum tile encontrado em %s (verifique o tiling).", staging)

    iterator: Iterable[tuple[Path, str, str, Path]] = tiles
    if progress:
        iterator = tqdm(tiles, desc="Pré-processamento")

    for src_path, label, stem, mag_dir in iterator:
        safe_lab = _safe_label_component(label)
        m = re.match(r"^(\d+)_(\d+)\.", src_path.name)
        if not m:
            logger.debug("Ignorar nome não padrão col_row: %s", src_path.name)
            continue
        col, row = m.group(1), m.group(2)
        mag_name = mag_dir.name
        out_name = f"{stem}_m{mag_name}_c{col}_r{row}{src_path.suffix.lower()}"
        dst_path = out_root / safe_lab / out_name

        if dst_path.is_file():
            continue

        if process_tile_image(src_path, dst_path, normalizer, cfg):
            count += 1

    if not cfg.keep_staging and tiles and staging.is_dir():
        try:
            shutil.rmtree(staging, ignore_errors=False)
            logger.info("Staging removido: %s", staging)
        except OSError as e:
            logger.warning("Não foi possível remover staging %s: %s", staging, e)

    return count


def pre_process_single_file(
    src_path: Path,
    dst_path: Path,
    cfg: WSIPipelineConfig,
) -> bool:
    """API pontual para um tile (útil em testes)."""
    n = _build_macenko_normalizer(cfg.template_image)
    return process_tile_image(src_path, dst_path, n, cfg)


def relative_paths_for_dataset(
    image_paths: Iterable[Path],
    image_root: Path | None = None,
) -> list[str]:
    """Converte caminhos absolutos em relativos a ``image_root`` (padrão: DATA_DIR)."""
    root = (image_root or config.DATA_DIR).resolve()
    rels = []
    for p in image_paths:
        p = p.resolve()
        try:
            rels.append(str(p.relative_to(root)).replace("\\", "/"))
        except ValueError:
            rels.append(str(p).replace("\\", "/"))
    return rels
