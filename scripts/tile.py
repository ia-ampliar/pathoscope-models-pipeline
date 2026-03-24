"""
Tiling de WSI com OpenSlide e DeepZoomGenerator (openslide-python).

Gera patches num diretório de staging por classe e lâmina, no formato
``<basename>_files/<magnificação>/<col>_<row>.<fmt>``, alinhado ao legado
usado com CSVs de caminhos de pastas de tiles.
"""

from __future__ import annotations

import logging
import re
import sys
from multiprocessing import JoinableQueue, Process
from pathlib import Path
from typing import Optional

import numpy as np
import openslide
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

from .wsi_pipeline_config import WSIPipelineConfig, read_wsi_manifest_rows

logger = logging.getLogger(__name__)


def _safe_label_component(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s[:200] if s else "class"


def white_background_fraction(
    pil_img: Image.Image,
    gray_threshold: int = 220,
) -> float:
    """Fração de pixels com luminância >= threshold (fundo claro)."""
    gray = pil_img.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    return float(np.mean(arr >= gray_threshold))


class TileWorker(Process):
    """Processo filho que gera tiles a partir da fila (um ficheiro por tarefa)."""

    def __init__(
        self,
        queue: JoinableQueue,
        slidepath: str,
        tile_size: int,
        overlap: int,
        limit_bounds: bool,
        quality: int,
        gray_threshold: int,
        max_white_fraction: float,
    ):
        super().__init__(name="TileWorker", daemon=True)
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._gray_threshold = gray_threshold
        self._max_white_fraction = max_white_fraction
        self._slide = None

    def run(self) -> None:
        self._slide = open_slide(self._slidepath)
        last_associated: Optional[str] = None
        dz: Optional[DeepZoomGenerator] = None
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if dz is None or associated != last_associated:
                dz = self._get_dz(associated)
                last_associated = associated
            tile = dz.get_tile(level, address)
            frac = white_background_fraction(tile, self._gray_threshold)
            if frac <= self._max_white_fraction:
                Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                tile.save(outfile, quality=self._quality)
            self._queue.task_done()

    def _get_dz(self, associated: Optional[str]) -> DeepZoomGenerator:
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(
            image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds
        )


def _magnification_for_level(
    slide: openslide.OpenSlide,
    dz: DeepZoomGenerator,
    level: int,
) -> float:
    factors = slide.level_downsamples
    try:
        objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            "WSI sem OBJECTIVE_POWER nas propriedades; não é possível escolher a magnificação."
        ) from e
    available0 = objective / factors[0]
    depth = dz.level_count - (level + 1)
    return float(available0 / (2**depth))


def _best_level_for_target_mag(
    slide: openslide.OpenSlide,
    dz: DeepZoomGenerator,
    target_mag: float,
    tolerance: float,
) -> Optional[tuple[int, int]]:
    """Escolhe um único nível DeepZoom cuja magnificação está mais próxima do alvo."""
    best: Optional[tuple[int, int, float]] = None
    for level in range(dz.level_count - 1, -1, -1):
        mag = _magnification_for_level(slide, dz, level)
        diff = abs(mag - target_mag)
        if diff <= tolerance and (best is None or diff < best[2]):
            best = (level, int(round(mag)), diff)
    if best is None:
        return None
    return best[0], best[1]


def tile_one_slide(
    slide_path: Path,
    label: str,
    cfg: WSIPipelineConfig,
) -> tuple[Path, str]:
    """
    Gera tiles para uma lâmina.

    Returns:
        (caminho da pasta de tiles ``..._files/<mag>``, nome da magnificação como string)
    """
    slide_path = slide_path.resolve()
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)

    label_dir = cfg.tiling_staging_dir / _safe_label_component(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    basename = label_dir / slide_path.stem

    fmt = cfg.image_format.lstrip(".")
    eff_tile = cfg.effective_tile_size()
    queue: JoinableQueue = JoinableQueue(max(2, 2 * cfg.workers))
    workers: list[TileWorker] = []
    for _ in range(cfg.workers):
        w = TileWorker(
            queue,
            str(slide_path),
            eff_tile,
            cfg.overlap,
            cfg.limit_bounds,
            cfg.jpeg_quality,
            cfg.white_pixel_gray_threshold,
            cfg.max_white_background_fraction,
        )
        w.start()
        workers.append(w)

    slide = open_slide(str(slide_path))
    try:
        dz = DeepZoomGenerator(
            slide, eff_tile, cfg.overlap, limit_bounds=cfg.limit_bounds
        )
        picked = _best_level_for_target_mag(
            slide, dz, float(cfg.target_magnification), cfg.magnification_tolerance
        )
        if picked is None:
            raise RuntimeError(
                f"Nenhum nível DeepZoom com magnificação ~{cfg.target_magnification} "
                f"(±{cfg.magnification_tolerance}) para {slide_path}"
            )
        level, mag_int = picked
        tile_dir = Path(f"{basename}_files") / str(mag_int)
        tile_dir.mkdir(parents=True, exist_ok=True)
        cols, rows = dz.level_tiles[level]
        total = cols * rows
        enq = 0
        for row in range(rows):
            for col in range(cols):
                tile_name = tile_dir / f"{col}_{row}.{fmt}"
                if tile_name.is_file():
                    continue
                queue.put((None, level, (col, row), str(tile_name)))
                enq += 1
                if enq % 200 == 0:
                    print(
                        f"Tiling {slide_path.name}: fila {enq} (nível {total} células)",
                        end="\r",
                        file=sys.stderr,
                    )
        print(file=sys.stderr)

        for _ in workers:
            queue.put(None)
        queue.join()
        for w in workers:
            w.join(timeout=600)

        return tile_dir, str(mag_int)
    finally:
        slide.close()


def tile_from_manifest(cfg: WSIPipelineConfig) -> list[tuple[str, str, str]]:
    """
    Processa todas as linhas do CSV configurado.

    Returns:
        Lista de (caminho da pasta de tiles para CSV, label, magnificação)
    """
    rows = read_wsi_manifest_rows(cfg.wsi_csv, cfg.wsi_path_column, cfg.label_column)
    out_rows: list[tuple[str, str, str]] = []
    for slide_path, label in rows:
        try:
            tile_dir, mag = tile_one_slide(slide_path, label, cfg)
            logger.info("Tiles: %s -> %s", slide_path, tile_dir)
            out_rows.append((str(tile_dir), label, mag))
        except Exception as e:
            logger.error("Falha ao tile %s: %s", slide_path, e)
    return out_rows
