"""Processo filho: executa pipeline de tiling WSI (OpenSlide + pré-processamento)."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python tiling_worker.py <config.json>", file=sys.stderr)
        sys.exit(2)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)

    from modules.wsi_pipeline import pre_processing, tile
    from modules.wsi_pipeline.wsi_pipeline_config import WSIPipelineConfig
    from server.schemas import TilingJobConfig

    tc = TilingJobConfig.model_validate(cfg)

    wsi_csv = Path(tc.wsi_csv) if not Path(tc.wsi_csv).is_absolute() else Path(tc.wsi_csv)
    if not wsi_csv.is_absolute():
        wsi_csv = (root / wsi_csv).resolve()

    processed_dir = Path(tc.processed_dataset_dir)
    if not processed_dir.is_absolute():
        processed_dir = (root / processed_dir).resolve()

    pipeline_cfg = WSIPipelineConfig(
        wsi_csv=wsi_csv,
        processed_dataset_dir=processed_dir,
        wsi_path_column=tc.wsi_path_column,
        label_column=tc.label_column,
        tile_size=tc.tile_size,
        overlap=tc.overlap,
        target_magnification=tc.target_magnification,
        max_white_background_fraction=tc.max_white_background_fraction,
        blur_laplacian_threshold=tc.blur_laplacian_threshold,
        workers=tc.workers,
        jpeg_quality=tc.jpeg_quality,
        keep_staging=tc.keep_staging,
    )

    rows = tile.tile_from_manifest(pipeline_cfg)
    logging.info("Tiling concluído para %d entradas.", len(rows))

    n = pre_processing.pre_process_staging_to_dataset(pipeline_cfg, progress=True)
    logging.info("Pré-processamento: %d imagens gravadas em %s", n, pipeline_cfg.processed_dataset_dir)


if __name__ == "__main__":
    main()
