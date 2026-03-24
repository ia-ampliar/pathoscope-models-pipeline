#!/usr/bin/env python3
"""
Orquestra tiling (OpenSlide + DeepZoom) e pré-processamento (Macenko, blur, fundo).

Saída final em ``processed_dataset_dir`` (por omissão ``datas/``), no formato esperado
por ``create_split.create_label_file``: ``<classe>/<ficheiros de imagem>``.

Exemplo::

    python scripts/build_wsi_dataset.py --config configs/wsi_pipeline.example.json

Requer biblioteca OpenSlide no sistema (Linux: libopenslide0; ver Dockerfile do projeto).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Garante raiz do repositório em sys.path quando se executa ``python scripts/...``
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import config as app_config
from scripts import pre_processing, tile
from scripts.wsi_pipeline_config import WSIPipelineConfig, load_wsi_pipeline_config


def _resolve_cfg_paths(cfg: WSIPipelineConfig, root: Path) -> WSIPipelineConfig:
    from dataclasses import replace

    def r(p: Path) -> Path:
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    return replace(
        cfg,
        wsi_csv=r(cfg.wsi_csv),
        template_image=r(cfg.template_image),
        tiling_staging_dir=r(cfg.tiling_staging_dir),
        processed_dataset_dir=r(cfg.processed_dataset_dir),
    )


def _merge_cli(cfg: WSIPipelineConfig, args: argparse.Namespace) -> WSIPipelineConfig:
    from dataclasses import replace

    kwargs = {}
    if args.wsi_csv is not None:
        kwargs["wsi_csv"] = Path(args.wsi_csv).expanduser()
    if args.template_image is not None:
        kwargs["template_image"] = Path(args.template_image).expanduser()
    if args.tiling_staging_dir is not None:
        kwargs["tiling_staging_dir"] = Path(args.tiling_staging_dir).expanduser()
    if args.processed_dataset_dir is not None:
        kwargs["processed_dataset_dir"] = Path(args.processed_dataset_dir).expanduser()
    return replace(cfg, **kwargs) if kwargs else cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline WSI: tiling + pré-processamento.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON com overrides (ver configs/wsi_pipeline.example.json).",
    )
    parser.add_argument("--wsi-csv", type=Path, default=None, help="CSV de lâminas (override).")
    parser.add_argument(
        "--template-image",
        type=Path,
        default=None,
        help="Imagem template Macenko (override; padrão: img_template.png na raiz).",
    )
    parser.add_argument("--tiling-staging-dir", type=Path, default=None)
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=None,
        help="Destino final (padrão: datas/ — deve coincidir com image_root do treino).",
    )
    parser.add_argument("--tile-only", action="store_true", help="Apenas gerar tiles no staging.")
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Apenas pré-processar tiles já existentes no staging.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.tile_only and args.preprocess_only:
        parser.error("Use apenas um de --tile-only ou --preprocess-only.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_wsi_pipeline_config(args.config)
    cfg = _resolve_cfg_paths(cfg, app_config.ROOT_DIR)
    cfg = _merge_cli(cfg, args)
    cfg = _resolve_cfg_paths(cfg, app_config.ROOT_DIR)

    if not cfg.wsi_csv.is_file() and not args.preprocess_only:
        parser.error(f"CSV de WSI não encontrado: {cfg.wsi_csv}")

    if not args.preprocess_only:
        rows = tile.tile_from_manifest(cfg)
        logging.info("Tiling concluído para %d entradas.", len(rows))

    if not args.tile_only:
        n = pre_processing.pre_process_staging_to_dataset(
            cfg,
            progress=not args.no_progress,
        )
        logging.info("Pré-processamento: %d imagens gravadas em %s", n, cfg.processed_dataset_dir)

    print(
        "\nPróximo passo (splits para o treino):\n"
        f"  python -m scripts.create_split {cfg.processed_dataset_dir}\n"
        "Certifique-se de que ``processed_dataset_dir`` é o mesmo que ``config.DATA_DIR`` "
        "usado pelo ``dataloader`` (por omissão, a pasta ``datas/``)."
    )


if __name__ == "__main__":
    main()
