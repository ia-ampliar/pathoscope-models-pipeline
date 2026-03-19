#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo "teste" para reconstruir a imagem sobreposta a partir do `.npz`
salvo pela inferência (`scripts/run_inference.py`) e da imagem original (WSI).
"""

# ---------- Limites de threads internas (ANTES de importar numpy/cv/tf) ----------
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# ------------------------------------------------------------------------------
import argparse
import os
from pathlib import Path

import numpy as np
import openslide
import tensorflow as tf
from matplotlib import cm
from PIL import Image


def _derive_image_id_and_base(image_path: str) -> tuple[str, str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    image_id = base.split(".", 1)[0]
    return image_id, base


def reconstruct_overlay_from_npz(
    npz_path: str,
    image_path: str,
    output_dir: str,
) -> Path:
    npz_path = str(npz_path)
    image_path = str(image_path)
    output_dir = str(output_dir)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ não encontrado: {npz_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    data = np.load(npz_path, allow_pickle=True)

    jet_heatmap_matrix = data["jet_heatmap_matrix"]
    overlay_level = int(data["overlay_level"])
    image_id_from_npz = str(data["image_id"])
    base_from_npz = str(data["base"])

    # Mantém a mesma lógica de identificação baseada no ID.
    image_id, base = _derive_image_id_and_base(image_path)
    # Se o .npz tiver info diferente por algum motivo, preferimos o que veio do .npz.
    image_id = image_id_from_npz or image_id
    base = base_from_npz or base

    image_output_dir = os.path.join(output_dir, f"{image_id}_results")
    os.makedirs(image_output_dir, exist_ok=True)

    # ====== reconstrói heatmap da matriz ======
    jet_heatmap_matrix_f16 = np.asarray(jet_heatmap_matrix, dtype=np.float16)
    # Mesma conversão usada no run_inference.py: escala por 255 e faz cast para uint8.
    # (Assumimos que a matriz está no intervalo esperado para não ocorrer wrap.)
    heatmap_matrix = np.uint8(255.0 * jet_heatmap_matrix_f16)
    jet_colors = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_matrix]
    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap)

    # ====== lê a WSI original para recorte no overlay_level ======
    wsi = openslide.OpenSlide(image_path)
    img = wsi.read_region(
        (0, 0),
        overlay_level,
        wsi.level_dimensions[overlay_level],
    ).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    wsi.close()

    # Redimensiona heatmap para as dimensões do recorte.
    jet_heatmap_img = jet_heatmap_img.resize((img_array.shape[1], img_array.shape[0]))
    superimposed_img = np.array(jet_heatmap_img, dtype=np.float32) + img_array
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    out_path = os.path.join(image_output_dir, f"superimposed_{base}_qat.jpg")
    superimposed_img.save(out_path, quality=100)
    return Path(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstrói overlay a partir de um .npz de heatmap.")
    parser.add_argument("--npz_path", required=True, help="Caminho do arquivo .npz com a matriz.")
    parser.add_argument("--image_path", required=True, help="Caminho da WSI original (svs).")
    parser.add_argument(
        "--output_dir",
        default=os.getenv("OUTPUT_DIR", "output"),
        help="Onde salvar o resultado reconstruído.",
    )
    args = parser.parse_args()

    out_path = reconstruct_overlay_from_npz(
        npz_path=args.npz_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
    )

    print(f"[OK] Superimposed reconstruído em: {out_path}")


if __name__ == "__main__":
    main()

