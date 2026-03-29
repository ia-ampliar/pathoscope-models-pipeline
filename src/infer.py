#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper simples para execução em Docker ou local.

Uso em Python:
    from modules.inference import run
    run("/caminho/da/imagem.svs")

Uso em Docker (exemplo):
    docker run --rm \\
      -e IMAGE_PATH="/data/minha_imagem.svs" \\
      -v $(pwd)/saida:/app/output \\
      meu-modelo-inferencia
"""

import os
import sys
from pathlib import Path

# Garante que o diretório raiz do projeto esteja no path ao rodar python src/infer.py
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse

import modules.inference.run_inference as _ri
from modules.inference.run_inference import run as _run


def _default_output_dir() -> str:
    """Detecta se está em Docker e retorna o OUTPUT_DIR padrão adequado."""
    if os.getenv("OUTPUT_DIR"):
        return os.getenv("OUTPUT_DIR")
    # Dentro do Docker: /app existe e é o diretório de trabalho
    if Path("/app").exists():
        return "/app/output"
    # Fora do Docker: salva em output/ na raiz do projeto
    return str(_project_root / "output")


# Garante que os arquivos gerados caiam no diretório correto
_ri.OUTPUT_DIR = _default_output_dir()
os.makedirs(_ri.OUTPUT_DIR, exist_ok=True)


def run(image_path: str, output_mode: str = "overlay") -> None:
    """
    Executa a inferência para uma única imagem WSI.

    Parameters
    ----------
    image_path : str
        Caminho absoluto (no container ou host montado) da imagem a ser analisada.
    """
    _run(image_path, output_mode=output_mode)


def main() -> None:
    """
    Ponto de entrada quando chamado via CLI.
    Lê o caminho da imagem a partir da variável de ambiente IMAGE_PATH.
    """
    parser = argparse.ArgumentParser(description="Inferência + geração de heatmap")
    parser.add_argument(
        "--image_path",
        default=os.getenv("IMAGE_PATH", "").strip(),
        help="Caminho absoluto para a WSI (svs). Pode usar env var IMAGE_PATH.",
    )
    parser.add_argument(
        "--output_mode",
        default=os.getenv("OUTPUT_HEATMAP_MODE", "overlay"),
        choices=["overlay", "matrix_only"],
        help="overlay gera imagens (JPG) + NPZ. matrix_only gera apenas NPZ.",
    )
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        raise SystemExit(
            "image_path/IMAGE_PATH não definido. "
            "Defina IMAGE_PATH ou use --image_path."
        )

    run(image_path=image_path, output_mode=args.output_mode)


if __name__ == "__main__":
    main()

