#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper simples para execução em Docker ou local.

Uso em Python:
    from scripts.infer import run
    run("/caminho/da/imagem.svs")

Uso em Docker (exemplo):
    docker run --rm \\
      -e IMAGE_PATH="/data/minha_imagem.svs" \\
      -v $(pwd)/saida:/app/output \\
      meu-modelo-inferencia
"""

import os

import scripts.run_inference as _ri
from scripts.run_inference import run as _run

# Garante que os arquivos gerados caiam em um diretório
# fácil de mapear como volume no host (por padrão: /app/output).
_ri.OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
os.makedirs(_ri.OUTPUT_DIR, exist_ok=True)


def run(image_path: str) -> None:
    """
    Executa a inferência para uma única imagem WSI.

    Parameters
    ----------
    image_path : str
        Caminho absoluto (no container ou host montado) da imagem a ser analisada.
    """
    _run(image_path)


def main() -> None:
    """
    Ponto de entrada quando chamado via CLI.
    Lê o caminho da imagem a partir da variável de ambiente IMAGE_PATH.
    """
    image_path = os.getenv("IMAGE_PATH", "").strip()
    if not image_path:
        raise SystemExit(
            "IMAGE_PATH não definido. "
            "Defina IMAGE_PATH ou chame scripts.infer.run('/caminho/da/imagem.svs') diretamente."
        )
    run(image_path)


if __name__ == "__main__":
    main()

