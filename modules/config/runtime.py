"""
Aplicação de overrides de hiperparâmetros em `modules.config` antes do treino.

O pipeline importa `import modules.config as config` e lê atributos no momento do uso;
atualizar este módulo após import é suficiente para o subprocesso de treino.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import modules.config as config


_TRAINING_KEYS = frozenset(
    {
        "RANDOM_SEED",
        "BATCH_SIZE",
        "EPOCHS_BASELINE",
        "INITIAL_EPOCHS_FRACTION",
        "FINE_TUNE_PERCENT",
        "LR_BASELINE",
        "LR_FINE",
        "LR_QAT",
        "PATIENCE_BASELINE",
        "PATIENCE_QAT",
        "IMG_HEIGHT",
        "IMG_WIDTH",
        "INPUT_SHAPE",
    }
)

_PATH_KEYS = frozenset({"DATA_DIR", "SPLIT_DIR", "MODELS_DIR", "METRICS_DIR", "TFLITE_DIR"})


def apply_training_overrides(overrides: Mapping[str, Any]) -> None:
    """Atualiza atributos conhecidos em `modules.config` a partir de um dict (ex.: JSON)."""
    for key, value in overrides.items():
        if key not in _TRAINING_KEYS and key not in _PATH_KEYS:
            continue
        if value is None:
            continue
        if key in _PATH_KEYS:
            setattr(config, key, Path(value))
        elif key == "INPUT_SHAPE" and isinstance(value, (list, tuple)) and len(value) == 3:
            setattr(config, key, (int(value[0]), int(value[1]), int(value[2])))
        else:
            setattr(config, key, value)

    # INPUT_SHAPE derivado de IMG_* se só um dos dois mudou
    if "INPUT_SHAPE" not in overrides or overrides.get("INPUT_SHAPE") is None:
        h = getattr(config, "IMG_HEIGHT", 224)
        w = getattr(config, "IMG_WIDTH", 224)
        config.INPUT_SHAPE = (int(h), int(w), 3)
