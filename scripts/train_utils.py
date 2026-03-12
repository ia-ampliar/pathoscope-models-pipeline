import json
import os
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model

from . import config


def save_history(history: tf.keras.callbacks.History, stage_name: str, checkpoint_path: str) -> Path:
    """Salva histórico de treinamento em JSON e retorna o caminho."""
    stage_dir = config.METRICS_DIR / stage_name
    os.makedirs(stage_dir, exist_ok=True)

    history_data = {}
    for key, values in history.history.items():
        history_data[key] = [float(v) for v in values]

    history_path = stage_dir / f"{Path(checkpoint_path).stem}_history.json"
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=4)

    return history_path


def plot_history(
    history: tf.keras.callbacks.History,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plota curvas de loss/acurácia e opcionalmente salva o gráfico."""
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Loss (treino)")
    plt.plot(history.history["val_loss"], label="Loss (validação)")
    plt.title(title + " - Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Acc (treino)")
    plt.plot(history.history["val_accuracy"], label="Acc (validação)")
    plt.title(title + " - Acurácia")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def train_with_callbacks(
    model: tf.keras.Model,
    train_gen,
    val_gen,
    epochs: int,
    checkpoint_path: str,
    stage_name: str,
    early_stop_patience: int,
    strategy: Optional[tf.distribute.Strategy] = None,
) -> Tuple[tf.keras.callbacks.History, str]:
    """Treina o modelo com callbacks e salva o melhor checkpoint e um artefato final."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    if strategy is not None:
        with strategy.scope():
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
    else:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

    final_path = os.path.join(
        str(config.MODELS_DIR),
        os.path.basename(checkpoint_path).replace(".keras", "_final.keras"),
    )

    try:
        best = load_model(checkpoint_path, compile=False)
        best.save(final_path)
    except Exception:
        model.save(final_path)

    save_history(history, stage_name, checkpoint_path)

    return history, final_path

