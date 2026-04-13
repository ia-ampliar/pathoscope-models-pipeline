import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model

import modules.config as config


def append_metrics_jsonl(path: str, payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_streaming_metrics_callback(
    stage_name: str,
    stream_path: str,
    overfitting_ratio_threshold: float = 1.6,
) -> tf.keras.callbacks.Callback:
    """Callback que grava uma linha JSON por época para telemetria (WebSocket no servidor)."""

    class StreamingMetricsCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None) -> None:
            append_metrics_jsonl(
                stream_path,
                {"type": "stage_start", "stage": stage_name},
            )

        def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
            logs = logs or {}
            metrics = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            warning: Optional[dict[str, Any]] = None
            loss = metrics.get("loss")
            val_loss = metrics.get("val_loss")
            if (
                loss is not None
                and val_loss is not None
                and loss > 1e-8
                and val_loss / loss >= overfitting_ratio_threshold
            ):
                warning = {
                    "type": "overfitting_risk",
                    "severity": "warn",
                    "ratio": val_loss / loss,
                }
            payload: dict[str, Any] = {
                "type": "epoch",
                "stage": stage_name,
                "epoch": epoch + 1,
                "metrics": metrics,
            }
            if warning:
                payload["warning"] = warning
            append_metrics_jsonl(stream_path, payload)

    return StreamingMetricsCallback()


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
    extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    metrics_stream_path: Optional[str] = None,
    overfitting_ratio_threshold: float = 1.6,
) -> Tuple[tf.keras.callbacks.History, str]:
    """Treina o modelo com callbacks e salva o melhor checkpoint e um artefato final."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks: List[tf.keras.callbacks.Callback] = [
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
    if metrics_stream_path:
        callbacks.append(
            make_streaming_metrics_callback(
                stage_name,
                metrics_stream_path,
                overfitting_ratio_threshold=overfitting_ratio_threshold,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

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

