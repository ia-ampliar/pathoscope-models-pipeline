from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import quantize_scope
import tensorflow_model_optimization as tfmot

import modules.config as config

from modules.training.train_utils import make_streaming_metrics_callback


def load_baseline_for_qat(baseline_final_path: Path) -> tf.keras.Model:
    """Carrega o modelo baseline treinado para ser quantizado."""
    with quantize_scope():
        # baseline_final não é quantizado ainda, mas o escopo é seguro
        model = load_model(str(baseline_final_path), compile=False)
    return model


def apply_qat(
    baseline_model: tf.keras.Model,
) -> tf.keras.Model:
    """Aplica Quantization Aware Training a um modelo previamente treinado."""
    quantized_model = tfmot.quantization.keras.quantize_model(baseline_model)
    quantized_model.compile(
        optimizer=Adam(learning_rate=config.LR_QAT),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return quantized_model


def train_qat(
    quantized_model: tf.keras.Model,
    train_gen,
    val_gen,
    epochs: int = 15,
    metrics_stream_path: Optional[str] = None,
    overfitting_ratio_threshold: float = 1.6,
    extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> Path:
    """Treina o modelo quantizado e retorna o caminho do melhor checkpoint."""
    checkpoint_path = config.MODELS_DIR / "qat_baseline_checkpoint.keras"

    callbacks: List[tf.keras.callbacks.Callback] = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE_QAT,
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
                "qat",
                metrics_stream_path,
                overfitting_ratio_threshold=overfitting_ratio_threshold,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    quantized_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return checkpoint_path


def export_qat_to_tflite(qat_checkpoint_path: Path) -> Path:
    """Converte o melhor modelo QAT para TFLite e retorna o caminho salvo."""
    with quantize_scope():
        qat_model = load_model(str(qat_checkpoint_path), compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = config.MODELS_DIR / "qat_baseline_final.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    return tflite_path

