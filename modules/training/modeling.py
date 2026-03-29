from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import modules.config as config


def build_baseline_model(
    input_shape: tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = 2,
    base_trainable: bool = False,
    learning_rate: float = config.LR_BASELINE,
) -> Tuple[Model, tf.keras.Model]:
    """Constrói o modelo baseline (MobileNetV2 + topo denso)."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = base_trainable

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base


def enable_fine_tuning(base_model: tf.keras.Model, fine_tune_percent: float) -> None:
    """Descongela a fração final das camadas convolucionais para fine-tuning."""
    if fine_tune_percent <= 0.0:
        return

    total_layers = len(base_model.layers)
    fine_tune_at = int(total_layers * (1.0 - fine_tune_percent))

    for i, layer in enumerate(base_model.layers):
        if i >= fine_tune_at and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True
        else:
            layer.trainable = False

