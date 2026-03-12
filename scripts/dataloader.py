from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from . import config


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Carrega os CSVs de treino/val/test e retorna também o número de classes."""
    train_csv, val_csv, test_csv = config.get_train_val_test_csv_paths()

    try:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        num_classes = train_df["lesion_type"].nunique()
    except Exception:
        # Fallback defensivo: datasets vazios e 3 classes padrão
        train_df = pd.DataFrame({"image_path": [], "lesion_type": []})
        val_df = train_df.copy()
        test_df = train_df.copy()
        num_classes = 3

    return train_df, val_df, test_df, num_classes


def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_root: Path | None = None,
    augment: bool = False,
    batch_size: int = config.BATCH_SIZE,
    img_size: tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
):
    """Cria geradores de imagens para treino, validação e teste."""
    if augment:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    common_kwargs = dict(
        x_col="image_path",
        y_col="lesion_type",
        target_size=img_size,
        class_mode="categorical",
    )

    directory = str(image_root) if image_root is not None else None

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=directory,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=directory,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    num_classes = len(train_generator.class_indices)

    return train_generator, val_generator, test_generator, num_classes

