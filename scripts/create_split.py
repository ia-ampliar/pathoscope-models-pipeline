#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para criar os arquivos de split (label_file, train_data, val_data, test_data)
a partir de dados já pré-processados.

Etapa posterior ao pré-processamento: organiza o split e referencia caminhos em CSVs.
Integra-se ao pipeline de treinamento como etapa anterior ao train.py.

Estrutura esperada do diretório de dados:
    dataset_path/
        classe_1/
            imagem1.jpeg
            imagem2.jpeg
        classe_2/
            ...
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# Extensões de imagem suportadas
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}


def create_label_file(
    dataset_path: str | Path,
    output_csv: str | Path,
    image_root: Optional[str | Path] = None,
    extensions: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Cria label_file.csv a partir de um diretório de imagens organizado por classe.

    Assume que os dados já passaram pelo pré-processamento. Cada subdiretório
    representa uma classe; os arquivos dentro são as amostras.

    Args:
        dataset_path: Caminho para o diretório do conjunto de dados (subdirs = classes).
        output_csv: Caminho para salvar label_file.csv.
        image_root: Base para caminhos relativos. Se None, usa caminhos absolutos.
        extensions: Extensões aceitas. Se None, usa IMAGE_EXTENSIONS.

    Returns:
        DataFrame com colunas Image_path, Label.
    """
    dataset_path = Path(dataset_path)
    output_csv = Path(output_csv)
    extensions = extensions or IMAGE_EXTENSIONS

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_path}")

    data = []
    for label in sorted(dataset_path.iterdir()):
        if not label.is_dir():
            continue
        for image_file in sorted(label.iterdir()):
            if image_file.suffix.lower() not in extensions:
                continue
            image_path = image_file.resolve()
            if image_root is not None:
                try:
                    image_path = os.path.relpath(image_path, start=image_root)
                    image_path = str(image_path).replace("\\", "/")
                except ValueError:
                    image_path = str(image_path)
            else:
                image_path = str(image_path)
            data.append({"Image_path": image_path, "Label": label.name})

    df = pd.DataFrame(data)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] label_file.csv criado em: {output_csv} ({len(df)} amostras)")

    return df


def create_splits(
    label_csv: str | Path,
    output_dir: str | Path,
    train_csv_name: str = "train_data.csv",
    val_csv_name: str = "val_data.csv",
    test_csv_name: str = "test_data.csv",
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: Optional[int] = None,
    path_column: str = "Image_path",
    label_column: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega label_file.csv, divide em train/val/test de forma estratificada
    e salva os CSVs no formato esperado pelo dataloader (image_path, lesion_type).

    Args:
        label_csv: Caminho para label_file.csv (Image_path, Label).
        output_dir: Diretório para salvar train_data, val_data, test_data.
        train_csv_name, val_csv_name, test_csv_name: Nomes dos arquivos.
        test_size: Proporção para teste+val (ex: 0.3 = 30%).
        val_size: Do conjunto test+val, proporção para val (ex: 0.5 = 50% val, 50% test).
        random_state: Semente para reprodutibilidade.
        path_column: Nome da coluna de caminhos no label_csv.
        label_column: Nome da coluna de rótulos no label_csv.

    Returns:
        (train_df, val_df, test_df) com colunas lesion_type, image_path.
    """
    label_csv = Path(label_csv)
    output_dir = Path(output_dir)
    random_state = random_state if random_state is not None else config.RANDOM_SEED

    if not label_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {label_csv}")

    df = pd.read_csv(label_csv)

    if path_column not in df.columns:
        raise ValueError(f"Coluna '{path_column}' não encontrada. Colunas: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Coluna '{label_column}' não encontrada. Colunas: {list(df.columns)}")

    # Padroniza para o formato do dataloader: image_path, lesion_type
    df = df.rename(columns={path_column: "image_path", label_column: "lesion_type"})
    df = df[["lesion_type", "image_path"]]  # ordem: lesion_type primeiro (como nos CSVs existentes)

    # Garante caminhos com barras corretas
    df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/")

    n_total = len(df)
    print(f"Total de amostras: {n_total}")
    print("Distribuição de classes:")
    print(df["lesion_type"].value_counts().to_string())

    # Split estratificado: train (70%) | temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["lesion_type"],
    )

    # temp em val (15%) e test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        stratify=temp_df["lesion_type"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / train_csv_name
    val_path = output_dir / val_csv_name
    test_path = output_dir / test_csv_name

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[OK] Splits salvos em {output_dir}:")
    print(f"  - {train_csv_name}: {len(train_df)} amostras")
    print(f"  - {val_csv_name}: {len(val_df)} amostras")
    print(f"  - {test_csv_name}: {len(test_df)} amostras")

    return train_df, val_df, test_df


def run_pipeline(
    dataset_path: str | Path,
    output_dir: Optional[str | Path] = None,
    image_root: Optional[str | Path] = None,
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: Optional[int] = None,
    use_relative_paths: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa o pipeline completo: cria label_file e em seguida os splits.

    Integra-se como etapa anterior ao treinamento. Os dados devem estar
    pré-processados em dataset_path (subdirs = classes).

    Args:
        dataset_path: Diretório com subdirs por classe.
        output_dir: Onde salvar os CSVs. Se None, usa config.SPLIT_DIR.
        image_root: Base para caminhos relativos. Se None e use_relative_paths,
                    usa dataset_path como base.
        test_size: Proporção para teste+val.
        val_size: Proporção de val dentro de test+val.
        random_state: Semente.
        use_relative_paths: Se True, salva caminhos relativos a image_root
                            (portabilidade e consistência com config.DATA_DIR).

    Returns:
        (train_df, val_df, test_df).
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir else config.SPLIT_DIR
    image_root = Path(image_root) if image_root else (dataset_path if use_relative_paths else None)

    label_csv = output_dir / "label_file.csv"

    print("[1/2] Criando label_file.csv...")
    create_label_file(
        dataset_path=dataset_path,
        output_csv=label_csv,
        image_root=image_root,
    )

    print("\n[2/2] Criando splits (train/val/test)...")
    train_df, val_df, test_df = create_splits(
        label_csv=label_csv,
        output_dir=output_dir,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state or config.RANDOM_SEED,
    )

    print("\n[CONCLUÍDO] Pipeline de split finalizado.")
    print("Próximo passo: python -m src.train")
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria label_file e splits (train/val/test) a partir de dados pré-processados."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Diretório com subdirs por classe (ex: datas/ ou pre-processado/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Diretório para os CSVs. Padrão: {config.SPLIT_DIR}",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Base para caminhos relativos. Padrão: dataset_path.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Proporção para teste+val (padrão: 0.3).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.5,
        help="Do conjunto test+val, proporção para val (padrão: 0.5).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help=f"Semente. Padrão: {config.RANDOM_SEED}",
    )
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Usar caminhos absolutos em vez de relativos.",
    )
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        image_root=args.image_root,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        use_relative_paths=not args.absolute_paths,
    )


if __name__ == "__main__":
    main()
