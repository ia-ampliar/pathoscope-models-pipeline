"""
Modelos Pydantic espelhando settings, split e inferência — fonte de verdade para GET /api/schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TrainingJobConfig(BaseModel):
    """Hiperparâmetros e flags do pipeline em src.train.run_training_pipeline."""

    use_strategy: bool = Field(True, description="Usar MirroredStrategy quando houver GPU.")
    use_qat: bool = Field(True, description="Executar QAT e exportar TFLite após o fine-tune.")
    augment: bool = Field(False, description="Ativar augmentação leve no ImageDataGenerator de treino.")

    random_seed: int = Field(42, ge=0, le=2**31 - 1, description="Semente global NumPy/TensorFlow.")
    batch_size: int = Field(32, ge=1, le=512, description="Tamanho do batch.")
    epochs_baseline: int = Field(100, ge=1, le=2000, description="Épocas totais planejadas (baseline + fine-tune).")
    initial_epochs_fraction: float = Field(
        0.4,
        ge=0.05,
        le=0.95,
        description="Fração de epochs_baseline para a fase baseline (resto vai para fine-tune).",
    )
    fine_tune_percent: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Percentual final das camadas MobileNetV2 descongeladas no fine-tune.",
    )
    lr_baseline: float = Field(1e-4, ge=1e-8, le=1.0, description="Learning rate da fase baseline.")
    lr_fine: float = Field(1e-5, ge=1e-8, le=1.0, description="Learning rate após descongelar (fine-tune).")
    lr_qat: float = Field(1e-5, ge=1e-8, le=1.0, description="Learning rate na fase QAT.")
    patience_baseline: int = Field(10, ge=1, le=100, description="Early stopping (baseline/fine-tune).")
    patience_qat: int = Field(5, ge=1, le=100, description="Early stopping (QAT).")
    qat_epochs: int = Field(15, ge=1, le=500, description="Épocas máximas na fase QAT.")

    img_height: int = Field(224, ge=32, le=512, description="Altura de entrada do modelo.")
    img_width: int = Field(224, ge=32, le=512, description="Largura de entrada do modelo.")

    overfitting_ratio_threshold: float = Field(
        1.6,
        ge=1.0,
        le=10.0,
        description="Se val_loss/loss ultrapassar este valor, emite aviso de possível overfitting.",
    )
    target_val_loss: Optional[float] = Field(
        None,
        ge=0.0,
        description="Opcional: meta de val_loss para celebrar convergência na UI.",
    )

    data_dir: Optional[str] = Field(
        None,
        description="Sobrescreve pasta de imagens (padrão: datas/ na raiz do projeto).",
    )
    models_dir: Optional[str] = Field(None, description="Sobrescreve pasta de checkpoints.")
    metrics_dir: Optional[str] = Field(None, description="Sobrescreve pasta de métricas JSON.")
    split_dir: Optional[str] = Field(None, description="Sobrescreve pasta dos CSVs de split.")
    tflite_dir: Optional[str] = Field(None, description="Sobrescreve pasta tfLite (legado).")

    def to_runtime_overrides(self) -> dict[str, Any]:
        """Mapeia para nomes em modules.config.settings."""
        o: dict[str, Any] = {
            "RANDOM_SEED": self.random_seed,
            "BATCH_SIZE": self.batch_size,
            "EPOCHS_BASELINE": self.epochs_baseline,
            "INITIAL_EPOCHS_FRACTION": self.initial_epochs_fraction,
            "FINE_TUNE_PERCENT": self.fine_tune_percent,
            "LR_BASELINE": self.lr_baseline,
            "LR_FINE": self.lr_fine,
            "LR_QAT": self.lr_qat,
            "PATIENCE_BASELINE": self.patience_baseline,
            "PATIENCE_QAT": self.patience_qat,
            "IMG_HEIGHT": self.img_height,
            "IMG_WIDTH": self.img_width,
            "INPUT_SHAPE": (self.img_height, self.img_width, 3),
        }
        if self.data_dir:
            o["DATA_DIR"] = self.data_dir
        if self.models_dir:
            o["MODELS_DIR"] = self.models_dir
        if self.metrics_dir:
            o["METRICS_DIR"] = self.metrics_dir
        if self.split_dir:
            o["SPLIT_DIR"] = self.split_dir
        if self.tflite_dir:
            o["TFLITE_DIR"] = self.tflite_dir
        return o


class SplitJobConfig(BaseModel):
    """Parâmetros alinhados a modules.split.create_split.run_pipeline."""

    dataset_path: str = Field(..., description="Diretório com subpastas por classe (imagens).")
    output_dir: Optional[str] = Field(None, description="Onde gravar CSVs; padrão split/.")
    image_root: Optional[str] = Field(None, description="Base para caminhos relativos no label_file.")
    test_size: float = Field(0.3, ge=0.05, le=0.9, description="Proporção (val+test) após train.")
    val_size: float = Field(0.5, ge=0.1, le=0.9, description="Dentro do holdout, fração para validação.")
    random_state: Optional[int] = Field(None, description="Semente; padrão RANDOM_SEED do projeto.")
    use_relative_paths: bool = Field(True, description="Usar caminhos relativos a dataset_path quando possível.")


class InputNormalization(str, Enum):
    minus1_1 = "minus1_1"
    zero_one = "0_1"
    zero_255 = "0_255"


class InferenceOutputMode(str, Enum):
    overlay = "overlay"
    matrix_only = "matrix_only"


class InferenceJobConfig(BaseModel):
    """Parâmetros alinhados a modules.inference.run_inference (env + run)."""

    image_path: Optional[str] = Field(
        None,
        description="Caminho absoluto no servidor para .svs (se não houver upload).",
    )
    tflite_path: Optional[str] = Field(None, description="Modelo TFLite; padrão models/qat_baseline_final.tflite.")
    output_dir: Optional[str] = Field(None, description="Pasta de saída; padrão output/ do projeto.")

    threshold: float = Field(0.9, ge=0.0, le=1.0, description="Limiar de probabilidade positiva por patch.")
    patch_multiplier: int = Field(4, ge=1, le=32, description="Multiplicador do patch 224 (tamanho = 224 * n).")
    overlay_level: int = Field(2, ge=0, le=10, description="Nível OpenSlide para imagem sobreposta.")
    processes: Optional[int] = Field(None, ge=1, le=64, description="Workers; None = min(8, cpu).")
    chunksize: int = Field(256, ge=1, le=4096, description="Chunksize do pool de inferência.")
    input_normalization: InputNormalization = Field(
        InputNormalization.minus1_1,
        description="Normalização de entrada para TFLite float32.",
    )
    output_mode: InferenceOutputMode = Field(
        InferenceOutputMode.overlay,
        description="overlay: JPG + NPZ; matrix_only: só NPZ.",
    )

    enable_blur: bool = Field(True)
    blur_kernel: int = Field(5, ge=3, le=31, description="Kernel Gaussian (ímpar).")
    enable_laplacian_gate: bool = Field(True)
    laplacian_min_var: float = Field(50.0, ge=0.0)
    enable_macenko: bool = Field(True)
    macenko_template_path: Optional[str] = Field(
        None,
        description="Template para normalização; padrão dependences/img_template.png.",
    )

    min_ram_gb: float = Field(8.0, ge=0.0, description="RAM mínima (Linux /proc); 0 desativa checagem.")


class TcgaDownloadJobConfig(BaseModel):
    """Download WSI via API GDC + manifest CSV (modules.tcga_dataset.download)."""

    case_id_column: str = Field(
        "case_submitter_id",
        description="Coluna do CSV de IDs com case submitter (ex.: TCGA-BR-4191).",
    )
    include_controlled: bool = Field(
        False,
        description="Incluir ficheiros não abertos (requer GDC_TOKEN no servidor).",
    )
    data_root: str = Field("data", description="Pasta base para data/<caso>/*.svs (relativa à raiz do repo).")
    manifest_output: str = Field(
        "wsi_manifest.csv",
        description="Onde gravar o manifest (relativo à raiz do repo).",
    )
    gdc_page_size: int = Field(500, ge=10, le=5000, description="Page size na API GDC files.")


class TcgaManifestDiskJobConfig(BaseModel):
    """Manifest só a partir de ficheiros .svs já em disco."""

    data_root: str = Field("data", description="Pasta a percorrer (relativa à raiz do repo).")
    manifest_output: str = Field(
        "wsi_manifest.csv",
        description="CSV de saída com coluna image_path (relativo à raiz do repo).",
    )


class TcgaLabelsJobConfig(BaseModel):
    """Cruza manifest com planilha (Patient ID + Subtype) → label_file.csv."""

    sheets_url: str = Field(
        ...,
        description="URL Google Sheets (export CSV público) ou caminho absoluto no servidor para .csv.",
    )
    manifest_csv: str = Field(
        "wsi_manifest.csv",
        description="Manifest com caminhos (relativo à raiz do repo ou absoluto).",
    )
    label_output: str = Field(
        "split/label_file.csv",
        description="Saída Image_path, Label (relativo à raiz do repo).",
    )
    patient_id_column: str = Field("Patient ID", description="Coluna na planilha com ID do caso/paciente.")
    subtype_column: str = Field("Subtype", description="Coluna na planilha com o rótulo.")


def build_api_schema_payload() -> dict[str, Any]:
    """JSON para o frontend: schemas + grupos."""
    return {
        "version": 1,
        "groups": [
            {"id": "training", "title": "Treino", "description": "MobileNetV2 + fine-tune + QAT opcional"},
            {"id": "split", "title": "Split", "description": "Gerar CSVs train/val/test a partir de pastas por classe"},
            {"id": "inference", "title": "Inferência", "description": "WSI .svs + TFLite + heatmap"},
            {
                "id": "tcga",
                "title": "TCGA / Dados",
                "description": "Download GDC, manifest a partir de disco, label_file via Google Sheets",
            },
        ],
        "schemas": {
            "TrainingJobConfig": TrainingJobConfig.model_json_schema(),
            "SplitJobConfig": SplitJobConfig.model_json_schema(),
            "InferenceJobConfig": InferenceJobConfig.model_json_schema(),
            "TcgaDownloadJobConfig": TcgaDownloadJobConfig.model_json_schema(),
            "TcgaManifestDiskJobConfig": TcgaManifestDiskJobConfig.model_json_schema(),
            "TcgaLabelsJobConfig": TcgaLabelsJobConfig.model_json_schema(),
        },
        "defaults": {
            "TrainingJobConfig": TrainingJobConfig().model_dump(),
            "SplitJobConfig": {
                "dataset_path": "datas",
                "output_dir": None,
                "image_root": None,
                "test_size": 0.3,
                "val_size": 0.5,
                "random_state": None,
                "use_relative_paths": True,
            },
            "InferenceJobConfig": InferenceJobConfig().model_dump(mode="json"),
            "TcgaDownloadJobConfig": TcgaDownloadJobConfig().model_dump(mode="json"),
            "TcgaManifestDiskJobConfig": TcgaManifestDiskJobConfig().model_dump(mode="json"),
            "TcgaLabelsJobConfig": TcgaLabelsJobConfig(
                sheets_url="https://docs.google.com/spreadsheets/d/EXAMPLE/edit",
            ).model_dump(mode="json"),
        },
    }
