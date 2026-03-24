from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]

# Diretórios principais (mantém compatibilidade com o notebook original)
DATA_DIR = ROOT_DIR / "datas"
SPLIT_DIR = ROOT_DIR / "split"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"
CALLBACKS_DIR = ROOT_DIR / "callbacks"
TFLITE_DIR = ROOT_DIR / "tfLite"


for _dir in [SPLIT_DIR, MODELS_DIR, METRICS_DIR, CALLBACKS_DIR, TFLITE_DIR]:
    os.makedirs(_dir, exist_ok=True)


# Hiperparâmetros
RANDOM_SEED = 42

IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

BATCH_SIZE = 32

EPOCHS_BASELINE = 100
INITIAL_EPOCHS_FRACTION = 0.4  # 40% do baseline

FINE_TUNE_PERCENT = 0.30

LR_BASELINE = 1e-4
LR_FINE = 1e-5
LR_QAT = 1e-5

PATIENCE_BASELINE = 10
PATIENCE_QAT = 5


def get_train_val_test_csv_paths():
    """Retorna os caminhos dos CSVs de split."""
    train_csv = SPLIT_DIR / "train_data.csv"
    val_csv = SPLIT_DIR / "val_data.csv"
    test_csv = SPLIT_DIR / "test_data.csv"
    return train_csv, val_csv, test_csv


def get_label_file_path():
    """Retorna o caminho do label_file.csv (criado por create_split.py)."""
    return SPLIT_DIR / "label_file.csv"

