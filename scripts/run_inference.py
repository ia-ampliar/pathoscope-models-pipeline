#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Limites de threads internas (ANTES de importar numpy/cv/tf) ----------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# ------------------------------------------------------------------------------
import os
import time
import json
import numpy as np
import cv2 as cv
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from itertools import product
from matplotlib import cm
import multiprocessing as mp

import tensorflow as tf

# =============================
# ### CONFIGURATIONS ### 
# =============================

# Diretório base fixo (para uso local e em Docker)
BASE_DIR = os.getenv(
    "APP_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Caminhos principais (fixos por padrão)
IMAGE_PATH = os.getenv("IMAGE_PATH", "")  # pode ser definido via ENV ou passado direto para run(image_path)
TFLITE_PATH = os.getenv("TFLITE_PATH", os.path.join(BASE_DIR, "models", "qat_baseline_final.tflite"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

# Hiperparâmetros principais
THRESHOLD = float(os.getenv("THRESHOLD", "0.9"))
PATCH_MULTIPLIER = int(os.getenv("PATCH_MULTIPLIER", "4"))
OVERLAY_LEVEL = int(os.getenv("OVERLAY_LEVEL", "2"))

# Processamento em lote / paralelismo
_ENV_PROCESSES = os.getenv("PROCESSES", "").strip()
if _ENV_PROCESSES:
    try:
        PROCESSES = int(_ENV_PROCESSES)
    except ValueError:
        PROCESSES = None
else:
    PROCESSES = None
CHUNKSIZE = int(os.getenv("CHUNKSIZE", "256"))

# Normalização de entrada do modelo
INPUT_NORMALIZATION = os.getenv("INPUT_NORMALIZATION", "minus1_1")

# Config do preprocessing (globais simples) – todas as etapas ATIVADAS por padrão
PREPROC_ENABLE_BLUR = os.getenv("ENABLE_BLUR", "true").lower() in {"1", "true", "yes"}
PREPROC_BLUR_KERNEL = int(os.getenv("BLUR_KERNEL", "5"))

PREPROC_ENABLE_LAPLACIAN_GATE = os.getenv("ENABLE_LAPLACIAN_GATE", "true").lower() in {"1", "true", "yes"}
PREPROC_LAPLACIAN_MIN_VAR = float(os.getenv("LAPLACIAN_MIN_VAR", "50.0"))

PREPROC_ENABLE_MACENKO = os.getenv("ENABLE_MACENKO", "true").lower() in {"1", "true", "yes"}
PREPROC_MACENKO_TEMPLATE_PATH = os.getenv(
    "MACENKO_TEMPLATE_PATH",
    os.path.join(BASE_DIR, "dependences", "img_template.png")
)

# Limite mínimo de RAM (em GB) recomendado para WSI
MIN_RAM_GB = float(os.getenv("MIN_RAM_GB", "8"))


def _get_total_ram_gb() -> float | None:
    """
    Retorna a memória RAM total aproximada em GB usando /proc/meminfo.
    Em ambientes sem /proc/meminfo, retorna None.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # valor em kB -> GB
                    mem_kb = float(parts[1])
                    return mem_kb / (1024.0 * 1024.0)
    except Exception:
        return None


def _check_ram_or_die() -> None:
    """
    Verifica a RAM disponível e encerra o processo com uma mensagem clara
    se estiver abaixo do mínimo recomendado para WSI de alta resolução.
    """
    total_gb = _get_total_ram_gb()
    if total_gb is None:
        print("[WARNING] Não foi possível determinar a quantidade total de RAM (MemTotal). "
              "Prosseguindo mesmo assim.")
        return

    if total_gb < MIN_RAM_GB:
        raise SystemExit(
            f"[ERROR] Memória total detectada ({total_gb:.1f} GB) é menor que o mínimo recomendado "
            f"({MIN_RAM_GB:.1f} GB) para processar WSI de histopatologia/genômica. "
            "Ajuste o limite de memória do container antes de rodar a inferência."
        )


# Verificação de memória logo no início da execução do script
_check_ram_or_die()


# Verificação opcional da versão do TF (deve ser >= 2.17 para suportar op version 12)
print(f"[INFO] TensorFlow version: {tf.__version__}")
if tf.__version__ < "2.17.0":
    print("[WARNING] Versão do TensorFlow pode ser muito antiga para o modelo TFLite.")

# -----------------------------
# PREPROCESSING (funções)
# -----------------------------
def _apply_blur_filter(image_bgr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)

def _variance_of_laplacian(image_bgr: np.ndarray) -> float:
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()

def _macenko_normalization(image_bgr: np.ndarray, template_path: str) -> np.ndarray:
    """
    Implementação simplificada (LAB mean/std matching nos canais a/b).
    """
    try:
        template = cv.imread(str(template_path))
        if template is None:
            print(f"[WARNING] Template não encontrado: {template_path}")
            return image_bgr

        image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
        template_lab = cv.cvtColor(template, cv.COLOR_BGR2LAB)

        for channel in [1, 2]:
            image_channel = image_lab[:, :, channel].astype(np.float32)
            template_channel = template_lab[:, :, channel].astype(np.float32)

            image_mean = np.mean(image_channel)
            image_std = np.std(image_channel)
            template_mean = np.mean(template_channel)
            template_std = np.std(template_channel)

            if image_std > 0:
                image_channel = (image_channel - image_mean) * (template_std / image_std) + template_mean
                image_lab[:, :, channel] = np.clip(image_channel, 0, 255).astype(np.uint8)

        normalized = cv.cvtColor(image_lab, cv.COLOR_LAB2BGR)
        return normalized
    except Exception as e:
        print(f"[WARNING] Erro na normalização Macenko: {e}")
        return image_bgr


# Estado do processo principal
patch_size = None
width = None
hight = None
dz_level = None
dz_cols = None
dz_rows = None
mask = None
indices = None

# Estado por worker
_worker_tiles = None
_worker_interpreter = None
_worker_input_details = None
_worker_output_details = None
_worker_patch_size = None
_worker_dz_level = None
_worker_dz_cols = None
_worker_dz_rows = None

# Preproc por worker
_worker_preproc = None  # dict com configs

# Normalização por worker
_worker_input_norm = None  # string: "0_255", "0_1", "minus1_1"


def processing_time(start_time, end_time):
    total_time = end_time - start_time
    minutos, segundos = divmod(total_time, 60)
    milisegundos = int((segundos - int(segundos)) * 1000)
    print(f"Tempo de execução:\n00:{int(minutos):02d}:{int(segundos):02d}.{milisegundos:03d}")


def _pick_mp_context():
    try:
        # No Windows, forkserver não existe -> cai no spawn
        if os.name == "nt":
            return mp.get_context("spawn"), "spawn"
        ctx = mp.get_context("forkserver")
        return ctx, "forkserver"
    except Exception:
        ctx = mp.get_context("spawn")
        return ctx, "spawn"


# ============================================================
# TFLITE IO (SUPORTA FP32 E MODELOS QUANTIZADOS)
# ============================================================

def _extract_quant_params(details: dict):
    """
    Retorna (scale, zero_point) se existir quantização válida, senão (0.0, 0).
    Compatível com 'quantization' e 'quantization_parameters'.
    """
    scale, zero_point = details.get("quantization", (0.0, 0))

    if (scale is None or scale == 0.0) and ("quantization_parameters" in details):
        qp = details["quantization_parameters"]
        if isinstance(qp, dict):
            scales = qp.get("scales", [])
            zps = qp.get("zero_points", [])
            if len(scales) > 0 and len(zps) > 0:
                scale = float(scales[0])
                zero_point = int(zps[0])

    if scale is None:
        scale = 0.0
    if zero_point is None:
        zero_point = 0

    return float(scale), int(zero_point)


def _prepare_input_for_tflite(normalized_img: np.ndarray, input_details: dict) -> np.ndarray:
    """
    Recebe imagem já normalizada (para float32) ou uint8 cru (para quantizado)
    e retorna o tensor pronto para o modelo, com o dtype correto.
    - Se modelo espera float32: apenas adiciona batch dimension.
    - Se modelo espera uint8/int8: aplica quantização (scale/zero point) se disponível,
      caso contrário converte diretamente.
    """
    expected_dtype = input_details["dtype"]
    scale, zero_point = _extract_quant_params(input_details)

    # Garantir que a imagem tenha shape (H, W, C)
    if normalized_img.ndim == 3:
        x = np.expand_dims(normalized_img, axis=0)  # [1, H, W, C]
    else:
        x = normalized_img

    if expected_dtype == np.float32:
        # Já deve estar normalizada, apenas garantir dtype
        return x.astype(np.float32, copy=False)

    # Para modelos quantizados, a entrada deve ser uint8/int8 no range [0,255] (ou o range original)
    # Se scale > 0, aplicamos a quantização: q = round(x/scale + zero_point)
    if scale != 0.0:
        # x deve estar em float32 representando os valores originais (ex: [0,255])
        q = np.rint(x / scale + zero_point)
        if expected_dtype == np.uint8:
            q = np.clip(q, 0, 255).astype(np.uint8)
        elif expected_dtype == np.int8:
            q = np.clip(q, -128, 127).astype(np.int8)
        else:
            q = q.astype(expected_dtype)
        return q
    else:
        # Sem quantização: assume que x já está no range correto para o dtype
        return x.astype(expected_dtype, copy=False)


def _dequantize_output_if_needed(output_tensor: np.ndarray, output_details: dict) -> np.ndarray:
    """
    Retorna saída sempre em float32.
    Se output for quantizado, aplica (y - zp) * scale.
    """
    out_dtype = output_details["dtype"]
    scale, zero_point = _extract_quant_params(output_details)

    y = output_tensor
    if out_dtype in (np.uint8, np.int8) and scale != 0.0:
        y = (y.astype(np.float32) - zero_point) * scale
        return y.astype(np.float32, copy=False)

    return y.astype(np.float32, copy=False)


# -----------------------------
# WORKERS
# -----------------------------
def _init_worker(
    image_path,
    tflite_path,
    _patch_size,
    _dz_level,
    preproc_cfg: dict,
    input_norm: str,
):
    global _worker_tiles, _worker_interpreter, _worker_input_details, _worker_output_details
    global _worker_patch_size, _worker_dz_level, _worker_dz_cols, _worker_dz_rows
    global _worker_preproc, _worker_input_norm

    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    _worker_patch_size = int(_patch_size)
    _worker_dz_level = int(_dz_level)
    _worker_preproc = dict(preproc_cfg or {})
    _worker_input_norm = input_norm

    wsi = openslide.OpenSlide(image_path)
    _worker_tiles = DeepZoomGenerator(wsi, tile_size=_worker_patch_size, overlap=0)
    _worker_dz_cols, _worker_dz_rows = _worker_tiles.level_tiles[_worker_dz_level]

    # Carregar modelo TFLite
    _worker_interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)
    _worker_interpreter.allocate_tensors()
    _worker_input_details = _worker_interpreter.get_input_details()
    _worker_output_details = _worker_interpreter.get_output_details()


def _process_block(y, x):
    if indices is None or mask is None:
        raise RuntimeError("indices/mask não inicializados no processo principal.")

    coords = []
    for m, n in product(indices, indices):
        yy = y + m
        xx = x + n
        if yy >= hight or xx >= width:
            continue
        y1 = min(yy + patch_size, hight)
        x1 = min(xx + patch_size, width)
        region = mask[yy:y1, xx:x1]
        if region.size == 0:
            continue
        if np.mean(region) > 16:
            coords.append((int(yy), int(xx)))
    return coords


def _apply_preprocessing_pipeline(image_bgr: np.ndarray):
    """
    Aplica blur + macenko + gate por laplacian variance.
    Retorna (image_bgr, keep: bool).
    """
    cfg = _worker_preproc or {}

    if cfg.get("enable_blur", False):
        k = int(cfg.get("blur_kernel", 5))
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        image_bgr = _apply_blur_filter(image_bgr, kernel_size=k)

    if cfg.get("enable_macenko", False):
        tpl = cfg.get("macenko_template_path", None)
        if tpl:
            image_bgr = _macenko_normalization(image_bgr, template_path=tpl)

    if cfg.get("enable_laplacian_gate", False):
        minv = float(cfg.get("laplacian_min_var", 0.0))
        v = _variance_of_laplacian(image_bgr)
        if v < minv:
            return image_bgr, False

    return image_bgr, True


def _normalize_image(img_uint8: np.ndarray, norm_type: str) -> np.ndarray:
    """
    Aplica normalização à imagem uint8 [0,255] e retorna float32 no range especificado.
    """
    if norm_type == "0_255":
        return img_uint8.astype(np.float32)  # mantém [0,255]
    elif norm_type == "0_1":
        return img_uint8.astype(np.float32) / 255.0
    elif norm_type == "minus1_1":
        return (img_uint8.astype(np.float32) / 127.5) - 1.0
    else:
        raise ValueError(f"Tipo de normalização desconhecido: {norm_type}")


def _process_patch(abs_y, abs_x):
    tile_row = abs_y // _worker_patch_size
    tile_col = abs_x // _worker_patch_size
    if tile_row >= _worker_dz_rows or tile_col >= _worker_dz_cols:
        return None

    # get tile (RGB PIL) -> numpy BGR uint8 (para pré-processamento)
    img_rgb = _worker_tiles.get_tile(_worker_dz_level, (tile_col, tile_row)).convert("RGB")
    img_bgr = cv.cvtColor(np.array(img_rgb, dtype=np.uint8), cv.COLOR_RGB2BGR)

    # pré-processamento (blur, macenko, laplacian) – mantido em BGR
    img_bgr, keep = _apply_preprocessing_pipeline(img_bgr)
    if not keep:
        return None

    # resize -> 224x224, ainda BGR
    img_bgr_224 = cv.resize(img_bgr, (224, 224), interpolation=cv.INTER_LINEAR)
    if img_bgr_224.dtype != np.uint8:
        img_bgr_224 = img_bgr_224.astype(np.uint8, copy=False)

    # Converter BGR para RGB (formato esperado pelo modelo)
    img_rgb_224 = cv.cvtColor(img_bgr_224, cv.COLOR_BGR2RGB)

    inp = _worker_input_details[0]
    expected_dtype = inp["dtype"]

    # --- Aplicar normalização conforme o tipo do modelo ---
    if expected_dtype == np.float32:
        # Modelo float32: normalizar segundo argumento do usuário
        img_normalized = _normalize_image(img_rgb_224, _worker_input_norm)
        # Verificar se a normalização escolhida é a padrão recomendada
        if _worker_input_norm == "0_255":
            print(f"[WARNING] Modelo float32 com normalização '0_255' (valores [0,255]) pode não ser o esperado pelo treinamento. Considere usar '0_1' ou 'minus1_1'.")
        img_input = _prepare_input_for_tflite(img_normalized, inp)
    else:
        # Modelo quantizado: manter uint8 RGB, a função _prepare_input_for_tflite aplicará quantização
        img_input = _prepare_input_for_tflite(img_rgb_224, inp)

    # fallback (caso ainda haja erro de shape/dtype)
    try:
        _worker_interpreter.set_tensor(inp["index"], img_input)
    except ValueError as e:
        print(f"[ERROR] Falha ao setar tensor: {e}")
        # Tentativa simples: converter para float32 se necessário
        if expected_dtype == np.float32:
            fallback = np.expand_dims(img_rgb_224.astype(np.float32), axis=0)
        else:
            fallback = np.expand_dims(img_rgb_224.astype(expected_dtype), axis=0)
        _worker_interpreter.set_tensor(inp["index"], fallback)

    _worker_interpreter.invoke()

    raw_out = _worker_interpreter.get_tensor(_worker_output_details[0]["index"])
    probs = _dequantize_output_if_needed(raw_out, _worker_output_details[0])

    pred = float(probs[0][0])  # assume segunda classe como positiva
    if pred > THRESHOLD:
        return (int(tile_row), int(tile_col), pred)
    return None


def run(image_path: str | None = None):
    global THRESHOLD, patch_size, width, hight, dz_level, dz_cols, dz_rows, mask, indices

    # Seleciona caminho da imagem
    if image_path is None or image_path == "":
        image_path = IMAGE_PATH
    if not image_path:
        raise ValueError(
            "Nenhum caminho de imagem fornecido. "
            "Defina IMAGE_PATH como variável de ambiente ou passe image_path diretamente para run(image_path)."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada em image_path='{image_path}'")

    if not os.path.exists(TFLITE_PATH):
        raise FileNotFoundError(f"Modelo TFLite não encontrado em TFLITE_PATH='{TFLITE_PATH}'")

    if PREPROC_MACENKO_TEMPLATE_PATH and not os.path.exists(PREPROC_MACENKO_TEMPLATE_PATH):
        print(f"[WARNING] Macenko template não encontrado em '{PREPROC_MACENKO_TEMPLATE_PATH}'. "
              "A normalização será ignorada.")

    # Configurações derivadas das constantes globais
    threshold = THRESHOLD
    patch_multiplier = PATCH_MULTIPLIER
    overlay_level = OVERLAY_LEVEL
    processes = PROCESSES
    chunksize = CHUNKSIZE

    enable_blur = PREPROC_ENABLE_BLUR
    blur_kernel = PREPROC_BLUR_KERNEL
    enable_laplacian_gate = PREPROC_ENABLE_LAPLACIAN_GATE
    laplacian_min_var = PREPROC_LAPLACIAN_MIN_VAR
    enable_macenko = PREPROC_ENABLE_MACENKO
    macenko_template_path = PREPROC_MACENKO_TEMPLATE_PATH
    input_normalization = INPUT_NORMALIZATION

    THRESHOLD = float(threshold)
    if patch_multiplier <= 0:
        raise ValueError("patch_multiplier deve ser positivo")

    # ====== principal: abre WSI para metadata + máscara ======
    wsi = openslide.OpenSlide(image_path)
    w, h = wsi.dimensions
    patch_size = int(224 * patch_multiplier)

    width = ((w + patch_size - 1) // patch_size) * patch_size
    hight = ((h + patch_size - 1) // patch_size) * patch_size

    tiles_main = DeepZoomGenerator(wsi, tile_size=patch_size, overlap=0)
    dz_level = tiles_main.level_count - 1
    dz_cols, dz_rows = tiles_main.level_tiles[dz_level]

    level = len(wsi.level_dimensions) - 1
    img_low = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
    gray_img = cv.cvtColor(np.array(img_low), cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    mask = cv.resize(mask, (width, hight), interpolation=cv.INTER_NEAREST)

    c = 4
    slide_dim = patch_size * c
    indices = np.arange(0, slide_dim, patch_size, dtype=np.int64)

    y_indices = np.arange(0, hight, slide_dim, dtype=np.int64)
    x_indices = np.arange(0, width, slide_dim, dtype=np.int64)

    block_list = []
    for y, x in product(y_indices, x_indices):
        y = int(y); x = int(x)
        y1 = min(y + slide_dim, hight)
        x1 = min(x + slide_dim, width)
        region = mask[y:y1, x:x1]
        if region.size and np.mean(region) > 16:
            block_list.append((y, x))

    jet_heatmap_matrix = np.zeros((int(hight / patch_size), int(width / patch_size)), dtype=np.float16)

    wsi.close()

    # ====== multiprocess context ======
    ctx, ctx_name = _pick_mp_context()
    print(f"[INFO] multiprocessing context: {ctx_name}")

    if processes is None:
        processes = min(8, os.cpu_count() or 8)
    processes = max(1, int(processes))

    preproc_cfg = dict(
        enable_blur=bool(enable_blur),
        blur_kernel=int(blur_kernel),
        enable_laplacian_gate=bool(enable_laplacian_gate),
        laplacian_min_var=float(laplacian_min_var),
        enable_macenko=bool(enable_macenko),
        macenko_template_path=macenko_template_path,
    )

    start_time = time.time()

    # ====== 1) block -> patch_coords (NO PRINCIPAL) ======
    patch_coords_nested = [_process_block(y, x) for (y, x) in block_list]
    patch_coords = [xy for sub in patch_coords_nested for xy in sub]
    print(f"Quantidade de patches a processar: {len(patch_coords)}")

    # ====== 2) patch -> inferência (NO POOL) ======
    with ctx.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(image_path, TFLITE_PATH, patch_size, dz_level, preproc_cfg, input_normalization),
        maxtasksperchild=200,
    ) as pool:
        results = pool.starmap(_process_patch, patch_coords, chunksize=chunksize)

    results = [r for r in results if r is not None]
    if results:
        arr = np.array(results, dtype=np.float32)
        ys = arr[:, 0].astype(np.int64)
        xs = arr[:, 1].astype(np.int64)
        vals = arr[:, 2].astype(np.float16)
        jet_heatmap_matrix[ys, xs] = vals

    end_time = time.time()
    processing_time(start_time, end_time)

    # ====== salvar ======
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    image_id = base.split(".", 1)[0]  # ex: TCGA-BR-...-BS1 (remove sufixos após o primeiro ponto)
    image_output_dir = os.path.join(OUTPUT_DIR, f"{image_id}_results")
    os.makedirs(image_output_dir, exist_ok=True)

    heatmap_path = os.path.join(image_output_dir, f"hm_{base}_qat.jpg")
    superimposed_path = os.path.join(image_output_dir, f"superimposed_{base}_qat.jpg")

    metrics = {
        "image": base,
        "patches_processed": len(results),
        "processing_time": end_time - start_time,
        "threshold": threshold,
        "patch_size": patch_size,
        "overlay_level": overlay_level,
        "input_normalization": input_normalization,
        "enable_blur": bool(enable_blur),
        "blur_kernel": int(blur_kernel),
        "enable_laplacian_gate": bool(enable_laplacian_gate),
        "laplacian_min_var": float(laplacian_min_var),
        "enable_macenko": bool(enable_macenko),
        "macenko_template_path": macenko_template_path,
    }
    metrics_path = os.path.join(image_output_dir, f"metrics_{base}_qat.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    heatmap_matrix = np.uint8(255 * jet_heatmap_matrix)
    jet_colors = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_matrix]
    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap_img.save(heatmap_path, quality=100)

    wsi = openslide.OpenSlide(image_path)
    img = wsi.read_region((0, 0), overlay_level, wsi.level_dimensions[overlay_level]).convert("RGB")
    img_array = np.array(img, dtype=np.float32)

    jet_heatmap_img = Image.open(heatmap_path).resize((img_array.shape[1], img_array.shape[0]))
    superimposed_img = np.array(jet_heatmap_img, dtype=np.float32) + img_array
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(superimposed_path, quality=100)
    wsi.close()

    print(f"Quantidade de patches processados: {len(results)}")
    print(f"Heatmap salvo em: {heatmap_path}")
    print(f"Superimposed salvo em: {superimposed_path}")


if __name__ == "__main__":
    # Permite rodar diretamente o script dentro do container:
    IMAGE_PATH="/home/ampliar/Desktop/tf_model/img_test/cancer/TCGA-BR-7197-01A-01-BS1.2fbf309e-bb43-49d2-aae9-b3016a520722.svs"
    run(IMAGE_PATH)