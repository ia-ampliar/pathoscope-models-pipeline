"""Processo filho: aplica env e executa inferência WSI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _set_env(inv, work_output: str) -> None:
    """Variáveis lidas em import time por modules.inference.run_inference."""
    os.environ["OUTPUT_DIR"] = work_output
    os.environ["THRESHOLD"] = str(inv.threshold)
    os.environ["PATCH_MULTIPLIER"] = str(inv.patch_multiplier)
    os.environ["OVERLAY_LEVEL"] = str(inv.overlay_level)
    os.environ["CHUNKSIZE"] = str(inv.chunksize)
    if inv.processes is not None:
        os.environ["PROCESSES"] = str(inv.processes)
    else:
        os.environ.pop("PROCESSES", None)

    norm = inv.input_normalization
    norm_val = norm.value if hasattr(norm, "value") else str(norm)
    os.environ["INPUT_NORMALIZATION"] = norm_val

    os.environ["ENABLE_BLUR"] = "true" if inv.enable_blur else "false"
    os.environ["BLUR_KERNEL"] = str(inv.blur_kernel)
    os.environ["ENABLE_LAPLACIAN_GATE"] = "true" if inv.enable_laplacian_gate else "false"
    os.environ["LAPLACIAN_MIN_VAR"] = str(inv.laplacian_min_var)
    os.environ["ENABLE_MACENKO"] = "true" if inv.enable_macenko else "false"
    if inv.macenko_template_path:
        os.environ["MACENKO_TEMPLATE_PATH"] = inv.macenko_template_path

    os.environ["MIN_RAM_GB"] = str(inv.min_ram_gb)

    root = Path(__file__).resolve().parents[1]
    if inv.tflite_path:
        os.environ["TFLITE_PATH"] = inv.tflite_path

    os.environ.setdefault("APP_BASE_DIR", str(root))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python infer_worker.py <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    result_path = Path(payload["result_json_path"])
    image_path = payload["image_path"]
    work_output = payload["work_output_dir"]

    from server.schemas import InferenceJobConfig

    inv = InferenceJobConfig.model_validate(payload["inference"])
    _set_env(inv, work_output)

    # Import após env (constantes do módulo dependem de getenv)
    from modules.inference.run_inference import run as infer_run

    mode = inv.output_mode.value if hasattr(inv.output_mode, "value") else str(inv.output_mode)
    infer_run(image_path=image_path, output_mode=mode)

    # Coleta artefatos gerados em OUTPUT_DIR efetivo
    out_base = Path(os.environ.get("OUTPUT_DIR", work_output))
    artifacts: dict = {"output_dir": str(out_base.resolve()), "files": []}
    metrics_data = None
    if out_base.exists():
        for p in sorted(out_base.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(out_base)).replace("\\", "/")
                artifacts["files"].append(
                    {
                        "path": rel,
                        "size": p.stat().st_size,
                        "suffix": p.suffix.lower(),
                    }
                )
                if p.name.startswith("metrics_") and p.suffix == ".json":
                    try:
                        with open(p, encoding="utf-8") as mf:
                            metrics_data = json.load(mf)
                    except Exception:
                        pass

    result = {
        "artifacts": artifacts,
        "metrics": metrics_data,
        "image_path": image_path,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(result, rf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
