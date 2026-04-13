"""
Processo filho: aplica overrides de config e executa run_training_pipeline.
Invocado como: python -m server.train_worker <config.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python -m server.train_worker <config.json>", file=sys.stderr)
        sys.exit(2)

    cfg_path = Path(sys.argv[1])
    with open(cfg_path, encoding="utf-8") as f:
        payload = json.load(f)

    from modules.config.runtime import apply_training_overrides
    from modules.training.train_utils import append_metrics_jsonl
    from server.schemas import TrainingJobConfig
    from src.train import run_training_pipeline

    training = TrainingJobConfig.model_validate(payload["training"])
    apply_training_overrides(training.to_runtime_overrides())

    stream_path = payload.get("metrics_stream_path")

    try:
        run_training_pipeline(
            use_strategy=training.use_strategy,
            use_qat=training.use_qat,
            augment=training.augment,
            metrics_stream_path=stream_path,
            qat_epochs=training.qat_epochs,
            overfitting_ratio_threshold=training.overfitting_ratio_threshold,
            target_val_loss=training.target_val_loss,
        )
    except Exception as e:
        if stream_path:
            append_metrics_jsonl(
                stream_path,
                {"type": "error", "message": str(e)},
            )
        raise


if __name__ == "__main__":
    main()
