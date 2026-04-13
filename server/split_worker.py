"""Processo filho: executa run_pipeline de split."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python split_worker.py <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)

    from modules.split.create_split import run_pipeline
    from server.schemas import SplitJobConfig

    s = SplitJobConfig.model_validate(cfg)
    run_pipeline(
        dataset_path=s.dataset_path,
        output_dir=s.output_dir,
        image_root=s.image_root,
        test_size=s.test_size,
        val_size=s.val_size,
        random_state=s.random_state,
        use_relative_paths=s.use_relative_paths,
    )


if __name__ == "__main__":
    main()
