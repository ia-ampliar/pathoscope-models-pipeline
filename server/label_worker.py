"""Processo filho: gera label_file.csv cruzando manifest WSI com planilha."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python label_worker.py <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)

    from modules.tcga_dataset.labels import build_label_file
    from server.schemas import LabelJobConfig

    lc = LabelJobConfig.model_validate(cfg)

    output_csv = Path(lc.output_csv) if lc.output_csv else root / "split" / "label_file.csv"

    build_label_file(
        manifest_csv=Path(lc.manifest_csv),
        sheets_ref=lc.sheets_ref,
        output_csv=output_csv,
        repo_root=root,
        manifest_path_column=lc.manifest_path_column,
        patient_id_column=lc.patient_id_column,
        subtype_column=lc.subtype_column,
    )


if __name__ == "__main__":
    main()
