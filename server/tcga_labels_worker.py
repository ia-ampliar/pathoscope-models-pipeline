"""Subprocesso: label_file a partir de manifest + planilha."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python -m server.tcga_labels_worker <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    result_path = Path(payload["result_json_path"])
    cfg_dict = payload["config"]
    manifest_arg = payload.get("manifest_csv_path")

    from modules.tcga_dataset.labels import build_label_file
    from server.schemas import TcgaLabelsJobConfig

    opt = TcgaLabelsJobConfig.model_validate(cfg_dict)
    if manifest_arg:
        manifest_csv = Path(manifest_arg).resolve()
    else:
        p = Path(opt.manifest_csv)
        manifest_csv = p if p.is_absolute() else (root / opt.manifest_csv).resolve()

    label_out = Path(opt.label_output)
    label_out = label_out if label_out.is_absolute() else (root / opt.label_output).resolve()

    out = build_label_file(
        manifest_csv=manifest_csv,
        sheets_ref=opt.sheets_url,
        output_csv=label_out,
        repo_root=root.resolve(),
        manifest_path_column=None,
        patient_id_column=opt.patient_id_column,
        subtype_column=opt.subtype_column,
    )
    rel = str(out.resolve().relative_to(root.resolve())).replace("\\", "/")
    result = {
        "kind": "tcga_labels",
        "label_file": rel,
        "label_absolute": str(out.resolve()),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(result, rf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
