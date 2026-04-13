"""
Processo filho: operações TCGA (download GDC, manifest disco, label file).
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
        print("Uso: python -m server.tcga_worker <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    result_path = Path(payload["result_json_path"])
    repo_root = Path(payload["root_dir"]).resolve()
    op = payload["operation"]

    from modules.tcga_dataset.config import TcgaDatasetConfig
    from modules.tcga_dataset.download import download_and_write_manifest
    from modules.tcga_dataset.labels import build_label_file
    from modules.tcga_dataset.manifest import build_wsi_manifest_from_disk

    out: dict = {"operation": op}

    if op == "download":
        data_root = Path(payload.get("data_root") or (repo_root / "data")).resolve()
        manifest_out = Path(payload.get("wsi_manifest_csv") or (repo_root / "wsi_manifest.csv")).resolve()
        cfg = TcgaDatasetConfig(
            root_dir=repo_root,
            data_root=data_root,
            ids_csv=Path(payload["ids_csv_path"]).resolve(),
            case_id_column=payload.get("case_id_column", "case_submitter_id"),
            wsi_manifest_csv=manifest_out,
            only_open_access=not payload.get("include_controlled", False),
            gdc_files_page_size=int(payload.get("gdc_files_page_size", 500)),
        )
        path = download_and_write_manifest(cfg)
        out["wsi_manifest_csv"] = str(path.resolve())
        out["data_root"] = str(data_root)

    elif op == "manifest":
        data_root = Path(payload.get("data_root") or (repo_root / "data")).resolve()
        manifest_out = Path(payload.get("wsi_manifest_csv") or (repo_root / "wsi_manifest.csv")).resolve()
        path = build_wsi_manifest_from_disk(
            data_root,
            repo_root,
            manifest_out,
            path_column="image_path",
        )
        out["wsi_manifest_csv"] = str(path.resolve())

    elif op == "labels":
        sheets_ref = payload["sheets_ref"]
        manifest_csv = Path(payload["manifest_csv_path"]).resolve()
        label_out = Path(
            payload.get("label_file_csv") or (repo_root / "split" / "label_file.csv")
        ).resolve()
        path = build_label_file(
            manifest_csv=manifest_csv,
            sheets_ref=sheets_ref,
            output_csv=label_out,
            repo_root=repo_root,
            manifest_path_column="image_path",
            patient_id_column=payload.get("patient_id_column", "Patient ID"),
            subtype_column=payload.get("subtype_column", "Subtype"),
        )
        out["label_file_csv"] = str(path.resolve())
        out["manifest_csv"] = str(manifest_csv)

    else:
        raise SystemExit(f"Operação desconhecida: {op}")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(out, rf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
