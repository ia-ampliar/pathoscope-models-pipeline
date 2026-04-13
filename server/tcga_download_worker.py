"""Subprocesso: download WSI via GDC + manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python -m server.tcga_download_worker <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    result_path = Path(payload["result_json_path"])
    ids_csv = Path(payload["ids_csv_path"])
    cfg_dict = payload["config"]

    from modules.tcga_dataset.config import TcgaDatasetConfig
    from modules.tcga_dataset.download import download_and_write_manifest
    from server.schemas import TcgaDownloadJobConfig

    opt = TcgaDownloadJobConfig.model_validate(cfg_dict)
    base = TcgaDatasetConfig(
        root_dir=root,
        data_root=(root / opt.data_root).resolve(),
        ids_csv=ids_csv.resolve(),
        case_id_column=opt.case_id_column,
        wsi_manifest_csv=(root / opt.manifest_output).resolve(),
        only_open_access=not opt.include_controlled,
        gdc_files_page_size=opt.gdc_page_size,
    )
    out = download_and_write_manifest(base)
    rel_manifest = str(out.resolve().relative_to(root.resolve())).replace("\\", "/")
    result = {
        "kind": "tcga_download",
        "manifest_csv": rel_manifest,
        "manifest_absolute": str(out.resolve()),
        "data_root": str(base.data_root.resolve()),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(result, rf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
