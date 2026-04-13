"""Subprocesso: manifest a partir de disco."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if len(sys.argv) < 2:
        print("Uso: python -m server.tcga_manifest_worker <config.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    result_path = Path(payload["result_json_path"])
    cfg_dict = payload["config"]

    from modules.tcga_dataset.manifest import build_wsi_manifest_from_disk
    from server.schemas import TcgaManifestDiskJobConfig

    opt = TcgaManifestDiskJobConfig.model_validate(cfg_dict)
    data_root = (root / opt.data_root).resolve()
    out_csv = (root / opt.manifest_output).resolve()
    path = build_wsi_manifest_from_disk(
        data_root,
        root.resolve(),
        out_csv,
        path_column="image_path",
    )
    rel = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    result = {
        "kind": "tcga_manifest_disk",
        "manifest_csv": rel,
        "manifest_absolute": str(path.resolve()),
        "rows_hint": None,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(result, rf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
