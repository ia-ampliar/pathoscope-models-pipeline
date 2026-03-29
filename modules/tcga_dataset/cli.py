#!/usr/bin/env python3
"""
CLI: download TCGA WSI, manifest a partir de disco, label_file a partir de planilha.

Exemplos::

    python -m modules.tcga_dataset download --ids-csv cases.csv --data-root data
    python -m modules.tcga_dataset manifest --data-root data --output wsi_manifest.csv
    python -m modules.tcga_dataset labels --manifest wsi_manifest.csv --sheets-url 'https://docs.google.com/...'
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import modules.config as app_config

from .config import TcgaDatasetConfig
from .download import download_and_write_manifest
from .labels import build_label_file
from .manifest import build_wsi_manifest_from_disk


def _cfg_from_args(ns: argparse.Namespace) -> TcgaDatasetConfig:
    root = Path(ns.root_dir).resolve() if ns.root_dir else app_config.ROOT_DIR
    data_root = Path(ns.data_root).resolve() if ns.data_root else root / "data"
    cfg = TcgaDatasetConfig(root_dir=root, data_root=data_root)
    updates: dict = {"only_open_access": not ns.include_controlled}
    if getattr(ns, "ids_csv", None):
        updates["ids_csv"] = Path(ns.ids_csv).resolve()
    if getattr(ns, "case_column", None):
        updates["case_id_column"] = ns.case_column
    if getattr(ns, "manifest_output", None):
        updates["wsi_manifest_csv"] = Path(ns.manifest_output).resolve()
    if getattr(ns, "label_output", None):
        updates["label_file_csv"] = Path(ns.label_output).resolve()
    if getattr(ns, "sheets_url", None):
        updates["sheets_url"] = ns.sheets_url
    if getattr(ns, "patient_id_column", None):
        updates["patient_id_column"] = ns.patient_id_column
    if getattr(ns, "subtype_column", None):
        updates["subtype_column"] = ns.subtype_column
    if getattr(ns, "gdc_page_size", None):
        updates["gdc_files_page_size"] = ns.gdc_page_size
    return replace(cfg, **updates)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TCGA/GDC: WSI, manifest e labels.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Raiz do repositório (padrão: config.ROOT_DIR).")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Pasta com subpastas por caso (padrão: <root>/data).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Consulta GDC e descarrega .svs; grava manifest CSV.")
    p_dl.add_argument("--ids-csv", type=Path, required=True, help="CSV com IDs de caso TCGA.")
    p_dl.add_argument(
        "--case-column",
        type=str,
        default="case_submitter_id",
        help="Nome da coluna de case submitter ID (ex.: TCGA-BR-4191).",
    )
    p_dl.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Onde gravar o manifest (padrão: <root>/wsi_manifest.csv).",
    )
    p_dl.add_argument(
        "--include-controlled",
        action="store_true",
        help="Incluir ficheiros não abertos (definir GDC_TOKEN).",
    )
    p_dl.add_argument("--gdc-page-size", type=int, default=500)

    p_man = sub.add_parser("manifest", help="Apenas percorre o disco e grava manifest (sem GDC).")
    p_man.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de saída (padrão: <root>/wsi_manifest.csv).",
    )

    p_lab = sub.add_parser("labels", help="Cruza manifest com planilha (Google Sheets ou CSV local).")
    p_lab.add_argument("--manifest", type=Path, required=True)
    g = p_lab.add_mutually_exclusive_group(required=True)
    g.add_argument("--sheets-url", type=str, help="URL da planilha Google (export CSV público).")
    g.add_argument("--sheets-csv", type=Path, help="Ficheiro CSV local (idem colunas da planilha).")
    p_lab.add_argument(
        "--label-output",
        type=Path,
        default=None,
        help="Saída label_file (padrão: split/label_file.csv).",
    )
    p_lab.add_argument("--patient-id-column", type=str, default="Patient ID")
    p_lab.add_argument("--subtype-column", type=str, default="Subtype")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "download":
        ns = argparse.Namespace(
            root_dir=args.root_dir,
            data_root=args.data_root,
            ids_csv=args.ids_csv,
            case_column=args.case_column,
            manifest_output=args.manifest_output,
            include_controlled=args.include_controlled,
            sheets_url="",
            gdc_page_size=args.gdc_page_size,
        )
        cfg = _cfg_from_args(ns)
        if args.manifest_output is None:
            cfg = replace(cfg, wsi_manifest_csv=cfg.root_dir / "wsi_manifest.csv")
        download_and_write_manifest(cfg)

    elif args.command == "manifest":
        root = Path(args.root_dir).resolve() if args.root_dir else app_config.ROOT_DIR
        data_root = Path(args.data_root).resolve() if args.data_root else root / "data"
        out = Path(args.output).resolve() if args.output else root / "wsi_manifest.csv"
        build_wsi_manifest_from_disk(data_root, root, out, path_column="image_path")
        # manifest from disk only has image_path column per plan - add empty columns? Plan says paths only.
        # For labels step, patient_id inferred from path. OK.

    elif args.command == "labels":
        root = Path(args.root_dir).resolve() if args.root_dir else app_config.ROOT_DIR
        sheet_ref = args.sheets_url if args.sheets_url else str(Path(args.sheets_csv).resolve())
        out = Path(args.label_output).resolve() if args.label_output else app_config.SPLIT_DIR / "label_file.csv"
        build_label_file(
            manifest_csv=Path(args.manifest).resolve(),
            sheets_ref=sheet_ref,
            output_csv=out,
            repo_root=root,
            manifest_path_column="image_path",
            patient_id_column=args.patient_id_column,
            subtype_column=args.subtype_column,
        )
    else:
        parser.error("Comando desconhecido.")


if __name__ == "__main__":
    main()
