"""
Cliente mínimo para a API REST do NCI GDC (ficheiros .SVS).
Documentação: https://docs.gdc.cancer.gov/API/Getting_Started/
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def build_svs_filters(case_submitter_ids: list[str], only_open_access: bool = True) -> dict[str, Any]:
    """Filtro: casos dados + formato SVS (lâmina digital TCGA)."""
    content: list[dict[str, Any]] = [
        {
            "op": "in",
            "content": {"field": "cases.submitter_id", "value": case_submitter_ids},
        },
        {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
    ]
    if only_open_access:
        content.append({"op": "=", "content": {"field": "access", "value": "open"}})
    return {"op": "and", "content": content}


def iter_file_hits(
    filters: dict[str, Any],
    fields: str | list[str],
    page_size: int,
    token: str | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Pagina ``POST /files`` e devolve cada hit bruto da API.
    A API GDC aceita ``fields`` como string separada por vírgulas (recomendado).
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Auth-Token"] = token

    fields_str = ",".join(fields) if isinstance(fields, list) else fields

    offset = 0
    while True:
        body = {
            "filters": filters,
            "fields": fields_str,
            "size": page_size,
            "from": offset,
        }
        r = requests.post(GDC_FILES_URL, json=body, headers=headers, timeout=120)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data") or {}
        hits = data.get("hits") or []
        if not hits:
            break
        for h in hits:
            yield h
        pag = data.get("pagination") or {}
        total = int(pag.get("total") or 0)
        offset += len(hits)
        if offset >= total:
            break


def normalize_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extrai file_id, file_name, file_size, case_submitter_id a partir de um hit.
    O GDC pode devolver ``id`` ou estrutura nested em ``cases``.
    """
    file_id = hit.get("file_id") or hit.get("id")
    file_name = hit.get("file_name")
    if not file_id or not file_name:
        return None

    file_size = hit.get("file_size")
    if file_size is not None:
        try:
            file_size = int(file_size)
        except (TypeError, ValueError):
            file_size = None

    case_submitter_id = None
    cases = hit.get("cases")
    if isinstance(cases, list) and cases:
        first = cases[0]
        if isinstance(first, dict):
            case_submitter_id = first.get("submitter_id") or first.get("case_submitter_id")

    if not case_submitter_id:
        subs = hit.get("cases.submitter_id")
        if isinstance(subs, str):
            case_submitter_id = subs
        elif isinstance(subs, list) and subs:
            case_submitter_id = subs[0]

    if not case_submitter_id:
        logger.warning("Hit sem cases.submitter_id ignorado: %s", file_id)
        return None

    return {
        "file_id": file_id,
        "file_name": file_name,
        "file_size": file_size,
        "md5sum": hit.get("md5sum"),
        "case_submitter_id": case_submitter_id,
    }


def download_file_to_path(
    file_id: str,
    dest: Path,
    token: str | None = None,
    timeout: int = 3600,
    expected_size: int | None = None,
) -> None:
    """GET /data/{file_id} com escrita em streaming."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GDC_DATA_URL}/{file_id}"
    headers = {}
    if token:
        headers["X-Auth-Token"] = token

    tmp = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)

    if expected_size is not None and dest.is_file():
        actual = dest.stat().st_size
        if actual != expected_size:
            raise IOError(f"Tamanho incorreto para {dest}: esperado {expected_size}, obtido {actual}")


def download_with_retries(
    file_id: str,
    dest: Path,
    token: str | None = None,
    timeout: int = 3600,
    expected_size: int | None = None,
    max_retries: int = 3,
) -> None:
    dest = Path(dest)
    for attempt in range(max_retries):
        try:
            download_file_to_path(
                file_id, dest, token=token, timeout=timeout, expected_size=expected_size
            )
            return
        except Exception as e:
            logger.warning("Download %s tentativa %d/%s: %s", file_id, attempt + 1, max_retries, e)
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            partial = dest.with_suffix(dest.suffix + ".partial")
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
