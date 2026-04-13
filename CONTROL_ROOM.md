# Control Room (Web)

Interface web para orquestrar **split**, **TCGA/GDC** (download WSI, manifest a partir de disco, `label_file` via planilha), **treino** (TensorFlow) e **inferência** (WSI + TFLite), com telemetria de treino por WebSocket.

## Requisitos

- Python 3.10+ com as mesmas dependências do pipeline (TensorFlow, `tensorflow-model-optimization` para QAT, etc.).
- `pip install -r requirements-server.txt`
- Node 20+ (apenas para desenvolvimento ou build do frontend): em `web/` execute `npm install` e `npm run build`.

## Executar

Na raiz do repositório (com `PYTHONPATH` apontando para a raiz, ou a partir dela):

```bash
# API + UI estática (após npm run build em web/)
PYTHONPATH=. python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Abra `http://localhost:8000`. Em desenvolvimento do frontend:

```bash
cd web && npm run dev
```

O Vite em `http://localhost:5173` faz proxy de `/api` e WebSocket de treino para `http://127.0.0.1:8000` (inicie a API em paralelo).

## TCGA / GDC (`modules.tcga_dataset`)

A aba **TCGA / Dados** chama os mesmos fluxos que o CLI (`python -m modules.tcga_dataset`). O download usa a **API REST do GDC** (não automação do browser).

- **CSV de IDs:** coluna por defeito `case_submitter_id` com valores tipo `TCGA-BR-4191` (configurável no formulário).
- **`GDC_TOKEN`:** defina no ambiente do servidor se precisar de ficheiros controlados e marcar “incluir controlados” no formulário.
- **Planilha (label file):** URL Google Sheets com export CSV público (ou CSV no servidor); colunas `Patient ID` e `Subtype` por defeito. O ID do caso na pasta `data/<caso>/` deve fazer match com `Patient ID`.

### Endpoints TCGA

- `POST /api/tcga/download` — `multipart`: ficheiro `ids_csv` + `config_json` (`TcgaDownloadJobConfig`).
- `POST /api/tcga/manifest-from-disk` — JSON `TcgaManifestDiskJobConfig` (percorre `data/` e grava manifest).
- `POST /api/tcga/labels` — `multipart`: `config_json` (`TcgaLabelsJobConfig`) + opcional `manifest_file`.
- `GET /api/tcga/jobs/{id}` — estado e `result` (paths dos artefatos).
- `GET /api/files/repo/{path}` — descarregar ficheiros sob a raiz do repo (ex.: `wsi_manifest.csv`, `split/label_file.csv`).

## Endpoints principais

- `GET /api/schema` — JSON Schema e defaults para formulários dinâmicos.
- `POST /api/train/jobs` — corpo `{ "training": { ... } }` (mesmos campos que `TrainingJobConfig` em `server/schemas.py`).
- `WebSocket /api/train/jobs/{id}/stream` — eventos `epoch`, `stage_start`, `completed`, `error`, etc.
- `POST /api/train/jobs/{id}/stop` — encerra o subprocesso de treino.
- `POST /api/split/jobs` — `{ "split": { ... } }`.
- `POST /api/inference/jobs` — `multipart/form-data`: `inference_json` (string JSON) e opcionalmente `file` (`.svs`).

Artefatos e logs temporários ficam em `server_state/` (ignorado no git). Uploads TCGA (CSV de IDs, manifest opcional) também em `server_state/uploads/`.

## Docker

```bash
docker build -f Dockerfile.control-room -t pathoscope-control-room .
```

Instale as dependências de ML na imagem (TensorFlow, OpenSlide, etc.) antes de usar treino/inferência em produção; o `Dockerfile` documenta um ponto de extensão.

## Ambientes TF (treino vs inferência)

O repositório descreve ambientes Conda distintos para treino e inferência. Em um **único host com GPU**, use o ambiente que contiver TensorFlow compatível com **treino + QAT + inferência TFLite**, ou rode apenas a API de **orquestração** no mesmo Python e mantenha versões alinhadas aos scripts testados no README principal.
