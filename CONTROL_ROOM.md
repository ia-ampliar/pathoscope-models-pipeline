# Control Room (Web)

Interface web para orquestrar **split**, **treino** (TensorFlow), **TCGA/GDC** (download WSI, manifest em disco, `label_file` via planilha) e **inferência** (WSI + TFLite), com telemetria de treino por WebSocket.

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

## Endpoints principais

- `GET /api/schema` — JSON Schema e defaults para formulários dinâmicos.
- `POST /api/train/jobs` — corpo `{ "training": { ... } }` (mesmos campos que `TrainingJobConfig` em `server/schemas.py`).
- `WebSocket /api/train/jobs/{id}/stream` — eventos `epoch`, `stage_start`, `completed`, `error`, etc.
- `POST /api/train/jobs/{id}/stop` — encerra o subprocesso de treino.
- `POST /api/split/jobs` — `{ "split": { ... } }`.
- `POST /api/tcga/jobs` — `multipart/form-data`: `operation` (`download` | `manifest` | `labels`), `config_json` (string), e ficheiros opcionais `ids_csv`, `manifest_csv`, `sheets_csv`. Ver `TcgaDownloadJobConfig`, `TcgaManifestJobConfig`, `TcgaLabelsJobConfig` em `server/schemas.py`).
- `GET /api/tcga/jobs/{id}` — estado e `result` (paths de `wsi_manifest.csv`, `label_file.csv`, etc.).
- `POST /api/inference/jobs` — `multipart/form-data`: `inference_json` (string JSON) e opcionalmente `file` (`.svs`).

### TCGA (fluxo na UI)

1. **Download** — CSV com coluna de IDs de caso (padrão `case_submitter_id`); os `.svs` são descarregados via API GDC para `data/<caso>/…` e é gerado `wsi_manifest.csv` na raiz (ou caminho configurado). Dados controlados: defina `GDC_TOKEN` no ambiente do servidor.
2. **Manifest (disco)** — se `data/` já existe, gera `wsi_manifest.csv` sem rede.
3. **Label file** — URL pública do Google Sheets (export CSV) **ou** upload de CSV local; cruza com o manifest usando `Patient ID` e `Subtype`; saída `split/label_file.csv` com colunas `Image_path` e `Label` (compatível com `create_split`).

CSV de exemplo para IDs:

```csv
case_submitter_id
TCGA-BR-4191
TCGA-BR-4257
```

Artefatos e logs temporários ficam em `server_state/` (ignorado no git).

## Docker

```bash
docker build -f Dockerfile.control-room -t pathoscope-control-room .
```

Instale as dependências de ML na imagem (TensorFlow, OpenSlide, etc.) antes de usar treino/inferência em produção; o `Dockerfile` documenta um ponto de extensão.

## Ambientes TF (treino vs inferência)

O repositório descreve ambientes Conda distintos para treino e inferência. Em um **único host com GPU**, use o ambiente que contiver TensorFlow compatível com **treino + QAT + inferência TFLite**, ou rode apenas a API de **orquestração** no mesmo Python e mantenha versões alinhadas aos scripts testados no README principal.
