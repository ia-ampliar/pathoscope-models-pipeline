# Pipeline pathoscope-models-pipeline: alvos por etapa.
# Requer GNU Make + bash no PATH (Git Bash ou WSL no Windows).

SHELL := bash
.SHELLFLAGS := -eu -o pipefail

.DEFAULT_GOAL := help

PYTHON ?= python
DATAS_DIR ?= datas
SPLIT_ARGS ?=
TRAIN_ARGS ?=

DATA_ROOT ?= data
TCGA_IDS_CSV ?=
TCGA_CASE_COLUMN ?= case_submitter_id
TCGA_MANIFEST ?=
TCGA_MANIFEST_OUT ?=
TCGA_SHEETS_URL ?=
TCGA_SHEETS_CSV ?=
TCGA_LABEL_OUTPUT ?=
TCGA_EXTRA_ARGS ?=

WSI_CONFIG ?= configs/wsi_pipeline.example.json
WSI_EXTRA_ARGS ?=

IMAGE_PATH ?=
NPZ_PATH ?=
HEATMAP_OUT_DIR ?= output

.PHONY: help setup-train split train train-cpu \
	tcga-download tcga-manifest tcga-labels \
	wsi-build infer heatmap-reconstruct pipeline-patch \
	control-room-api control-room-web

help:
	@echo "Pathoscope models pipeline — alvos Make"
	@echo ""
	@echo "Pré-requisitos: GNU Make e bash (Git Bash ou WSL no Windows)."
	@echo "Após setup-train: conda activate com o nome em dependences/train_environment.yml"
	@echo ""
	@echo "Alvos:"
	@echo "  help                — esta mensagem (padrão)"
	@echo "  setup-train         — bash setup_train.sh (ambiente Conda de treino)"
	@echo "  split               — CSVs em split/ a partir de DATAS_DIR (padrão: datas)"
	@echo "  train               — python -m src.train"
	@echo "  train-cpu           — treino com CUDA_VISIBLE_DEVICES vazio"
	@echo "  pipeline-patch      — split depois train (dados já organizados em DATAS_DIR)"
	@echo "  tcga-download       — exige TCGA_IDS_CSV"
	@echo "  tcga-manifest       — manifest a partir de DATA_ROOT"
	@echo "  tcga-labels         — exige TCGA_MANIFEST e TCGA_SHEETS_URL ou TCGA_SHEETS_CSV"
	@echo "  wsi-build           — tiling WSI + pré-processamento (WSI_CONFIG)"
	@echo "  infer               — exige IMAGE_PATH"
	@echo "  heatmap-reconstruct — exige NPZ_PATH e IMAGE_PATH"
	@echo "  control-room-api    — PYTHONPATH=. uvicorn server.main:app :8000"
	@echo "  control-room-web    — cd web && npm run dev (com API em :8000)"
	@echo ""
	@echo "Exemplos:"
	@echo "  make split DATAS_DIR=datas SPLIT_ARGS=\"--test_size 0.2\""
	@echo "  make train TRAIN_ARGS=\"--no-qat\""
	@echo "  make tcga-download TCGA_IDS_CSV=tcga_case_ids.csv"
	@echo "  make tcga-manifest DATA_ROOT=data TCGA_MANIFEST_OUT=wsi_manifest.csv"
	@echo "  make tcga-labels TCGA_MANIFEST=wsi_manifest.csv TCGA_SHEETS_URL=\"https://...\""
	@echo "  make wsi-build WSI_CONFIG=configs/wsi_pipeline.example.json"
	@echo "  make infer IMAGE_PATH=/caminho/imagem.svs"
	@echo "  make heatmap-reconstruct NPZ_PATH=out.npz IMAGE_PATH=imagem.svs HEATMAP_OUT_DIR=output"

setup-train:
	bash setup_train.sh

split:
	$(PYTHON) -m modules.split.create_split $(DATAS_DIR) $(SPLIT_ARGS)

train:
	$(PYTHON) -m src.train $(TRAIN_ARGS)

train-cpu:
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m src.train $(TRAIN_ARGS)

pipeline-patch: split train

tcga-download:
	@test -n "$(TCGA_IDS_CSV)" || { echo "Defina TCGA_IDS_CSV=... (CSV com IDs de caso)."; exit 1; }
	$(PYTHON) -m modules.tcga_dataset download \
		--ids-csv $(TCGA_IDS_CSV) \
		--case-column $(TCGA_CASE_COLUMN) \
		$(if $(TCGA_MANIFEST_OUT),--manifest-output $(TCGA_MANIFEST_OUT),) \
		$(TCGA_EXTRA_ARGS)

tcga-manifest:
	$(PYTHON) -m modules.tcga_dataset manifest \
		--data-root $(DATA_ROOT) \
		$(if $(TCGA_MANIFEST_OUT),--output $(TCGA_MANIFEST_OUT),)

tcga-labels:
	@test -n "$(TCGA_MANIFEST)" || { echo "Defina TCGA_MANIFEST=... (CSV de manifest)."; exit 1; }
	@if [ -n "$(TCGA_SHEETS_URL)" ]; then :; \
	elif [ -n "$(TCGA_SHEETS_CSV)" ]; then :; \
	else echo "Defina TCGA_SHEETS_URL ou TCGA_SHEETS_CSV."; exit 1; fi
	$(PYTHON) -m modules.tcga_dataset labels \
		--manifest $(TCGA_MANIFEST) \
		$(if $(TCGA_SHEETS_URL),--sheets-url $(TCGA_SHEETS_URL),--sheets-csv $(TCGA_SHEETS_CSV)) \
		$(if $(TCGA_LABEL_OUTPUT),--label-output $(TCGA_LABEL_OUTPUT),)

wsi-build:
	$(PYTHON) modules/wsi_pipeline/build_wsi_dataset.py \
		--config $(WSI_CONFIG) \
		$(WSI_EXTRA_ARGS)

infer:
	@test -n "$(IMAGE_PATH)" || { echo "Defina IMAGE_PATH (WSI .svs)."; exit 1; }
	$(PYTHON) -m src.infer --image_path $(IMAGE_PATH)

heatmap-reconstruct:
	@test -n "$(NPZ_PATH)" || { echo "Defina NPZ_PATH."; exit 1; }
	@test -n "$(IMAGE_PATH)" || { echo "Defina IMAGE_PATH (WSI original)."; exit 1; }
	$(PYTHON) -m modules.heatmap.heatmap_reconstruct_from_npz \
		--npz_path $(NPZ_PATH) \
		--image_path $(IMAGE_PATH) \
		--output_dir $(HEATMAP_OUT_DIR)

control-room-api:
	PYTHONPATH=. $(PYTHON) -m uvicorn server.main:app --host 0.0.0.0 --port 8000

control-room-web:
	cd web && npm run dev
