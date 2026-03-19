#!/usr/bin/env bash
# =============================================================================
# Fluxo completo: Inferência (Docker) + Reconstrução da imagem a partir do NPZ
#
# Garante que os artefatos gerados no diretório local (mapeado do Docker)
# tenham permissões corretas para operações locais (reconstruir, copy, remove).
#
# Uso:
#   ./run_inference_with_reconstruct.sh
#   ./run_inference_with_reconstruct.sh --image /caminho/para/imagem.svs
#
# Requer: docker, docker compose (ou docker-compose)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Carrega variáveis do .env se existir
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Configurações (valores padrão compatíveis com .env.example)
DATA_DIR="${DATA_DIR:-./img_test/cancer}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
IMAGE_PATH="${IMAGE_PATH:-}"
IMAGE_NAME="${1:-}"  # Permite passar imagem como primeiro arg
DOCKER_IMAGE="${DOCKER_IMAGE:-histology-model-inference:latest}"
OUTPUT_HEATMAP_MODE="${OUTPUT_HEATMAP_MODE:-matrix_only}"

# Resolve caminhos absolutos para montagem no Docker
DATA_DIR_ABS="$(cd "${DATA_DIR}" 2>/dev/null && pwd || echo "${SCRIPT_DIR}/${DATA_DIR}")"
OUTPUT_DIR_ABS="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

# Determina UID/GID do usuário atual (evita Permission denied nos artefatos)
# No Linux: arquivos criados no volume terão ownership do host
# No Windows/Mac: id pode não existir, usa fallback
DOCKER_USER=""
if command -v id >/dev/null 2>&1; then
  _uid="$(id -u 2>/dev/null || true)"
  _gid="$(id -g 2>/dev/null || true)"
  if [[ -n "${_uid:-}" && -n "${_gid:-}" && "${_uid}" != "0" ]]; then
    DOCKER_USER="-u ${_uid}:${_gid}"
  fi
fi

echo "[INFO] DATA_DIR: ${DATA_DIR_ABS}"
echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR_ABS}"
echo "[INFO] Modo heatmap: ${OUTPUT_HEATMAP_MODE}"

# Permite sobrescrever IMAGE_PATH via argumento
if [[ -n "${IMAGE_NAME}" && "${IMAGE_NAME}" != "--"* ]]; then
  IMAGE_PATH="${IMAGE_NAME}"
fi

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "[ERRO] IMAGE_PATH não definido. Defina no .env ou passe como argumento:"
  echo "  ./run_inference_with_reconstruct.sh /caminho/para/imagem.svs"
  exit 1
fi

# Caminho da imagem DENTRO do container (/data = DATA_DIR montado)
IMAGE_IN_CONTAINER="/data/$(basename "${IMAGE_PATH}")"

# Verifica se a imagem existe no diretório de dados
if [[ ! -f "${DATA_DIR_ABS}/$(basename "${IMAGE_PATH}")" ]]; then
  # Tenta o caminho absoluto direto
  if [[ -f "${IMAGE_PATH}" ]]; then
    IMAGE_IN_CONTAINER="/data/$(basename "${IMAGE_PATH}")"
    # Se a imagem está fora de DATA_DIR, precisamos montar o dir pai
    IMAGE_PARENT="$(cd "$(dirname "${IMAGE_PATH}")" && pwd)"
    DATA_DIR_ABS="${IMAGE_PARENT}"
  else
    echo "[ERRO] Imagem não encontrada: ${IMAGE_PATH}"
    echo "  Coloque a imagem em ${DATA_DIR_ABS} ou use o caminho absoluto."
    exit 1
  fi
fi

echo "[1/2] Executando inferência no Docker (gera NPZ)..."
docker run --rm \
  ${DOCKER_USER} \
  -e "IMAGE_PATH=${IMAGE_IN_CONTAINER}" \
  -e "OUTPUT_HEATMAP_MODE=${OUTPUT_HEATMAP_MODE}" \
  -e "OUTPUT_DIR=/app/output" \
  -v "${DATA_DIR_ABS}:/data:ro" \
  -v "${OUTPUT_DIR_ABS}:/app/output" \
  "${DOCKER_IMAGE}"

echo "[2/2] Reconstruindo imagem sobreposta a partir do NPZ..."

# Encontra todos os NPZ gerados
NPZ_FILES=()
while IFS= read -r -d '' f; do
  NPZ_FILES+=("$f")
done < <(find "${OUTPUT_DIR_ABS}" -name "heatmap_matrix_*.npz" -print0 2>/dev/null || true)

if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then
  echo "[AVISO] Nenhum arquivo NPZ encontrado em ${OUTPUT_DIR_ABS}"
  echo "  A inferência pode ter falhado ou o diretório está vazio."
  exit 1
fi

for npz_host in "${NPZ_FILES[@]}"; do
  npz_name="$(basename "${npz_host}")"
  # Extrai base do nome do npz: heatmap_matrix_IMAGE_ID_BASE_qat.npz -> BASE
  # O npz contém a chave 'base'; usamos Python para extrair
  PYTHON_CMD="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo 'python3')"
  base=$("${PYTHON_CMD}" -c "
import numpy as np
import sys
if len(sys.argv) < 2:
    sys.exit(1)
d = np.load(sys.argv[1], allow_pickle=True)
print(str(d['base']))
" "${npz_host}" 2>/dev/null) || base=""

  if [[ -z "${base}" ]]; then
    echo "[AVISO] Não foi possível extrair 'base' do NPZ ${npz_name}, pulando."
    continue
  fi

  # Procura a imagem WSI correspondente (extensões comuns)
  image_host=""
  for ext in svs SVS tif tiff; do
    candidate="${DATA_DIR_ABS}/${base}.${ext}"
    if [[ -f "${candidate}" ]]; then
      image_host="${candidate}"
      break
    fi
  done

  if [[ -z "${image_host}" ]]; then
    # Tenta no mesmo diretório da imagem original
    for ext in svs SVS tif tiff; do
      candidate="$(dirname "${IMAGE_PATH}")/${base}.${ext}"
      if [[ -f "${candidate}" ]]; then
        image_host="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${image_host}" ]]; then
    echo "[AVISO] Imagem não encontrada para base=${base}, pulando reconstrução."
    continue
  fi

  # Caminhos DENTRO do container (volumes montados)
  npz_rel="${npz_host#${OUTPUT_DIR_ABS}/}"
  npz_rel="${npz_rel//\\//}"
  npz_container="/app/output/${npz_rel}"
  image_container="/data/$(basename "${image_host}")"

  # Diretório da imagem para montar como /data
  img_parent="$(cd "$(dirname "${image_host}")" && pwd)"

  docker run --rm \
    ${DOCKER_USER} \
    -v "${img_parent}:/data:ro" \
    -v "${OUTPUT_DIR_ABS}:/app/output" \
    -w /app \
    "${DOCKER_IMAGE}" \
    python scripts/heatmap_reconstruct_from_npz.py \
      --npz_path "${npz_container}" \
      --image_path "${image_container}" \
      --output_dir "/app/output"

  echo "  [OK] Reconstruído: $(basename "${npz_host}" .npz).jpg"
done

echo ""
echo "[CONCLUÍDO] Artefatos em: ${OUTPUT_DIR_ABS}"
echo "  - NPZ (matriz heatmap)"
echo "  - JPG (imagem sobreposta reconstruída)"
echo "  - JSON (métricas)"
echo ""
echo "Você pode copiar, remover ou processar os arquivos localmente sem restrições de permissão."
