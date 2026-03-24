#!/usr/bin/env bash

set -euo pipefail

# Diretório raiz do projeto (TF_MODEL)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/dependences/train_environment.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Arquivo dependences/train_environment.yml não encontrado em ${ENV_FILE}."
  exit 1
fi

# Nome do ambiente definido no dependences/train_environment.yml
ENV_NAME="$(grep '^name:' "${ENV_FILE}" | awk '{print $2}')"

if [[ -z "${ENV_NAME}" ]]; then
  echo "Não foi possível determinar o nome do ambiente a partir de dependences/train_environment.yml."
  exit 1
fi

echo "Usando arquivo de ambiente de treino: ${ENV_FILE}"
echo "Nome do ambiente Conda de treino: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda não encontrado no PATH."
  echo "Instale Miniconda/Anaconda e garanta que o comando 'conda' esteja disponível."
  exit 1
fi

# Garante que o comando 'conda' esteja inicializado no shell
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Ambiente ${ENV_NAME} já existe. Atualizando a partir de dependences/train_environment.yml..."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "Criando ambiente de treino ${ENV_NAME} a partir de dependences/train_environment.yml..."
  conda env create -f "${ENV_FILE}"
fi

echo
echo "Ambiente de treino configurado com sucesso."
echo "Para ativar o ambiente de TREINO, execute:"
echo
echo "    conda activate ${ENV_NAME}"
echo
echo "Importante: este ambiente é apenas para TREINAMENTO."
echo "O ambiente de INFERÊNCIA permanece isolado em './dependences/dependences/train_environment.yml'."

