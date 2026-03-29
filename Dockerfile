FROM python:3.10-slim

WORKDIR /app

# Dependências de sistema necessárias para openslide / opencv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenslide0 \
        libglib2.0-0 \
        libgl1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copia arquivos essenciais E a pasta do código-fonte
COPY dependences/pyproject.toml dependences/environment.yml README.md ./ 
COPY src/ ./src/

# Instala o projeto e dependências (agora o pip vai encontrar a pasta src/)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Agora copia o restante do código (inclui modules/, models/, data/)
COPY . .

# Diretório padrão para saídas de inferência (para mapear volume)
ENV OUTPUT_DIR="/app/output"
RUN mkdir -p "${OUTPUT_DIR}"

# Adiciona a raiz do projeto ao path do Python para achar 'modules' e 'src'
ENV PYTHONPATH="/app"

# O script src/infer.py lê IMAGE_PATH e escreve em OUTPUT_DIR.
ENTRYPOINT ["python", "src/infer.py"]