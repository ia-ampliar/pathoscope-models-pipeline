## Pipeline de Treinamento TensorFlow (MobileNetV2 + QAT)

Este projeto organiza o fluxo experimental do notebook `train-pipeline.ipynb` em um pipeline de treinamento modular e pronto para produção no diretório `TF_MODEL`.

### Estrutura de diretórios

- **Datas/**: imagens de entrada usadas no treinamento.
- **split/**: arquivos `train_data.csv`, `val_data.csv`, `test_data.csv` com os caminhos das imagens e rótulos.
- **models/**: checkpoints `.keras` e o modelo final quantizado `.tflite`.
- **metrics/**: históricos de treino em JSON e gráficos de treinamento (podem ser gerados via `train_utils.plot_history`).
- **callbacks/**: artefatos de callbacks, se necessários.
- **tfLite/**: pasta de apoio para experimentos com TFLite (compatível com o notebook).
- **scripts/**:
  - `config.py`: configuração de caminhos e hiperparâmetros.
  - `dataloader.py`: carregamento de CSVs e criação de `ImageDataGenerator`.
  - `modeling.py`: definição da arquitetura MobileNetV2 + topo denso.
  - `train_utils.py`: funções genéricas de treino, callbacks, salvamento de histórico e gráficos.
  - `qat.py`: funções específicas de Quantization Aware Training e exportação para TFLite.
  - `train.py`: orquestra o pipeline completo (baseline, fine-tune, avaliação e QAT).
- **environment.yml**: ambiente de **treino** (TensorFlow 2.10.1 + TF-MOT, CUDA 11.2).
- **dependences/environment.yml**: ambiente de **inferência** (TensorFlow 2.17.0 + libs de produção, mantido isolado).

### Ambientes: treino x produção

- **Treino**: usa estritamente `environment.yml` na raiz do projeto (`name: qat_tf_env`), com TensorFlow 2.10.1 e `tensorflow-model-optimization` para QAT.
- **Produção/Inferência**: usa `dependences/environment.yml` (`name: inference-env`), com TensorFlow 2.17.0 e dependências específicas de inferência.
- Os ambientes são **isolados**: o script de configuração de treino nunca toca o ambiente de inferência.

### Como configurar o ambiente de TREINO

1. No terminal (bash, WSL ou Git Bash), dentro de `TF_MODEL`, execute:

```bash
bash setup_train.sh
```

2. Após a criação/atualização do ambiente Conda, ative explicitamente:

```bash
conda activate qat_tf_env
```

3. Verifique a versão do TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Ela deve ser `2.10.1`, conforme definido em `environment.yml`.

### Como rodar o pipeline de treinamento

Com o ambiente de treino (`qat_tf_env`) ativado e os CSVs em `split/` preparados:

```bash
cd TF_MODEL
python -m scripts.train
```

Opções adicionais:

- **Desabilitar MirroredStrategy** (forçar treino em um único dispositivo):

```bash
python -m scripts.train --no-strategy
```

- **Desabilitar QAT** (rodar somente baseline + fine-tune FP32):

```bash
python -m scripts.train --no-qat
```

- **Rodar forçando somente CPU (evitar erros de CUDA/GPU)**:

  Se você estiver em uma máquina com GPU muito nova ou com incompatibilidade de driver/CUDA, pode forçar o TensorFlow a ignorar a GPU usando a variável de ambiente `CUDA_VISIBLE_DEVICES`:

  ```bash
  CUDA_VISIBLE_DEVICES="" python -m scripts.train
  ```

  Ou, se quiser desabilitar QAT enquanto testa:

  ```bash
  CUDA_VISIBLE_DEVICES="" python -m scripts.train --no-qat
  ```

### Etapas do pipeline

- **1. Carregamento de dados (`dataloader.py`)**
  - Lê `train_data.csv`, `val_data.csv`, `test_data.csv` de `split/`.
  - Cria `ImageDataGenerator` com normalização `preprocess_input` (MobileNetV2).
  - Gera `train_gen`, `val_gen`, `test_gen` com `class_mode='categorical'`.

- **2. Construção do modelo baseline (`modeling.py`)**
  - Backbone `MobileNetV2(weights='imagenet', include_top=False)` com `input_shape=(224, 224, 3)`.
  - Topo denso: `GlobalAveragePooling2D` → `Dropout(0.4)` → `Dense(128, relu)` → `Dropout(0.2)` → `Dense(num_classes, softmax)`.
  - Otimizador `Adam` com `learning_rate=1e-4`, perda `categorical_crossentropy`, métrica `accuracy`.

- **3. Treino baseline + fine-tune (`train_utils.py` + `modeling.py`)**
  - **Baseline (head only)**:
    - Backbone congelado (`base_trainable=False`).
    - Número de épocas iniciais: `0.4 * EPOCHS_BASELINE` (por padrão, 40 de 100).
    - Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
    - Salva o melhor checkpoint em `models/baseline_checkpoint.keras` e um artefato final `baseline_checkpoint_final.keras`.
    - Histórico em JSON na pasta `metrics/baseline/`.
  - **Fine-tuning**:
    - Descongela ~30% final das camadas convolucionais (exceto `BatchNormalization`), via `enable_fine_tuning`.
    - Recompila com `learning_rate=1e-5`.
    - Treina pelo restante das épocas (ex.: 60 de 100), salvando `fine_tuned_checkpoint.keras` e `fine_tuned_checkpoint_final.keras`.
    - Histórico em `metrics/fine_tune/`.

- **4. Avaliação no conjunto de teste**
  - Carrega o melhor modelo FP32 (`fine_tuned_checkpoint_final.keras`).
  - Avalia em `test_gen` e imprime `Test Loss` e `Test Accuracy`.

- **5. Quantization Aware Training + TFLite (`qat.py`)**
  - Carrega o modelo baseline final (`baseline_checkpoint_final.keras`).
  - Aplica QAT usando `tfmot.quantization.keras.quantize_model`.
  - Compila com `Adam(1e-5)`, `categorical_crossentropy`, métricas `accuracy` e `AUC`.
  - Treina com callbacks (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`), salvando `models/qat_baseline_checkpoint.keras`.
  - Converte o melhor modelo QAT para TFLite, com otimizações padrão (`Optimize.DEFAULT`), salvando `models/qat_baseline_final.tflite`.

### Escolha do "melhor" modelo

- O notebook original mostra que o modelo QAT atinge **acurácia de validação muito próxima (ou superior)** ao modelo FP32, com vantagem clara em:
  - **Tamanho do modelo** (TFLite quantizado é significativamente menor).
  - **Latência de inferência** em dispositivos com poucos recursos.
- Por isso, o pipeline considera o **modelo QAT exportado para TFLite (`models/qat_baseline_final.tflite`)** como o **modelo recomendado para produção**, enquanto o modelo FP32 fine-tunado (`fine_tuned_checkpoint_final.keras`) é mantido como referência para análise offline e comparações.

### Notas finais

- O ambiente de treino (`qat_tf_env`) é totalmente separado do ambiente de inferência (`inference-env` em `dependences/environment.yml`), evitando conflitos de versões do TensorFlow e de bibliotecas de sistema.
- Para pipelines de inferência, use apenas o ambiente de produção (`inference-env`) e os artefatos em `models/` (principalmente o `.tflite`).

### Como usar com Docker

Abaixo está um fluxo completo para empacotar o **pipeline de treinamento** em uma imagem Docker e executá-lo de forma reprodutível.

#### 1. Exemplo de `Dockerfile` (treino)

Crie um arquivo `Dockerfile` na raiz de `TF_MODEL` com um conteúdo semelhante a:

```dockerfile
FROM continuumio/miniconda3

WORKDIR /workspace

# Copia apenas os arquivos necessários para definir o ambiente
COPY environment.yml ./environment.yml

# Cria o ambiente de treino dentro da imagem
RUN conda env create -f environment.yml && conda clean -afy

# Ativa o ambiente por padrão em novos shells
SHELL ["bash", "-lc"]

# Define a variável de ambiente para facilitar a ativação
ENV CONDA_DEFAULT_ENV=qat_tf_env
ENV PATH=/opt/conda/envs/qat_tf_env/bin:$PATH

# Copia o restante do código do projeto
COPY . .

# Comando padrão: rodar o pipeline de treinamento
CMD ["python", "-m", "scripts.train"]
```

> **Observação**: o `Dockerfile` acima é voltado para o **ambiente de treino**. O ambiente de inferência (`dependences/environment.yml`) continua isolado e pode ter outro `Dockerfile` específico, se necessário.

#### 2. Construindo a imagem Docker

No diretório `TF_MODEL`, execute:

```bash
docker build -t tf-model-train:latest .
```

- **Espera-se ver**:
  - Download da imagem base `continuumio/miniconda3`.
  - Etapas de criação do ambiente Conda com base no `environment.yml`.
  - Cópia dos arquivos do projeto para `/workspace`.
  - Ao final, uma mensagem indicando sucesso, por exemplo:
    - `Successfully built <IMAGE_ID>`
    - `Successfully tagged tf-model-train:latest`

Você pode listar a imagem criada com:

```bash
docker images tf-model-train
```

#### 3. Executando o treinamento dentro do container

Para rodar o pipeline de treinamento conforme definido em `scripts/train.py`:

```bash
docker run --rm \
  -v /caminho/para/Datas:/workspace/Datas \
  -v /caminho/para/split:/workspace/split \
  -v /caminho/para/models:/workspace/models \
  -v /caminho/para/metrics:/workspace/metrics \
  tf-model-train:latest
```

- **Montagens (`-v`) recomendadas**:
  - `Datas/` e `split/` como **entrada somente leitura** (se desejar, usando `:ro`).
  - `models/` e `metrics/` como saída persistente para capturar checkpoints e históricos fora do container.

- **Saída esperada**:
  - Logs do TensorFlow mostrando:
    - Detecção de GPU(s) (se mapeadas para o container).
    - Sumário do modelo (MobileNetV2 + head).
    - Progresso de épocas com métricas `loss`, `accuracy` e `val_*`.
  - Mensagens de salvamento de checkpoints, por exemplo:
    - `Epoch X: val_loss improved from ... saving model to /workspace/models/baseline_checkpoint.keras`
    - `Epoch Y: val_loss improved from ... saving model to /workspace/models/fine_tuned_checkpoint.keras`
  - Ao final:
    - Impressão da avaliação em teste:
      - `FP32 - Test Loss: <valor>, Test Accuracy: <valor>`
    - Se QAT estiver habilitado (padrão), logs de treino do modelo quantizado e mensagem de exportação:
      - `QAT TFLite salvo em: /workspace/models/qat_baseline_final.tflite`

#### 4. Rodando com parâmetros adicionais

Você pode sobrescrever o comando padrão do container para passar flags ao script de treino. Por exemplo, para desabilitar QAT:

```bash
docker run --rm \
  -v /caminho/para/Datas:/workspace/Datas \
  -v /caminho/para/split:/workspace/split \
  -v /caminho/para/models:/workspace/models \
  -v /caminho/para/metrics:/workspace/metrics \
  tf-model-train:latest \
  python -m scripts.train --no-qat
```

De forma análoga, você pode adicionar `--no-strategy` se quiser forçar treino em um único dispositivo dentro do container.
