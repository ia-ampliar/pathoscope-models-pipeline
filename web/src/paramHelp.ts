export type ParamHelp = {
  what: string;
  how: string;
  tradeoffs: string;
  source: string;
  essential?: boolean;
};

const H: Record<string, ParamHelp> = {
  /* ── Training ─────────────────────────────────────── */
  batch_size: {
    what: "Número de patches processados em paralelo antes de cada passo de gradiente. Define o uso de VRAM e a estabilidade do gradiente.",
    how: "Comece com o maior valor que caiba na VRAM (potências de 2: 16, 32, 64, 128). MobileNetV2 a 224×224 em 12 GB cabe bem em 64. Se OOM, divida por 2.",
    tradeoffs:
      "↑ treino mais estável, melhor uso de GPU, mais VRAM necessária. ↓ gradientes mais ruidosos (pode escapar de mínimos locais), menos VRAM, treino mais lento por época.",
    source: "modules/config/settings.py → BATCH_SIZE.",
    essential: true,
  },
  epochs_baseline: {
    what: "Número total de épocas planejadas para baseline + fine-tune. O early stopping interrompe antes se não houver melhora por `patience` épocas consecutivas.",
    how: "100 é bom ponto de partida. Para datasets grandes (>10k patches), 50 podem bastar. Para datasets pequenos, 200+ podem ser necessários.",
    tradeoffs:
      "↑ mais oportunidade de convergência, treino mais longo. ↓ treino rápido, pode não convergir completamente.",
    source: "Dividido internamente por initial_epochs_fraction: ~40% baseline, ~60% fine-tune.",
    essential: true,
  },
  initial_epochs_fraction: {
    what: "Fração de epochs_baseline destinada à fase baseline (cabeça densa, backbone congelado). O restante vai para fine-tune.",
    how: "0.4 (40%) é o padrão. Com 100 épocas totais: 40 baseline + 60 fine-tune. Aumente se a cabeça precisar de mais convergência antes de descongelar o backbone.",
    tradeoffs:
      "↑ mais épocas de cabeça antes de descongelar, backbone menos influenciado. ↓ mais épocas de fine-tune, possível catástrofe de esquecimento se LR for alto.",
    source: "modules/config/settings.py → INITIAL_EPOCHS_FRACTION.",
    essential: false,
  },
  fine_tune_percent: {
    what: "Fração final das camadas convolucionais do MobileNetV2 descongeladas durante o fine-tune. 0.3 = 30% das camadas do topo.",
    how: "0.3 é conservador e funciona bem para domínios próximos ao ImageNet. Para histologia muito distinta, tente 0.5–0.7. Para datasets pequenos, mantenha baixo.",
    tradeoffs:
      "↑ mais camadas adaptadas ao domínio, risco de esquecimento catastrófico com LR alto. ↓ menos adaptação, treino mais rápido, mais seguro.",
    source: "modules/config/settings.py → FINE_TUNE_PERCENT.",
    essential: true,
  },
  lr_baseline: {
    what: "Learning rate para a fase baseline (apenas cabeça densa treinada, backbone congelado).",
    how: "1e-4 é o padrão para Adam com cabeça nova. Valores acima de 1e-3 causam instabilidade na loss logo nos primeiros epochs.",
    tradeoffs:
      "↑ convergência inicial mais rápida, risco de oscilação. ↓ convergência mais lenta e estável.",
    source: "Padrão Adam para transfer learning (He et al., 2015).",
    essential: true,
  },
  lr_fine: {
    what: "Learning rate para o fine-tune (backbone parcialmente descongelado). Deve ser sempre menor que lr_baseline.",
    how: "1e-5 é o padrão. Regra geral: 1/10 do lr_baseline. Nunca use o mesmo LR do baseline — destruirá os pesos pré-treinados.",
    tradeoffs:
      "↑ adaptação mais agressiva, risco de catástrofe de esquecimento. ↓ adaptação lenta, mais segura.",
    source: "modules/config/settings.py → LR_FINE.",
    essential: true,
  },
  lr_qat: {
    what: "Learning rate para a fase QAT (Quantization-Aware Training). Geralmente igual ou menor que lr_fine.",
    how: "1e-5 é o padrão. O QAT é uma fase de ajuste fino para quantização, não de aprendizado profundo.",
    tradeoffs:
      "↑ ajuste mais rápido, possível degradação de acurácia. ↓ ajuste mais conservador, modelo mais estável.",
    source: "modules/config/settings.py → LR_QAT.",
    essential: false,
  },
  patience_baseline: {
    what: "Número de épocas sem melhora em val_loss antes de acionar o early stopping nas fases baseline e fine-tune.",
    how: "10 é um bom padrão. Aumente para 15–20 se o modelo tender a platear e depois melhorar.",
    tradeoffs: "↑ treino mais longo aguardando melhora. ↓ para mais cedo, pode encerrar antes de convergir.",
    source: "modules/config/settings.py → PATIENCE_BASELINE.",
    essential: false,
  },
  use_qat: {
    what: "Ativa o Quantization-Aware Training após fine-tune, exportando um modelo int8 TFLite compacto para produção/edge.",
    how: "Mantenha ligado para uso em produção (mobile, FPGA). Desligue apenas para inspeção rápida do modelo FP32.",
    tradeoffs:
      "Ligado: modelo TFLite ~4x menor, inferência mais rápida em CPU/mobile, ~20 min a mais de treino. Desligado: só modelo FP32, treino mais rápido.",
    source: "tensorflow_model_optimization (tfmot).",
    essential: true,
  },
  augment: {
    what: "Ativa augmentação de dados leve no ImageDataGenerator (flips horizontal/vertical, pequenas rotações).",
    how: "Útil para datasets pequenos (<500 WSIs). Em histopatologia, flips geralmente são válidos pois tecido não tem orientação canônica.",
    tradeoffs:
      "Ligado: melhor generalização em datasets pequenos, treino mais lento. Desligado: mais rápido, necessário se augmentação violar invariâncias do domínio.",
    source: "modules/training/dataloader.py → ImageDataGenerator.",
    essential: true,
  },
  use_strategy: {
    what: "Ativa tf.distribute.MirroredStrategy para paralelismo multi-GPU. Distribui o batch entre todas as GPUs disponíveis.",
    how: "Mantenha ligado se houver mais de uma GPU. Desligue com `--no-strategy` se houver erros de CUDA ou em ambiente CPU.",
    tradeoffs:
      "Ligado: treino mais rápido com múltiplas GPUs, overhead de sincronização. Desligado: compatível com qualquer ambiente.",
    source: "modules/config/settings.py. Detecta GPUs via tf.config.list_physical_devices.",
    essential: false,
  },
  /* ── Split ──────────────────────────────────────────── */
  dataset_path: {
    what: "Diretório raiz contendo subpastas nomeadas por classe, cada uma com as imagens (tiles) daquela classe.",
    how: "Use o mesmo diretório gerado pelo Tiling (padrão: datas/). Estrutura esperada: datas/classe_A/img1.jpg, datas/classe_B/img2.jpg, ...",
    tradeoffs: "N/A — é o diretório de entrada.",
    source: "Saída da etapa de Tiling → processed_dataset_dir.",
    essential: true,
  },
  test_size: {
    what: "Fração dos dados reservada para validação + teste combinados (holdout). 0.3 = 30% holdout, 70% treino.",
    how: "0.3 é o padrão. Para datasets pequenos (<500 WSIs), considere 0.2 para ter mais dados de treino.",
    tradeoffs:
      "↑ avaliação mais robusta, menos dados de treino. ↓ mais dados de treino, estimativa de performance mais ruidosa.",
    source: "modules/split/create_split.py → StratifiedShuffleSplit.",
    essential: true,
  },
  val_size: {
    what: "Dentro do holdout, a fração destinada à validação (o restante vai para teste). val_size=0.5 divide o holdout em 50% val / 50% test.",
    how: "0.5 é o padrão. Aumente para ter mais validação (monitoramento de overfitting); diminua para teste mais robusto.",
    tradeoffs: "↑ mais dados para val, menos para test. ↓ test mais representativo, val menor.",
    source: "modules/split/create_split.py → segundo StratifiedShuffleSplit.",
    essential: true,
  },
  /* ── Tiling ─────────────────────────────────────────── */
  tile_size: {
    what: "Tamanho do tile extraído pelo DeepZoom em pixels, na resolução de magnificação alvo. O tamanho final após resize para o modelo pode diferir.",
    how: "1000 a 20x resulta em patches ricos em contexto. Ajuste considerando as estruturas de interesse: glândulas (~500–1000μm) vs. células (~10–20μm).",
    tradeoffs:
      "↑ mais contexto morfológico por patch, mais VRAM. ↓ mais detalhe celular, menos contexto, mais tiles por WSI.",
    source: "modules/wsi_pipeline/wsi_pipeline_config.py → tile_size.",
    essential: true,
  },
  target_magnification: {
    what: "Nível de magnificação alvo para extração dos tiles (ex: 20 = 20x). Deve coincidir com a resolução esperada pelo modelo.",
    how: "20x é o padrão em histopatologia. Use 10x para mais contexto (morfologia glandular), 40x para mais detalhe nuclear.",
    tradeoffs: "↑ mais detalhe celular, tiles menores em área, mais tiles por WSI. ↓ mais contexto, menos tiles.",
    source: "Nível OpenSlide mais próximo da magnificação alvo, com tolerância configurável.",
    essential: true,
  },
  max_white_background_fraction: {
    what: "Fração máxima da área do tile que pode ser fundo branco (tecido ausente). Tiles acima desse limiar são descartados.",
    how: "0.5 (50%) é o padrão. Para tecidos esparsos, aumente. Para tecidos densos, diminua para maior pureza.",
    tradeoffs:
      "↑ mais tiles preservados incluindo bordas esparsas. ↓ tiles mais ricos em tecido, dataset menor.",
    source: "modules/wsi_pipeline/pre_processing.py → filtro de background.",
    essential: true,
  },
  /* ── Labels ──────────────────────────────────────────── */
  manifest_csv: {
    what: "CSV gerado pela etapa de Download contendo os caminhos das WSIs e os IDs dos casos (case_submitter_id).",
    how: "Use o arquivo wsi_manifest.csv gerado automaticamente pelo Download TCGA. Deve ter ao menos uma coluna image_path.",
    tradeoffs: "N/A — é o arquivo de entrada.",
    source: "Saída da etapa de Download → wsi_manifest.csv na raiz do projeto.",
    essential: true,
  },
  sheets_ref: {
    what: "URL do Google Sheets ou caminho para CSV local contendo os rótulos dos pacientes (colunas Patient ID e Subtype).",
    how: "Google Sheets: compartilhe como 'qualquer pessoa com o link pode ver' e cole a URL. CSV local: informe o caminho relativo à raiz do projeto.",
    tradeoffs:
      "Google Sheets: atualização automática quando a planilha mudar. CSV local: funciona offline, mais controle de versão.",
    source: "Planilha interna do projeto ou exportação do portal clínico.",
    essential: true,
  },
  /* ── TCGA Download ───────────────────────────────────── */
  ids_csv: {
    what: "CSV com os IDs dos casos TCGA a baixar. Deve conter ao menos uma coluna com IDs no formato TCGA-XX-XXXX.",
    how: "Exporte do portal GDC (portal.gdc.cancer.gov) após filtrar por projeto/cohort, ou use lista curada interna.",
    tradeoffs: "N/A — é o arquivo de entrada.",
    source: "portal.gdc.cancer.gov → Cart → Download → Sample Sheet ou TSV de metadados.",
    essential: true,
  },
  only_open_access: {
    what: "Filtra apenas arquivos de acesso aberto no GDC (access=open). Dados controlados requerem token de autenticação.",
    how: "Mantenha ligado para dados públicos (maioria dos projetos TCGA). Desligue e forneça GDC_TOKEN para acesso a dados controlados.",
    tradeoffs:
      "Ligado: simples, sem token, cobre a maioria dos casos. Desligado: acesso a mais dados, exige token válido do portal GDC.",
    source: "portal.gdc.cancer.gov → Perfil → Download Token (válido por 30 dias).",
    essential: true,
  },
  download_concurrency: {
    what: "Número máximo de downloads paralelos de arquivos SVS. Aumentar acelera o download mas exige mais largura de banda e memória.",
    how: "4 é o padrão. Em conexões rápidas (>100 Mbps), aumente para 8. Em conexões lentas ou instáveis, reduza para 2.",
    tradeoffs: "↑ download mais rápido, mais uso de banda e CPU. ↓ download mais lento, mais estável.",
    source: "modules/tcga_dataset/download.py → asyncio gather.",
    essential: false,
  },
  /* ── Inference ───────────────────────────────────────── */
  threshold: {
    what: "Probabilidade mínima de classe positiva para classificar um patch como positivo no heatmap de reconstrução.",
    how: "0.9 é conservador (alta especificidade). Para triagem (alta sensibilidade), use 0.5–0.7. Calibre na curva ROC do teste.",
    tradeoffs:
      "↑ menos falsos positivos, possível perda de regiões verdadeiras. ↓ mais patches marcados, mais ruído no heatmap.",
    source: "Calibrar na curva ROC gerada durante a avaliação do modelo (test_eval).",
    essential: true,
  },
  patch_multiplier: {
    what: "Multiplicador do patch de entrada do modelo (224px × n). Define o tamanho das regiões inferidas na WSI.",
    how: "4 gera patches de 896px. Aumente para inferência mais rápida com menor resolução; diminua para mais detalhe.",
    tradeoffs: "↑ inferência mais rápida, resolução do heatmap menor. ↓ mais detalhe, inferência mais lenta.",
    source: "modules/inference/run_inference.py → patch_size = 224 * patch_multiplier.",
    essential: true,
  },
  tflite_path: {
    what: "Caminho para o modelo TFLite (int8) exportado na fase de QAT. É o artefato de produção gerado pelo Treino.",
    how: "Use models/qat_baseline_final.tflite gerado automaticamente. Para comparar versões, especifique o path manualmente.",
    tradeoffs: "N/A — é o modelo a usar.",
    source: "Saída da etapa de Treino → models/qat_baseline_final.tflite.",
    essential: true,
  },
};

export function getParamHelp(key: string): ParamHelp | undefined {
  return H[key];
}

export function isEssential(key: string): boolean {
  const h = H[key];
  return h?.essential !== false;
}
