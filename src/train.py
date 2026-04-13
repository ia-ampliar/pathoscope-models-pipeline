import argparse
from typing import Optional

import numpy as np
import tensorflow as tf

import modules.config as config
from modules.training import dataloader, modeling, train_utils, qat


def set_global_seed():
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)


def get_strategy():
    try:
        strategy = tf.distribute.MirroredStrategy()
        return strategy
    except Exception:
        return None


def run_training_pipeline(
    use_strategy: bool = True,
    use_qat: bool = True,
    augment: bool = False,
    metrics_stream_path: Optional[str] = None,
    qat_epochs: Optional[int] = None,
    overfitting_ratio_threshold: float = 1.6,
    target_val_loss: Optional[float] = None,
) -> None:
    set_global_seed()

    strategy = get_strategy() if use_strategy else None

    # 1) Dados
    train_df, val_df, test_df, _ = dataloader.load_splits()
    train_gen, val_gen, test_gen, num_classes = dataloader.make_generators(
        train_df,
        val_df,
        test_df,
        image_root=config.DATA_DIR,
        augment=augment,
        batch_size=config.BATCH_SIZE,
    )

    stream_kw = {}
    if metrics_stream_path:
        stream_kw = {
            "metrics_stream_path": metrics_stream_path,
            "overfitting_ratio_threshold": overfitting_ratio_threshold,
        }

    # 2) Baseline FP32
    if strategy is not None:
        with strategy.scope():
            model, base = modeling.build_baseline_model(
                num_classes=num_classes,
                base_trainable=False,
                learning_rate=config.LR_BASELINE,
            )
    else:
        model, base = modeling.build_baseline_model(
            num_classes=num_classes,
            base_trainable=False,
            learning_rate=config.LR_BASELINE,
        )

    baseline_checkpoint = str(config.MODELS_DIR / "baseline_checkpoint.keras")
    initial_epochs = int(config.EPOCHS_BASELINE * config.INITIAL_EPOCHS_FRACTION)

    history_baseline, baseline_final_path = train_utils.train_with_callbacks(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=initial_epochs,
        checkpoint_path=baseline_checkpoint,
        stage_name="baseline",
        early_stop_patience=config.PATIENCE_BASELINE,
        strategy=strategy,
        **stream_kw,
    )

    # 3) Fine-tuning (opcional, baseado no notebook original)
    modeling.enable_fine_tuning(base, config.FINE_TUNE_PERCENT)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_checkpoint = str(config.MODELS_DIR / "fine_tuned_checkpoint.keras")
    fine_epochs = config.EPOCHS_BASELINE - initial_epochs

    history_finetune, fine_final_path = train_utils.train_with_callbacks(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=fine_epochs,
        checkpoint_path=fine_checkpoint,
        stage_name="fine_tune",
        early_stop_patience=config.PATIENCE_BASELINE,
        strategy=strategy,
        **stream_kw,
    )

    # 4) Avaliação do melhor modelo FP32 no conjunto de teste
    best_fp32_path = fine_final_path
    best_fp32_model = tf.keras.models.load_model(best_fp32_path)
    test_loss, test_acc = best_fp32_model.evaluate(test_gen, verbose=1)
    print(f"FP32 - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    if metrics_stream_path:
        train_utils.append_metrics_jsonl(
            metrics_stream_path,
            {
                "type": "test_eval",
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
            },
        )
        val_losses = history_finetune.history.get("val_loss", [])
        if val_losses and target_val_loss is not None:
            best_val = min(float(x) for x in val_losses)
            train_utils.append_metrics_jsonl(
                metrics_stream_path,
                {
                    "type": "target_val_loss_check",
                    "best_val_loss": best_val,
                    "target_val_loss": float(target_val_loss),
                    "hit": best_val <= target_val_loss,
                },
            )

    # 5) QAT + TFLite
    qat_epochs_eff = int(qat_epochs if qat_epochs is not None else 15)
    tflite_path = None
    if use_qat:
        baseline_final = qat.load_baseline_for_qat(baseline_final_path)
        quantized_model = qat.apply_qat(baseline_final)
        qat_checkpoint = qat.train_qat(
            quantized_model,
            train_gen=train_gen,
            val_gen=val_gen,
            epochs=qat_epochs_eff,
            **stream_kw,
        )
        tflite_path = qat.export_qat_to_tflite(qat_checkpoint)
        print(f"QAT TFLite salvo em: {tflite_path}")

    if metrics_stream_path:
        payload = {
            "type": "completed",
            "best_fp32_model": str(best_fp32_path),
            "baseline_final": str(baseline_final_path),
        }
        if tflite_path is not None:
            payload["tflite"] = str(tflite_path)
        train_utils.append_metrics_jsonl(metrics_stream_path, payload)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento TensorFlow (baseline + fine-tune + QAT)"
    )
    parser.add_argument(
        "--no-strategy",
        action="store_true",
        help="Desabilita MirroredStrategy mesmo que haja múltiplas GPUs.",
    )
    parser.add_argument(
        "--no-qat",
        action="store_true",
        help="Desabilita etapa de Quantization Aware Training.",
    )
    args = parser.parse_args()

    run_training_pipeline(
        use_strategy=not args.no_strategy,
        use_qat=not args.no_qat,
    )


if __name__ == "__main__":
    main()
