export type ApiSchemaPayload = {
  version: number;
  groups: { id: string; title: string; description: string }[];
  schemas: Record<string, Record<string, unknown>>;
  defaults: Record<string, Record<string, unknown>>;
};

export type StreamEvent =
  | { type: "stage_start"; stage: string }
  | {
      type: "epoch";
      stage: string;
      epoch: number;
      metrics: Record<string, number>;
      warning?: { type: string; severity: string; ratio?: number };
    }
  | { type: "test_eval"; test_loss: number; test_accuracy: number }
  | {
      type: "target_val_loss_check";
      best_val_loss: number;
      target_val_loss: number;
      hit: boolean;
    }
  | { type: "completed"; best_fp32_model?: string; baseline_final?: string; tflite?: string }
  | { type: "error"; message: string }
  | { type: "stream_end"; status: string };
