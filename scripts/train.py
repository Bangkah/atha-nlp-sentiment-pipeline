from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import evaluate
from huggingface_hub import HfApi
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "indobenchmark/indobert-base-p1"
    train_file: str = "data/raw/train.csv"
    valid_file: str = "data/raw/valid.csv"
    output_dir: str = "artifacts/model"
    max_length: int = 128
    num_labels: int = 3
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8


LABEL_NAMES = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="", help="Override base model")
    parser.add_argument("--train-file", default="", help="Path CSV untuk split train")
    parser.add_argument("--valid-file", default="", help="Path CSV untuk split validation")
    parser.add_argument("--epochs", type=int, default=0, help="Override jumlah epoch")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model ke Hugging Face Hub setelah training",
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="Format: username/nama-repo (wajib jika --push-to-hub)",
    )
    parser.add_argument(
        "--private-repo",
        action="store_true",
        help="Saat push, repo di Hub dibuat private",
    )
    return parser.parse_args()


def resolve_config(cli_args: argparse.Namespace, base_cfg: TrainConfig) -> TrainConfig:
    return TrainConfig(
        model_name=cli_args.model_name or base_cfg.model_name,
        train_file=cli_args.train_file or base_cfg.train_file,
        valid_file=cli_args.valid_file or base_cfg.valid_file,
        output_dir=base_cfg.output_dir,
        max_length=base_cfg.max_length,
        num_labels=base_cfg.num_labels,
        epochs=cli_args.epochs if cli_args.epochs > 0 else base_cfg.epochs,
        train_batch_size=base_cfg.train_batch_size,
        eval_batch_size=base_cfg.eval_batch_size,
    )


def build_model_card(
    metrics: dict[str, float],
    matrix: np.ndarray,
    report_text: str,
    base_model_name: str,
) -> str:
    matrix_rows = ["| true\\pred | negative | neutral | positive |", "|---|---:|---:|---:|"]
    for idx, label in enumerate(LABEL_NAMES):
        row = matrix[idx]
        matrix_rows.append(f"| {label} | {int(row[0])} | {int(row[1])} | {int(row[2])} |")

    return f"""---
library_name: transformers
license: apache-2.0
base_model: {base_model_name}
language:
    - id
datasets:
    - Bangkah/atha-text-dataset
tags:
    - sentiment-analysis
    - text-classification
    - indonesian
metrics:
    - accuracy
    - f1
---

# atha-text-classifier

Model ini adalah fine-tuned {base_model_name} untuk klasifikasi sentimen Bahasa Indonesia 3 kelas.

Training data: https://huggingface.co/datasets/Bangkah/atha-text-dataset

## Validation Metrics

- Loss: {metrics['eval_loss']:.4f}
- Accuracy: {metrics['eval_accuracy']:.4f}
- Macro F1: {metrics['eval_f1']:.4f}

## Confusion Matrix

{chr(10).join(matrix_rows)}

## Classification Report

```text
{report_text}
```
"""


def build_error_analysis(
    *,
    texts: list[str],
    labels: np.ndarray,
    logits: np.ndarray,
    output_dir: str,
) -> tuple[Path, Path]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    pred_ids = np.argmax(logits, axis=1)
    confidences = probs[np.arange(len(pred_ids)), pred_ids]

    rows: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        true_id = int(labels[i])
        pred_id = int(pred_ids[i])
        rows.append(
            {
                "text": text,
                "true_label": LABEL_NAMES[true_id],
                "pred_label": LABEL_NAMES[pred_id],
                "confidence": round(float(confidences[i]), 6),
                "is_error": true_id != pred_id,
            }
        )

    analysis_df = pd.DataFrame(rows)
    errors_df = analysis_df[analysis_df["is_error"]].sort_values(
        by="confidence",
        ascending=False,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "error_analysis.csv"
    errors_df.to_csv(csv_path, index=False)

    total = len(analysis_df)
    err_count = len(errors_df)
    error_rate = (err_count / total) if total else 0.0
    top_preview = errors_df.head(10)
    preview_lines = []
    for _, row in top_preview.iterrows():
        preview_lines.append(
            f"- [{row['true_label']} -> {row['pred_label']}] conf={row['confidence']}: {row['text']}"
        )

    md_path = out_dir / "error_analysis.md"
    md_path.write_text(
        "\n".join(
            [
                "# Error Analysis",
                "",
                f"- Total validation samples: {total}",
                f"- Misclassified samples: {err_count}",
                f"- Error rate: {error_rate:.4f}",
                "",
                "## Top Misclassifications (highest confidence)",
                *(preview_lines if preview_lines else ["- Tidak ada error pada validation set."]),
                "",
                f"CSV detail: {csv_path.name}",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def validate_dataset_schema(dataset_dict: Any) -> None:
    required_columns = {"text", "label"}
    allowed_labels = {0, 1, 2}

    for split_name in ("train", "validation"):
        split = dataset_dict[split_name]
        actual_columns = set(split.column_names)
        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise ValueError(
                f"CSV split '{split_name}' wajib memiliki kolom: text,label. "
                f"Kolom yang hilang: {sorted(missing_columns)}"
            )

        labels = set(int(x) for x in split["label"])
        invalid = labels - allowed_labels
        if invalid:
            raise ValueError(
                f"CSV split '{split_name}' memiliki label tidak valid: {sorted(invalid)}. "
                "Label valid adalah 0=negative, 1=neutral, 2=positive"
            )


def main() -> None:
    cli_args = parse_args()
    cfg = resolve_config(cli_args, TrainConfig())

    if cli_args.push_to_hub and not cli_args.repo_id:
        raise ValueError("repo-id wajib diisi saat memakai --push-to-hub")

    if not Path(cfg.train_file).exists() or not Path(cfg.valid_file).exists():
        raise FileNotFoundError(
            "Dataset belum ada. Jalankan dulu: python scripts/create_dataset.py"
        )

    dataset = load_dataset(
        "csv",
        data_files={"train": cfg.train_file, "validation": cfg.valid_file},
    )
    validate_dataset_schema(dataset)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    tokenized = dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1_score = f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        return {"accuracy": float(accuracy_score), "f1": float(f1_score)}

    args_kwargs: dict[str, object] = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.train_batch_size,
        "per_device_eval_batch_size": cfg.eval_batch_size,
        "save_strategy": "epoch",
        "logging_steps": 10,
        "load_best_model_at_end": False,
        "report_to": "none",
        "push_to_hub": cli_args.push_to_hub,
        "hub_model_id": cli_args.repo_id if cli_args.push_to_hub else None,
        "hub_private_repo": cli_args.private_repo,
    }

    training_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_params:
        args_kwargs["eval_strategy"] = "epoch"
    else:
        args_kwargs["evaluation_strategy"] = "epoch"

    args = TrainingArguments(**args_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    predictions_output = trainer.predict(tokenized["validation"])
    predicted_ids = np.argmax(predictions_output.predictions, axis=-1)
    true_ids = predictions_output.label_ids
    matrix = confusion_matrix(true_ids, predicted_ids, labels=[0, 1, 2])
    report = classification_report(
        true_ids,
        predicted_ids,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    model_card_text = build_model_card(metrics, matrix, report, cfg.model_name)
    model_card_path = Path(cfg.output_dir) / "README.md"
    model_card_path.write_text(model_card_text, encoding="utf-8")
    custom_model_card_path = Path(cfg.output_dir) / "README.custom.md"
    custom_model_card_path.write_text(model_card_text, encoding="utf-8")

    metrics_path = Path(cfg.output_dir) / "metrics_summary.txt"
    metrics_path.write_text(
        f"accuracy={metrics['eval_accuracy']:.4f}\n"
        f"macro_f1={metrics['eval_f1']:.4f}\n"
        f"loss={metrics['eval_loss']:.4f}\n\n"
        "confusion_matrix:\n"
        f"{matrix}\n\n"
        "classification_report:\n"
        f"{report}\n",
        encoding="utf-8",
    )

    print(f"Model saved to: {cfg.output_dir}")

    csv_error_path, md_error_path = build_error_analysis(
        texts=dataset["validation"]["text"],
        labels=true_ids,
        logits=predictions_output.predictions,
        output_dir=cfg.output_dir,
    )
    print(f"Error analysis saved: {csv_error_path} and {md_error_path}")

    if cli_args.push_to_hub:
        trainer.push_to_hub()
        tokenizer.push_to_hub(cli_args.repo_id)
        HfApi().upload_file(
            path_or_fileobj=str(custom_model_card_path),
            path_in_repo="README.md",
            repo_id=cli_args.repo_id,
            repo_type="model",
        )
        HfApi().upload_file(
            path_or_fileobj=str(md_error_path),
            path_in_repo="error_analysis.md",
            repo_id=cli_args.repo_id,
            repo_type="model",
        )
        print(f"Model pushed to Hub: {cli_args.repo_id}")


if __name__ == "__main__":
    main()
