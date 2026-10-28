"""Five-epoch, nine-variant DistilBERT grid search on AG News validation data.

Run from the project root with the CUDA-enabled project environment:
    .venv/Scripts/python.exe src/tune_baseline.py

Each variant starts from the same pretrained weights and seed. The original
test split is never used for model selection.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "ag_news"
MODEL_ROOT = ROOT / "models" / "distilbert_grid"
RESULTS_DIR = ROOT / "results" / "hyperparameter_tuning"
MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
EPOCHS = 5
LEARNING_RATES = (2e-5, 3e-5, 5e-5)
WEIGHT_DECAYS = (0.0, 0.01, 0.1)
METRICS = (
    "train_loss",
    "eval_loss",
    "eval_accuracy",
    "eval_macro_precision",
    "eval_macro_recall",
    "eval_macro_f1",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_metrics(prediction) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1)
    actual = prediction.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


class SaveBestValidationModel(TrainerCallback):
    """Persist the best state for one variant after each validation epoch."""

    def __init__(self, model, tokenizer, model_dir: Path) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.best_f1 = -math.inf
        self.best_loss = math.inf
        self.best_epoch = 0
        self.history: list[dict[str, float | int]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_macro_f1" not in metrics:
            return control
        epoch = int(round(state.epoch))
        row = {
            "epoch": epoch,
            **{
                key: float(metrics[key])
                for key in METRICS[1:]
                if key in metrics
            },
        }
        self.history.append(row)
        f1 = row["eval_macro_f1"]
        loss = row["eval_loss"]
        if f1 > self.best_f1 or (f1 == self.best_f1 and loss < self.best_loss):
            self.best_f1 = f1
            self.best_loss = loss
            self.best_epoch = epoch
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(self.model_dir), safe_serialization=True)
            self.tokenizer.save_pretrained(str(self.model_dir))
            print(
                f"  saved best epoch {epoch}: macro F1={f1:.4f}, "
                f"validation loss={loss:.4f}",
                flush=True,
            )
        return control


def history_by_epoch(log_history, eval_history):
    rows = {int(row["epoch"]): dict(row) for row in eval_history}
    for event in log_history:
        if "loss" in event and "eval_loss" not in event:
            epoch = int(round(event["epoch"]))
            if epoch in rows:
                rows[epoch]["train_loss"] = float(event["loss"])
    result = [rows[epoch] for epoch in sorted(rows)]
    if len(result) != EPOCHS or any("train_loss" not in row for row in result):
        raise RuntimeError(f"Incomplete training or validation history: {result}")
    return result


def variant_name(learning_rate: float, weight_decay: float) -> str:
    return f"lr_{learning_rate:.0e}_wd_{weight_decay:g}"


def plot_grid(variants: list[dict]) -> None:
    fig, axes = plt.subplots(
        len(variants), len(METRICS), figsize=(27, 27), sharex=True, sharey="col",
        constrained_layout=True,
    )
    titles = {
        "train_loss": "Training loss",
        "eval_loss": "Validation loss",
        "eval_accuracy": "Validation accuracy",
        "eval_macro_precision": "Validation macro precision",
        "eval_macro_recall": "Validation macro recall",
        "eval_macro_f1": "Validation macro F1",
    }
    for row_index, variant in enumerate(variants):
        history = variant["history"]
        epochs = [row["epoch"] for row in history]
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            ax.plot(epochs, [row[metric] for row in history], marker="o", linewidth=1.7)
            ax.axvline(variant["best_epoch"], color="tab:orange", alpha=0.45, linestyle="--")
            ax.grid(alpha=0.25)
            ax.set_xticks(range(1, EPOCHS + 1))
            if column_index >= 2:
                values = [
                    row[metric]
                    for item in variants
                    for row in item["history"]
                ]
                ax.set_ylim(max(0, min(values) - 0.015), min(1, max(values) + 0.015))
            if row_index == 0:
                ax.set_title(titles[metric])
            if column_index == 0:
                ax.set_ylabel(
                    f"LR={variant['learning_rate']:g}\nWD={variant['weight_decay']:g}"
                )
            if row_index == len(variants) - 1:
                ax.set_xlabel("Epoch")
    fig.suptitle(
        "DistilBERT AG News grid search: 5 epochs per variant\n"
        "Dashed line = saved best epoch (highest validation macro F1)",
        fontsize=16,
    )
    fig.savefig(RESULTS_DIR / "hyperparameter_grid.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this 9-run grid search")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    splits = load_from_disk(str(DATA_DIR))
    if len(splits["train"]) != 8000 or len(splits["validation"]) != 2000:
        raise ValueError("Expected 8,000 training and 2,000 validation samples")
    label_names = splits["train"].features["label"].names
    if len(label_names) != 4:
        raise ValueError(f"Expected four AG News labels, got {label_names}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_train = splits["train"].map(
        tokenize, batched=True, remove_columns=["text"]
    )
    tokenized_validation = splits["validation"].map(
        tokenize, batched=True, remove_columns=["text"]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variants = []
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            name = variant_name(learning_rate, weight_decay)
            result_path = RESULTS_DIR / f"{name}.json"
            model_dir = MODEL_ROOT / name
            if result_path.exists() and (model_dir / "model.safetensors").exists():
                print("Reusing completed variant:", name, flush=True)
                variants.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            print(f"\nTraining {name} for {EPOCHS} epochs", flush=True)
            set_seed(SEED)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=4,
                id2label={index: label for index, label in enumerate(label_names)},
                label2id={label: index for index, label in enumerate(label_names)},
                local_files_only=True,
            )
            callback = SaveBestValidationModel(model, tokenizer, model_dir)
            args = TrainingArguments(
                output_dir=str(ROOT / "checkpoints" / "hyperparameter_tuning" / name),
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                optim="adamw_torch",
                eval_strategy="epoch",
                save_strategy="no",
                logging_strategy="epoch",
                seed=SEED,
                data_seed=SEED,
                bf16=True,
                report_to="none",
                disable_tqdm=True,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_validation,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
                callbacks=[callback],
            )
            trainer.train()
            history = history_by_epoch(trainer.state.log_history, callback.history)
            variant = {
                "variant": name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": EPOCHS,
                "seed": SEED,
                "train_samples": 8000,
                "validation_samples": 2000,
                "model_name": MODEL_NAME,
                "model_dir": str(model_dir.relative_to(ROOT)).replace("\\", "/"),
                "best_epoch": callback.best_epoch,
                "best_validation_macro_f1": callback.best_f1,
                "best_validation_loss": callback.best_loss,
                "history": history,
            }
            write_json(result_path, variant)
            write_json(model_dir / "training_config.json", {key: value for key, value in variant.items() if key != "history"})
            variants.append(variant)
            print(
                f"Completed {name}: best epoch {callback.best_epoch}, "
                f"macro F1={callback.best_f1:.4f}",
                flush=True,
            )
            del trainer, model
            torch.cuda.empty_cache()

    best = max(
        variants,
        key=lambda variant: (
            variant["best_validation_macro_f1"],
            -variant["best_validation_loss"],
        ),
    )
    summary = {
        "selection_metric": "validation macro F1 (validation loss breaks ties)",
        "test_set_used": False,
        "gpu": torch.cuda.get_device_name(0),
        "variants": variants,
        "best_variant": best["variant"],
        "best_model_dir": best["model_dir"],
        "best_validation_macro_f1": best["best_validation_macro_f1"],
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    with (RESULTS_DIR / "best_variants.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "variant", "learning_rate", "weight_decay", "best_epoch",
                "best_validation_macro_f1", "best_validation_loss", "model_dir",
            ],
        )
        writer.writeheader()
        for variant in sorted(
            variants, key=lambda item: item["best_validation_macro_f1"], reverse=True
        ):
            writer.writerow({key: variant[key] for key in writer.fieldnames})
    with (RESULTS_DIR / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["variant", "learning_rate", "weight_decay", "best_epoch", "epoch", *METRICS]
        )
        writer.writeheader()
        for variant in variants:
            for row in variant["history"]:
                writer.writerow({
                    "variant": variant["variant"],
                    "learning_rate": variant["learning_rate"],
                    "weight_decay": variant["weight_decay"],
                    "best_epoch": variant["best_epoch"],
                    **row,
                })
    plot_grid(variants)
    print("\nBest variant:", best["variant"], flush=True)
    print("Best validation macro F1:", best["best_validation_macro_f1"], flush=True)
    print("Saved results to:", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
