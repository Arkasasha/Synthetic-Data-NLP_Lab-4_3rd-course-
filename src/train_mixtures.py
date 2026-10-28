"""Train five balanced real/synthetic AG News mixtures on CUDA.

Each 8,000-row mixture is trained for one epoch with the same DistilBERT
architecture, initialization seed, optimizer, and hyperparameters. Validation
is used during training; the original test set is evaluated once after each
model finishes training.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "ag_news"
SYNTHETIC_CSV = ROOT / "data" / "synthetic" / "final_8000.csv"
MODEL_ROOT = ROOT / "models" / "mixtures"
RESULTS_DIR = ROOT / "results" / "mixture_training"
MODEL_NAME = "distilbert/distilbert-base-uncased"
PERCENTAGES = (0, 25, 50, 75, 100)
SEED = 42
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
EPOCHS = 1
MAX_LENGTH = 128


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def score(actual, predicted, labels: list[str]) -> dict:
    classes = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=classes, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
    }


def compute_metrics(prediction) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1)
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    metrics = score(prediction.label_ids, predicted, labels)
    return {key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}


def make_orders(real: Dataset, synthetic: Dataset) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    orders = []
    for dataset in (real, synthetic):
        by_label = {}
        labels = dataset["label"]
        for class_id in range(4):
            indices = [index for index, label in enumerate(labels) if label == class_id]
            if len(indices) != 2000:
                raise ValueError(f"Expected 2,000 rows for class {class_id}; got {len(indices)}")
            random.Random(SEED + class_id).shuffle(indices)
            by_label[class_id] = indices
        orders.append(by_label)
    return orders[0], orders[1]


def make_mixture(
    real: Dataset,
    synthetic: Dataset,
    real_order: dict[int, list[int]],
    synthetic_order: dict[int, list[int]],
    percentage: int,
) -> tuple[Dataset, dict]:
    synthetic_per_class = 2000 * percentage // 100
    real_per_class = 2000 - synthetic_per_class
    real_indices = [
        index for class_id in range(4) for index in real_order[class_id][:real_per_class]
    ]
    synthetic_indices = [
        index for class_id in range(4) for index in synthetic_order[class_id][:synthetic_per_class]
    ]
    mixture = concatenate_datasets(
        [real.select(real_indices), synthetic.select(synthetic_indices)]
    ).shuffle(seed=SEED)
    if len(mixture) != 8000 or any(mixture["label"].count(i) != 2000 for i in range(4)):
        raise ValueError(f"Unbalanced mixture at {percentage}% synthetic")
    manifest = {
        "synthetic_percentage": percentage,
        "real_count": len(real_indices),
        "synthetic_count": len(synthetic_indices),
        "class_counts": {str(i): 2000 for i in range(4)},
        "real_train_indices": real_indices,
        "synthetic_sample_ids": [synthetic[index]["sample_id"] for index in synthetic_indices],
        "seed": SEED,
    }
    return mixture.remove_columns("sample_id"), manifest


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for mixture training")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    splits = load_from_disk(str(DATA_DIR))
    if {name: len(split) for name, split in splits.items()} != {
        "train": 8000, "validation": 2000, "test": 7600
    }:
        raise ValueError("Unexpected prepared AG News split sizes")
    real = splits["train"]
    label_names = real.features["label"].names
    if label_names != ["World", "Sports", "Business", "Sci/Tech"]:
        raise ValueError(f"Unexpected label mapping: {label_names}")
    source_sha256 = hashlib.sha256(SYNTHETIC_CSV.read_bytes()).hexdigest()
    frame = pd.read_csv(SYNTHETIC_CSV, dtype={"sample_id": str})
    required = {"sample_id", "text", "label"}
    if not required.issubset(frame.columns) or len(frame) != 8000:
        raise ValueError("Expected 8,000 synthetic rows with sample_id, text, label")
    if frame[list(required)].isna().any().any() or frame["text"].str.strip().eq("").any():
        raise ValueError("Missing or blank synthetic values")
    if not frame["sample_id"].is_unique or not frame["label"].isin(range(4)).all():
        raise ValueError("Invalid synthetic IDs or labels")
    synthetic_text = set(frame["text"].map(normalize))
    if len(synthetic_text) != len(frame):
        raise ValueError("Duplicate synthetic texts")
    for split_name, split in splits.items():
        if synthetic_text & {normalize(text) for text in split["text"]}:
            raise ValueError(f"Synthetic text overlaps {split_name} split")
    features = Features({
        "text": Value("string"),
        "label": real.features["label"],
        "sample_id": Value("string"),
    })
    synthetic = Dataset.from_dict(
        {
            "text": frame["text"].astype(str).tolist(),
            "label": frame["label"].astype(int).tolist(),
            "sample_id": frame["sample_id"].astype(str).tolist(),
        },
        features=features,
    )
    real = real.add_column("sample_id", [f"real_{i}" for i in range(len(real))])
    real_order, synthetic_order = make_orders(real, synthetic)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    validation = splits["validation"].map(tokenize, batched=True, remove_columns=["text"])
    test = splits["test"].map(tokenize, batched=True, remove_columns=["text"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for percentage in PERCENTAGES:
        name = f"synthetic_{percentage:03d}"
        model_dir = MODEL_ROOT / name
        result_dir = RESULTS_DIR / name
        result_file = result_dir / "metrics.json"
        if result_file.exists() and (model_dir / "model.safetensors").exists():
            saved = json.loads(result_file.read_text(encoding="utf-8"))
            if saved["synthetic_source_sha256"] != source_sha256:
                raise ValueError(f"Synthetic source changed since {name} was trained")
            print("Reusing completed mixture:", percentage, "%", flush=True)
            summary_rows.append(saved["summary_row"])
            continue
        mixture, manifest = make_mixture(
            real, synthetic, real_order, synthetic_order, percentage
        )
        manifest["synthetic_source_sha256"] = source_sha256
        write_json(result_dir / "mixture_manifest.json", manifest)
        print(
            f"Training {percentage}% synthetic: {manifest['real_count']} real + "
            f"{manifest['synthetic_count']} synthetic", flush=True
        )
        train_data = mixture.map(
            tokenize, batched=True, remove_columns=["text"]
        )
        set_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=4,
            id2label={i: label for i, label in enumerate(label_names)},
            label2id={label: i for i, label in enumerate(label_names)},
            local_files_only=True,
        )
        args = TrainingArguments(
            output_dir=str(ROOT / "checkpoints" / "mixtures" / name),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="no",
            seed=SEED,
            data_seed=SEED,
            bf16=True,
            report_to="none",
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=validation,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        if trainer.args.device.type != "cuda":
            raise RuntimeError("Trainer did not select CUDA")
        train_result = trainer.train()
        validation_events = [
            row for row in trainer.state.log_history if "eval_macro_f1" in row
        ]
        if len(validation_events) != 5 or validation_events[-1]["step"] != 500:
            raise RuntimeError("Expected five validation checks through step 500")
        final_validation = validation_events[-1]
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        config = {
            "model": MODEL_NAME,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
            "synthetic_percentage": percentage,
            "real_count": manifest["real_count"],
            "synthetic_count": manifest["synthetic_count"],
            "train_samples": len(train_data),
            "validation_samples": len(validation),
            "test_samples": len(test),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": 16,
            "eval_batch_size": 32,
            "max_length": MAX_LENGTH,
            "precision": "bf16",
            "synthetic_source_sha256": source_sha256,
        }
        write_json(model_dir / "training_config.json", config)
        write_json(result_dir / "training_history.json", trainer.state.log_history)
        print("Evaluating original test set for", percentage, "%", flush=True)
        test_prediction = trainer.predict(test)
        predicted = np.argmax(test_prediction.predictions, axis=-1)
        test_scores = score(test_prediction.label_ids, predicted, label_names)
        row = {
            "synthetic_percentage": percentage,
            "real_count": manifest["real_count"],
            "synthetic_count": manifest["synthetic_count"],
            "train_loss": float(train_result.training_loss),
            "validation_loss": float(final_validation["eval_loss"]),
            "validation_accuracy": float(final_validation["eval_accuracy"]),
            "validation_macro_precision": float(final_validation["eval_macro_precision"]),
            "validation_macro_recall": float(final_validation["eval_macro_recall"]),
            "validation_macro_f1": float(final_validation["eval_macro_f1"]),
            "test_loss": float(test_prediction.metrics["test_loss"]),
            "test_accuracy": test_scores["accuracy"],
            "test_macro_precision": test_scores["macro_precision"],
            "test_macro_recall": test_scores["macro_recall"],
            "test_macro_f1": test_scores["macro_f1"],
        }
        write_json(result_file, {
            "summary_row": row,
            "test_per_class": test_scores["per_class"],
            "test_confusion_matrix": test_scores["confusion_matrix"],
            "label_names": label_names,
            "validation_history": validation_events,
            "model_dir": str(model_dir.relative_to(ROOT)).replace("\\", "/"),
            "synthetic_source_sha256": source_sha256,
        })
        summary_rows.append(row)
        print(
            f"Completed {percentage}%: test accuracy={row['test_accuracy']:.4f}, "
            f"macro F1={row['test_macro_f1']:.4f}", flush=True
        )
        del trainer, model
        torch.cuda.empty_cache()

    table = pd.DataFrame(summary_rows).sort_values("synthetic_percentage")
    if table["synthetic_percentage"].tolist() != list(PERCENTAGES):
        raise RuntimeError("Missing mixture results")
    table.to_csv(RESULTS_DIR / "mixture_metrics.csv", index=False)
    write_json(RESULTS_DIR / "summary.json", {
        "model": MODEL_NAME,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "train_size": 8000,
        "synthetic_source": str(SYNTHETIC_CSV.relative_to(ROOT)).replace("\\", "/"),
        "synthetic_source_sha256": source_sha256,
        "results": summary_rows,
    })
    print("\nFinal results table:", flush=True)
    print(table.to_string(index=False), flush=True)
    print("Saved:", RESULTS_DIR / "mixture_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
