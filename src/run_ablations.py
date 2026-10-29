"""Run reproducible 50% synthetic DistilBERT ablations on CUDA."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mixtures import MODEL_NAME, MAX_LENGTH, compute_metrics, normalize, score, write_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ablations"
MODELS = ROOT / "models" / "ablations"
SOURCES = {"filtered": ROOT / "data/synthetic/full_filtered.csv",
           "unfiltered": ROOT / "data/synthetic/full_unfiltered.csv"}
SEEDS = (42, 123, 2026)
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def select_rows(frame: pd.DataFrame, strategy: str, count: int, seed: int) -> pd.DataFrame:
    chunks = []
    for label in range(4):
        pool = frame[(frame.label == label) & (frame.strategy == strategy)]
        if len(pool) < count:
            raise ValueError(f"Only {len(pool)} {strategy} rows for class {label}; need {count}")
        chunks.append(pool.sample(n=count, random_state=seed + label).copy())
    return pd.concat(chunks, ignore_index=True)


def make_training(real: Dataset, frame: pd.DataFrame, seed: int, basic_enriched: bool) -> tuple[Dataset, dict]:
    # The filtered source contains only 109 basic Business examples. Use 100
    # per class so all four experimental cells have exactly the same design.
    basic_per_class = 100 if basic_enriched else 0
    advanced_per_class = 1000 - basic_per_class
    parts = [select_rows(frame, "advanced", advanced_per_class, seed)]
    if basic_per_class:
        parts.append(select_rows(frame, "basic", basic_per_class, seed))
    synthetic = pd.concat(parts, ignore_index=True)
    real_indices = []
    labels = np.asarray(real["label"])
    for label in range(4):
        indices = np.flatnonzero(labels == label)
        real_indices.extend(np.random.default_rng(seed + label).permutation(indices)[:1000].tolist())
    real_frame = pd.DataFrame(real.select(real_indices))[["text", "label"]]
    real_frame["origin"] = "real"
    real_frame["sample_id"] = [f"real_{i}" for i in real_indices]
    synthetic_frame = synthetic[["text", "label", "sample_id"]].copy()
    synthetic_frame["origin"] = "synthetic"
    combined = pd.concat([real_frame, synthetic_frame], ignore_index=True)
    if len(combined) != 8000 or combined.groupby("label").size().to_dict() != dict.fromkeys(range(4), 2000):
        raise ValueError("Training set is not balanced")
    if combined.text.map(normalize).duplicated().any():
        raise ValueError("Duplicate text in training mixture")
    manifest = {"seed": seed, "real_count": 4000, "synthetic_count": 4000,
                "basic_count": 4 * basic_per_class,
                "advanced_count": 4 * advanced_per_class,
                "real_train_indices": real_indices,
                "synthetic_sample_ids": synthetic.sample_id.astype(str).tolist(),
                "class_counts": dict.fromkeys(map(str, range(4)), 2000)}
    return Dataset.from_pandas(combined[["text", "label"]], preserve_index=False).shuffle(seed=seed), manifest


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    splits = load_from_disk(str(ROOT / "data/processed/ag_news"))
    if {k: len(v) for k, v in splits.items()} != {"train": 8000, "validation": 2000, "test": 7600}:
        raise ValueError("Unexpected real split sizes")
    sources = {name: pd.read_csv(path, dtype={"sample_id": str}) for name, path in SOURCES.items()}
    real_text = {normalize(text) for split in splits.values() for text in split["text"]}
    for name, frame in sources.items():
        if not {"text", "label", "strategy", "sample_id"}.issubset(frame.columns):
            raise ValueError(f"Missing columns in {name}")
        if frame[["text", "label", "strategy", "sample_id"]].isna().any().any():
            raise ValueError(f"Missing values in {name}")
        texts = frame.text.map(normalize)
        if texts.duplicated().any() or set(texts) & real_text:
            raise ValueError(f"Duplicate or real-overlapping synthetic text in {name}")
        if not frame.sample_id.is_unique or not frame.label.isin(range(4)).all():
            raise ValueError(f"Invalid IDs or labels in {name}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    validation = splits["validation"].map(tokenize, batched=True, remove_columns=["text"])
    test = splits["test"].map(tokenize, batched=True, remove_columns=["text"])
    rows = []
    for filtered in ("unfiltered", "filtered"):
        frame = sources[filtered]
        source_hash = hashlib.sha256(SOURCES[filtered].read_bytes()).hexdigest()
        for prompt in ("advanced_only", "basic_enriched"):
            for seed in SEEDS:
                run = f"{filtered}_{prompt}_seed_{seed}"
                result_dir, model_dir = RESULTS / run, MODELS / run
                metrics_file = result_dir / "metrics.json"
                mixture, manifest = make_training(splits["train"], frame, seed, prompt == "basic_enriched")
                manifest.update({"filtering": filtered, "prompt_mix": prompt,
                                 "source": str(SOURCES[filtered].relative_to(ROOT)).replace("\\", "/"),
                                 "source_sha256": source_hash})
                if metrics_file.exists() and (model_dir / "model.safetensors").exists():
                    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
                    if saved["manifest"] != manifest:
                        raise ValueError(f"Existing run {run} has a different input manifest")
                    rows.append(saved["summary_row"])
                    print("Reusing", run, flush=True)
                    continue
                write_json(result_dir / "manifest.json", manifest)
                train = mixture.map(tokenize, batched=True, remove_columns=["text"])
                set_seed(seed)
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=4,
                    id2label=dict(enumerate(LABELS)), label2id={v: k for k, v in enumerate(LABELS)},
                    local_files_only=True)
                args = TrainingArguments(
                    output_dir=str(ROOT / "checkpoints/ablations" / run),
                    num_train_epochs=1, per_device_train_batch_size=16,
                    per_device_eval_batch_size=32, learning_rate=5e-5,
                    weight_decay=0.01, lr_scheduler_type="linear", optim="adamw_torch",
                    eval_strategy="steps", eval_steps=100, logging_strategy="steps",
                    logging_steps=100, save_strategy="no", seed=seed, data_seed=seed,
                    bf16=True, report_to="none", disable_tqdm=True)
                trainer = Trainer(model=model, args=args, train_dataset=train,
                                  eval_dataset=validation, processing_class=tokenizer,
                                  data_collator=DataCollatorWithPadding(tokenizer),
                                  compute_metrics=compute_metrics)
                if trainer.args.device.type != "cuda":
                    raise RuntimeError("Trainer did not select CUDA")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                print("Training", run, flush=True)
                training = trainer.train()
                peak_gpu_gb = torch.cuda.max_memory_reserved() / 1024**3
                val_events = [entry for entry in trainer.state.log_history if "eval_macro_f1" in entry]
                if len(val_events) != 5:
                    raise RuntimeError("Expected 5 validation checks")
                trainer.save_model(str(model_dir))
                tokenizer.save_pretrained(str(model_dir))
                write_json(model_dir / "training_config.json", {
                    "model": MODEL_NAME, "seed": seed, "epochs": 1, "learning_rate": 5e-5,
                    "weight_decay": 0.01, "batch_size": 16, "max_length": MAX_LENGTH,
                    "precision": "bf16", "source_sha256": source_hash,
                    "gpu": torch.cuda.get_device_name(0)})
                write_json(result_dir / "training_history.json", trainer.state.log_history)
                prediction = trainer.predict(test)
                metrics = score(prediction.label_ids, np.argmax(prediction.predictions, axis=-1), LABELS)
                final_val = val_events[-1]
                row = {"run": run, "filtering": filtered, "prompt_mix": prompt, "seed": seed,
                       "real_count": 4000, "synthetic_count": 4000,
                       "basic_count": manifest["basic_count"],
                       "advanced_count": manifest["advanced_count"],
                       "train_runtime_sec": float(training.metrics["train_runtime"]),
                       "peak_gpu_reserved_gb": peak_gpu_gb,
                       "validation_accuracy": float(final_val["eval_accuracy"]),
                       "validation_macro_f1": float(final_val["eval_macro_f1"]),
                       "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                       **{f"test_recall_{LABELS[i].lower().replace('/', '_')}": metrics["per_class"][LABELS[i]]["recall"] for i in range(4)}}
                write_json(metrics_file, {"summary_row": row, "manifest": manifest,
                                          "test_per_class": metrics["per_class"],
                                          "test_confusion_matrix": metrics["confusion_matrix"],
                                          "validation_history": val_events})
                rows.append(row)
                print(run, "test macro F1", round(row["test_macro_f1"], 4), flush=True)
                del trainer, model
                torch.cuda.empty_cache()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "runs.csv", index=False)
    print("Saved", RESULTS / "runs.csv", flush=True)


if __name__ == "__main__":
    main()
