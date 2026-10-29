"""Compare saved real-only and 25% synthetic checkpoints on AG News test."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/error_analysis"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
MODELS = {"real_only": ROOT / "models/mixtures/synthetic_000",
          "synthetic_25": ROOT / "models/mixtures/synthetic_025"}


@torch.inference_mode()
def predict(model_dir: Path, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).cuda().eval()
    probabilities = []
    for start in range(0, len(texts), 64):
        tokens = tokenizer(texts[start:start + 64], truncation=True, max_length=128,
                           padding=True, return_tensors="pt").to("cuda")
        probabilities.append(torch.softmax(model(**tokens).logits.float(), dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    probs = np.concatenate(probabilities)
    return probs.argmax(axis=1), probs.max(axis=1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for inference")
    test = load_from_disk(str(ROOT / "data/processed/ag_news"))["test"]
    if len(test) != 7600 or test.features["label"].names != LABELS:
        raise ValueError("Unexpected test set")
    texts, actual = test["text"], np.asarray(test["label"])
    frame = pd.DataFrame({"test_index": range(len(test)), "text": texts,
                          "true_label": actual, "true_category": [LABELS[i] for i in actual]})
    for name, model_dir in MODELS.items():
        print("Predicting", name, "on", torch.cuda.get_device_name(0), flush=True)
        pred, confidence = predict(model_dir, texts)
        frame[f"{name}_label"] = pred
        frame[f"{name}_category"] = [LABELS[i] for i in pred]
        frame[f"{name}_confidence"] = confidence
        frame[f"{name}_correct"] = pred == actual
        saved = json.loads((ROOT / "results/mixture_training" /
                            ("synthetic_000" if name == "real_only" else "synthetic_025") /
                            "metrics.json").read_text(encoding="utf-8"))
        measured = float((pred == actual).mean())
        if abs(measured - saved["summary_row"]["test_accuracy"]) > 1 / len(test):
            raise RuntimeError(f"Predictions do not reproduce saved accuracy for {name}: {measured}")
    frame["outcome"] = np.select(
        [frame.real_only_correct & frame.synthetic_25_correct,
         ~frame.real_only_correct & frame.synthetic_25_correct,
         frame.real_only_correct & ~frame.synthetic_25_correct],
        ["both_correct", "synthetic_fixes", "synthetic_new_error"], default="both_wrong")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "paired_predictions.csv", index=False)
    outcomes = frame.outcome.value_counts().reindex(
        ["both_correct", "synthetic_fixes", "synthetic_new_error", "both_wrong"], fill_value=0)
    confusion = {name: confusion_matrix(actual, frame[f"{name}_label"], labels=range(4)).tolist()
                 for name in MODELS}
    by_class = (frame.groupby("true_category")
                .agg(real_only_accuracy=("real_only_correct", "mean"),
                     synthetic_25_accuracy=("synthetic_25_correct", "mean")))
    by_class["difference"] = by_class.synthetic_25_accuracy - by_class.real_only_accuracy
    by_class.to_csv(OUT / "per_class.csv")
    summary = {"models": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in MODELS.items()},
               "test_count": len(test), "outcomes": outcomes.to_dict(),
               "accuracy": {name: float(frame[f"{name}_correct"].mean()) for name in MODELS},
               "confusion_matrices": confusion,
               "mean_confidence_incorrect": {name: float(frame.loc[~frame[f"{name}_correct"], f"{name}_confidence"].mean()) for name in MODELS}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    matrices = [np.asarray(confusion[name]) for name in MODELS]
    for ax, matrix, title in zip(axes, [*matrices, matrices[1] - matrices[0]],
                                 ["Real only", "25% synthetic", "Difference: synthetic minus real"]):
        ax.imshow(matrix, cmap="Blues" if "Difference" not in title else "RdBu_r")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(4), LABELS, rotation=45, ha="right")
        ax.set_yticks(range(4), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    fig.savefig(OUT / "confusion_comparison.png", dpi=160)
    plt.close(fig)
    # Select examples from every outcome, prioritizing common confusions and
    # low-confidence errors; retain full text and test index for auditability.
    examples = []
    for outcome, count in [("synthetic_new_error", 5), ("synthetic_fixes", 5),
                           ("both_wrong", 4), ("both_correct", 0)]:
        subset = frame[frame.outcome == outcome].copy()
        if outcome == "both_wrong":
            subset["sort_key"] = subset[["real_only_confidence", "synthetic_25_confidence"]].min(axis=1)
            subset = subset.sort_values("sort_key")
        else:
            subset = subset.sort_values(["true_label", "synthetic_25_confidence"], ascending=[True, True])
            subset = pd.concat([part.head(max(1, count // 4)) for _, part in subset.groupby("true_label")])
            if len(subset) < count:
                remaining = frame[(frame.outcome == outcome) & ~frame.test_index.isin(subset.test_index)]
                subset = pd.concat([subset, remaining.sort_values("synthetic_25_confidence").head(count - len(subset))])
        examples.append(subset.head(count))
    pd.concat(examples).drop(columns=["sort_key"], errors="ignore").to_csv(OUT / "representative_errors.csv", index=False)
    print(json.dumps(summary["outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
