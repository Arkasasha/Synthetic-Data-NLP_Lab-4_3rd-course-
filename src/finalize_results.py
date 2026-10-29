"""Build final tables, figures, and an auditable experiment index from saved runs."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final"
MIXTURE = ROOT / "results/mixture_comparison/comparison.csv"
ABLATION = ROOT / "results/ablations/comparison.csv"
RUNS = ROOT / "results/ablations/runs.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_results = pd.read_csv(MIXTURE).sort_values("synthetic_percentage")
    if main_results.synthetic_percentage.tolist() != [0, 25, 50, 75, 100]:
        raise ValueError("Missing mixture experiment")
    if main_results.groupby("synthetic_percentage").size().ne(1).any():
        raise ValueError("Duplicate mixture experiment")
    main_results.to_csv(OUT / "main_results.csv", index=False)
    ablation = pd.read_csv(ABLATION)
    runs = pd.read_csv(RUNS)
    if len(ablation) != 4 or len(runs) != 12 or runs.groupby(["filtering", "prompt_mix"]).size().ne(3).any():
        raise ValueError("Incomplete ablation experiment")
    ablation.to_csv(OUT / "ablation_results.csv", index=False)
    splits = load_from_disk(str(ROOT / "data/processed/ag_news"))
    if {name: len(split) for name, split in splits.items()} != {"train": 8000, "validation": 2000, "test": 7600}:
        raise ValueError("Unexpected prepared dataset")
    requirements = [Requirement(line) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
    versions = {req.name: metadata.version(req.name) for req in requirements}
    unsatisfied = [str(req) for req in requirements if versions[req.name] not in req.specifier]
    if unsatisfied:
        raise RuntimeError(f"Unsatisfied requirements: {unsatisfied}")

    # Main figure: scores from the fixed original test set, with the baseline shown.
    x = main_results.synthetic_percentage.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(x, 100 * main_results.test_accuracy, "o-", linewidth=2, label="Accuracy")
    ax.plot(x, 100 * main_results.test_macro_f1, "s-", linewidth=2, label="Macro F1")
    ax.axhline(100 * main_results.iloc[0].test_macro_f1, color="gray", linestyle="--", linewidth=1,
               label="0% macro F1")
    ax.set(xlabel="Synthetic share of 8,000 training articles (%)", ylabel="Test score (%)",
           title="DistilBERT on the original 7,600-article AG News test set")
    ax.set_xticks(x)
    ax.set_ylim(74, 94)
    ax.grid(alpha=.25)
    ax.legend(loc="lower left")
    fig.savefig(OUT / "performance_vs_synthetic.png", dpi=180)
    plt.close(fig)

    labels = [f"{row.filtering}\n{row.prompt_mix.replace('_', ' ')}" for row in ablation.itertuples()]
    positions = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for ax, metric, title in zip(axes, ["test_accuracy", "test_macro_f1"], ["Accuracy", "Macro F1"]):
        ax.errorbar(positions, 100 * ablation[f"{metric}_mean"],
                    yerr=100 * ablation[f"{metric}_std"], fmt="o", markersize=7, capsize=5)
        ax.set(title=title, ylabel="Mean test score ± SD (%)", xticks=positions,
               xticklabels=labels, ylim=(88, 91.2))
        ax.grid(axis="y", alpha=.25)
        ax.text(.02, .02, "Axis starts at 88%", transform=ax.transAxes, fontsize=8)
    fig.suptitle("50% synthetic: three seeds per condition")
    fig.savefig(OUT / "ablation_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, category in zip(axes.flat, ["world", "sports", "business", "sci_tech"]):
        key = f"test_recall_{category}"
        values = 100 * ablation[f"{key}_mean"].to_numpy()
        errors = 100 * ablation[f"{key}_std"].to_numpy()
        ax.errorbar(positions, values, yerr=errors, fmt="o", capsize=4)
        low, high = (values - errors).min(), (values + errors).max()
        ax.set_ylim(low - 1, high + 1)
        ax.set(title=category.replace("_", "/").title(), ylabel="Mean recall ± SD (%)",
               xticks=positions, xticklabels=labels)
        ax.grid(axis="y", alpha=.25)
        ax.text(.02, .02, "Truncated axis", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Per-class recall, 50% synthetic ablations")
    fig.savefig(OUT / "ablation_per_class_recall.png", dpi=180)
    plt.close(fig)

    sources = [MIXTURE, ABLATION, RUNS,
               ROOT / "results/hyperparameter_tuning/best_variants.csv",
               ROOT / "results/mixture_training/summary.json",
               ROOT / "results/ablations/paired_differences.json",
               ROOT / "results/error_analysis/summary.json",
               ROOT / "data/synthetic/final_8000.csv"]
    index = {
        "dataset": "fancyzhx/ag_news", "prepared_split_sizes": {name: len(split) for name, split in splits.items()},
        "main_experiment": {"script": "src/train_mixtures.py", "seed": 42, "train_size": 8000,
                            "learning_rate": 5e-5, "weight_decay": .01, "epochs": 1,
                            "batch_size": 16, "validation_interval_steps": 100,
                            "synthetic_percentages": [0, 25, 50, 75, 100],
                            "training_runtime_seconds": main_results[["synthetic_percentage", "training_runtime_seconds"]].to_dict("records"),
                            "peak_gpu_memory": "not recorded during original runs"},
        "ablation_experiment": {"script": "src/run_ablations.py", "seeds": [42, 123, 2026],
                                "train_size": 8000, "synthetic_percentage": 50,
                                "learning_rate": 5e-5, "weight_decay": .01, "epochs": 1,
                                "batch_size": 16, "run_count": len(runs),
                                "mean_training_runtime_seconds": float(runs.train_runtime_sec.mean()),
                                "peak_gpu_reserved_gb_range": [float(runs.peak_gpu_reserved_gb.min()),
                                                               float(runs.peak_gpu_reserved_gb.max())]},
        "hyperparameter_grid": {"script": "src/tune_baseline.py", "learning_rates": [2e-5, 3e-5, 5e-5],
                                "weight_decays": [0, .01, .1], "epochs_per_run": 5},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "gpu_total_memory_mib": int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
                     if torch.cuda.is_available() else None,
                     "cpu": platform.processor() or "not available", "os": platform.platform()},
        "software": {"python": platform.python_version(), **versions},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in sources},
        "result_files": ["results/final/main_results.csv", "results/final/ablation_results.csv",
                         "results/final/performance_vs_synthetic.png", "results/final/ablation_comparison.png",
                         "results/final/ablation_per_class_recall.png"],
    }
    (OUT / "experiment_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("Final results written to", OUT)
    print("Requirements verified:", len(requirements))


if __name__ == "__main__":
    main()
