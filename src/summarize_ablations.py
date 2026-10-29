"""Summarize completed ablation runs and generate the comparison notebook."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ablations"
NOTEBOOK = ROOT / "notebooks/12_ablation_experiments.ipynb"
METRICS = ["test_accuracy", "test_macro_f1", "test_recall_world", "test_recall_sports",
           "test_recall_business", "test_recall_sci_tech"]


def summarize() -> None:
    runs = pd.read_csv(OUT / "runs.csv")
    if len(runs) != 12 or runs.groupby(["filtering", "prompt_mix"]).size().ne(3).any():
        raise ValueError("Expected three seeds in each of four ablation conditions")
    group = runs.groupby(["filtering", "prompt_mix"], sort=True)
    means = group[METRICS + ["validation_accuracy", "validation_macro_f1",
                             "train_runtime_sec", "peak_gpu_reserved_gb"]].agg(["mean", "std"])
    means.columns = [f"{metric}_{stat}" for metric, stat in means.columns]
    means.reset_index().to_csv(OUT / "comparison.csv", index=False)
    plot = means.reset_index()
    plot["condition"] = plot["filtering"] + "\n" + plot["prompt_mix"].str.replace("_", " ")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, metric in zip(axes.flat, METRICS):
        ax.bar(plot["condition"], plot[f"{metric}_mean"], yerr=plot[f"{metric}_std"], capsize=4)
        ax.set_title(metric.replace("test_", "").replace("_", " ").title())
        ax.set_ylabel("Mean score ± SD over 3 seeds")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("50% synthetic: prompting and filtering ablations")
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    # Paired differences use the same seeds in both conditions.
    paired = []
    for filter_name in ("unfiltered", "filtered"):
        subset = runs[runs.filtering == filter_name].pivot(index="seed", columns="prompt_mix", values="test_macro_f1")
        diff = subset["basic_enriched"] - subset["advanced_only"]
        paired.append({"comparison": "basic_enriched minus advanced_only", "filtering": filter_name,
                       "mean_macro_f1_difference": float(diff.mean()), "seed_differences": diff.to_dict()})
    for prompt_name in ("advanced_only", "basic_enriched"):
        subset = runs[runs.prompt_mix == prompt_name].pivot(index="seed", columns="filtering", values="test_macro_f1")
        diff = subset["filtered"] - subset["unfiltered"]
        paired.append({"comparison": "filtered minus unfiltered", "prompt_mix": prompt_name,
                       "mean_macro_f1_difference": float(diff.mean()), "seed_differences": diff.to_dict()})
    (OUT / "paired_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")


def notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# M4: 50% synthetic ablation experiments\n\n"
            "Four conditions compare prompting composition and synthetic filtering over seeds 42, 123, and 2026. "
            "Every run uses 4,000 real and 4,000 synthetic training examples, balanced at 2,000 per class. "
            "The basic pool is too small for 4,000 basic examples, so *basic enriched* uses 400 basic "
            "(100 per class) and 3,600 advanced; *advanced only* uses 4,000 advanced. "
            "Filtered data comes from `full_filtered.csv`; unfiltered data comes from `full_unfiltered.csv`. "
            "All runs use DistilBERT, one epoch, learning rate 5e-5, weight decay 0.01, batch size 16, "
            "max length 128, and BF16 on CUDA. Validation is checked every 100 steps; the fixed original "
            "AG News test set is evaluated after training. No test results are used to choose checkpoints. "
            "The results describe this finite set of seeds and synthetic samples; the basic comparison "
            "changes only 10% of the synthetic half (5% of the entire training set)."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nOUT = ROOT / 'results' / 'ablations'\nassert (OUT / 'runs.csv').exists(), 'Run python src/run_ablations.py first'"),
        nbf.v4.new_code_cell("runs = pd.read_csv(OUT / 'runs.csv')\ndisplay(runs[['filtering','prompt_mix','seed','basic_count','advanced_count','validation_macro_f1','test_accuracy','test_macro_f1','test_recall_world','test_recall_sports','test_recall_business','test_recall_sci_tech','train_runtime_sec','peak_gpu_reserved_gb']].round(4))"),
        nbf.v4.new_code_cell("comparison = pd.read_csv(OUT / 'comparison.csv')\ndisplay(comparison.round(4))"),
        nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'comparison.png')))"),
        nbf.v4.new_code_cell("paired = json.loads((OUT / 'paired_differences.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame(paired))"),
        nbf.v4.new_code_cell("best = comparison.loc[comparison['test_macro_f1_mean'].idxmax()]\nprint(f\"Highest mean test macro F1: {best['filtering']} / {best['prompt_mix']} = {best['test_macro_f1_mean']:.4f} ± {best['test_macro_f1_std']:.4f}\")\nprint('Small sample: three training seeds per condition. Treat differences as descriptive, not definitive.')"),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    summarize()
    notebook()
    print("Saved", NOTEBOOK)
