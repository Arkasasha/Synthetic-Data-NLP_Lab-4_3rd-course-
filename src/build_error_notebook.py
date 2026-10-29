"""Create an executable, evidence-backed notebook from paired test predictions."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/13_error_analysis.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# M4: Classification error analysis\n\n"
        "Compare the saved **0% synthetic (real-only)** and **25% synthetic** DistilBERT checkpoints "
        "on the same original 7,600-article AG News test split. The 25% checkpoint had the highest "
        "test macro F1 among the five M3 mixtures. This choice was made after inspecting test scores, "
        "so the comparison is exploratory and its small lead should not be treated as an unbiased "
        "model-selection estimate. Both checkpoints use one epoch, seed 42, learning rate 5e-5, "
        "weight decay 0.01, and the same architecture. GPU inference is performed once by "
        "`python src/analyze_errors.py`; this notebook reads the saved paired predictions."),
    nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\n"
        "import matplotlib.pyplot as plt\nfrom IPython.display import display, Image\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "OUT = ROOT / 'results' / 'error_analysis'\n"
        "pred = pd.read_csv(OUT / 'paired_predictions.csv')\n"
        "examples = pd.read_csv(OUT / 'representative_errors.csv')\n"
        "summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))\n"
        "assert len(pred) == 7600 and len(examples) >= 10"),
    nbf.v4.new_markdown_cell("## Overall paired outcomes\n\nA fix means the baseline was wrong and the synthetic model was correct. A new error means the reverse. Each row is the *same test article*, so these counts expose what changed beyond aggregate accuracy."),
    nbf.v4.new_code_cell(
        "display(pd.DataFrame({'count': pred.outcome.value_counts(), 'share': pred.outcome.value_counts(normalize=True)}).round(4))\n"
        "print('Real-only accuracy:', round(summary['accuracy']['real_only'], 4))\n"
        "print('25% synthetic accuracy:', round(summary['accuracy']['synthetic_25'], 4))\n"
        "print('Net additional correct:', summary['outcomes']['synthetic_fixes'] - summary['outcomes']['synthetic_new_error'])"),
    nbf.v4.new_markdown_cell("## Category-level errors and confusions\n\nRows are true labels; columns are predictions. The difference matrix is synthetic minus real-only. Positive diagonal values mean more correct predictions; positive off-diagonal values mean more mistakes of that type."),
    nbf.v4.new_code_cell(
        "display(Image(filename=str(OUT / 'confusion_comparison.png')))\n"
        "display(pd.read_csv(OUT / 'per_class.csv').round(4))\n"
        "for name in ['real_only', 'synthetic_25']:\n"
        "    wrong = pred[~pred[f'{name}_correct']]\n"
        "    print('\\n', name, 'most frequent confusions')\n"
        "    display(wrong.groupby(['true_category', f'{name}_category']).size().sort_values(ascending=False).head(8).rename('count').reset_index())"),
    nbf.v4.new_markdown_cell("## Representative articles\n\nThe table contains **5 new errors, 5 fixes, and 4 errors shared by both models**. Selection is deterministic and spans categories where possible. Full article text and test indices are retained in `representative_errors.csv`; the display shortens text for readability. These examples illustrate mechanisms, not their population frequency."),
    nbf.v4.new_code_cell(
        "view = examples[['test_index','outcome','true_category','real_only_category','synthetic_25_category','real_only_confidence','synthetic_25_confidence','text']].copy()\n"
        "view['text'] = view['text'].str.slice(0, 320)\n"
        "display(view.round(3))"),
    nbf.v4.new_markdown_cell(
        "### What the examples suggest\n\n"
        "- **Topic boundaries are ambiguous.** Test article 1273 discusses box-office results and is labeled Business, while the synthetic model predicts World. Article 3916 mixes election policy and stem-cell research; the synthetic model corrects a baseline Sci/Tech prediction to World.\n"
        "- **Context can outweigh the headline.** Article 952 concerns TV coverage of the Olympics: the baseline predicts Business and the synthetic model predicts Sports. Article 5024 has a venue/reunion headline and moves the other way, from a correct Sports prediction to World.\n"
        "- **Some labels may be debatable from the short visible text.** Article 7440 is labeled Sci/Tech although the excerpt mentions holiday bargain hunting; it is included as an observed new error, not proof that the model learned the wrong concept. The full text should be read before assigning a cause.\n"
        "- **Both models can fail on unusual wording.** Article 481 uses metaphorical language about a business takeover; the baseline predicts Sci/Tech and the synthetic model predicts World instead of Business.\n\n"
        "These observations are hypotheses about individual predictions. They do not establish that synthetic data itself caused a particular semantic shift."),
    nbf.v4.new_markdown_cell("## Confidence and difficult predictions\n\nConfidence here is the largest softmax probability. It is a ranking signal, not a calibrated probability of correctness."),
    nbf.v4.new_code_cell(
        "rows = []\n"
        "for name in ['real_only','synthetic_25']:\n"
        "    wrong = pred[~pred[f'{name}_correct']]\n"
        "    rows.append({'model':name,'incorrect':len(wrong),'incorrect_confidence_below_0_5':int((wrong[f'{name}_confidence']<0.5).sum()),'incorrect_confidence_at_least_0_9':int((wrong[f'{name}_confidence']>=0.9).sum()),'mean_incorrect_confidence':wrong[f'{name}_confidence'].mean()})\n"
        "display(pd.DataFrame(rows).round(4))\n"
        "low = pred[(~pred.synthetic_25_correct) & (pred.synthetic_25_confidence < 0.5)]\n"
        "display(low.nsmallest(8, 'synthetic_25_confidence')[['test_index','true_category','synthetic_25_category','synthetic_25_confidence','text']].assign(text=lambda d:d.text.str.slice(0,220)).round(3))"),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "for name in ['real_only','synthetic_25']:\n"
        "    ax.hist(pred.loc[~pred[f'{name}_correct'],f'{name}_confidence'],bins=np.linspace(0.25,1,16),alpha=.55,label=name)\n"
        "ax.set(xlabel='Maximum softmax confidence on incorrect predictions',ylabel='Number of articles',title='Confidence of errors')\n"
        "ax.legend(); fig.tight_layout(); fig.savefig(OUT/'error_confidence.png',dpi=160); display(fig); plt.close(fig)"),
    nbf.v4.new_markdown_cell(
        "## Limitations\n\n"
        "The 25% model gains only nine net correct predictions: 144 fixes and 135 new errors. "
        "World and Sci/Tech recall improve slightly while Sports and Business recall fall slightly. "
        "The frequent Business–Sci/Tech confusion remains in both models, consistent with articles "
        "about technology companies or technical products that also have a financial angle. "
        "The models were trained with one seed and one epoch, and some test articles have short or "
        "noisy text. Softmax confidence is uncalibrated. The selected synthetic mixture was chosen "
        "using these test scores, and individual example interpretations are qualitative; repeat "
        "with held-out selection data and multiple seeds before claiming a robust gain."),
]
nbf.write(nb, NOTEBOOK)
print("Saved", NOTEBOOK)
