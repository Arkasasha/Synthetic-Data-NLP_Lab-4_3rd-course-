# DistilBERT training methodology and results

## Research question

This project tests how the proportion and preparation of synthetic AG News articles affect four-class news classification. The main comparison holds the training set size, class balance, model architecture, and optimization settings fixed while replacing real training articles with generated articles.

## Model and training setup

The classifier starts from `distilbert/distilbert-base-uncased` and adds a four-label sequence-classification head. Its saved configuration specifies six Transformer layers, 12 attention heads, 768-dimensional hidden states, and a 30,522-token vocabulary. The labels are World (0), Sports (1), Business (2), and Sci/Tech (3). The tokenizer truncates articles at 128 tokens; batches are padded dynamically. DistilBERT is fine-tuned end to end rather than used only as a frozen feature extractor.

The controlled baseline and mixture experiments use an NVIDIA GeForce RTX 4060 Laptop GPU, BF16 precision, AdamW, a linear learning-rate schedule, one epoch, batch size 16, evaluation batch size 32, learning rate `5e-5`, weight decay `0.01`, and seed 42. Validation runs every 100 optimizer steps (five checks across 500 training steps). The final model state is saved after the epoch, and the original test split is evaluated once after training. No checkpoint is chosen using test predictions. The exact configuration and training history for each mixture are stored with its model and results.

An earlier standalone baseline appears in [`05_baseline_training.ipynb`](../notebooks/05_baseline_training.ipynb) and [`06_baseline_evaluation.ipynb`](../notebooks/06_baseline_evaluation.ipynb). It obtained test accuracy 0.9078 and macro F1 0.9077. For the main comparison below, the **controlled 0% mixture run** is the appropriate baseline because it was trained within the same pipeline as the synthetic mixtures. Its test accuracy is 0.9075 and macro F1 is 0.9073. These are different trained checkpoints, so their scores should not be treated as interchangeable.

## Data and preprocessing

[`src/prepare_data.py`](../src/prepare_data.py) downloads `fancyzhx/ag_news` through Hugging Face Datasets. Using seed 42, it selects 2,000 training and 500 validation articles per class from the original training split. This yields 8,000 real training articles and 2,000 validation articles. The original 7,600-article test split is retained, with 1,900 examples per class. The preparation script checks sizes, class counts, duplicate text, and overlap among real splits. Prepared datasets are stored at `data/processed/ag_news/`.

Synthetic articles were generated with the local `qwen3:4b-instruct` model using basic and advanced prompt strategies. The generation records sample IDs, labels, prompts or prompt hashes, model information, and text. Quality screening removes near duplicates, possible label mismatches, and other flagged items; replacement articles were then used to produce the balanced [`final_8000.csv`](../data/synthetic/final_8000.csv) source with 2,000 articles per class. The training script checks for missing values, duplicate synthetic text, valid labels and IDs, and normalized-text overlap with all real splits before using this source. Synthetic generation and filtering are implemented in [`src/generate.py`](../src/generate.py) and [`src/filter_data.py`](../src/filter_data.py). The generated articles are fictional, and automated quality checks cannot guarantee that every label or factual detail is correct.

## Main mixture experiment

Five training sets contain 8,000 articles each and remain balanced at 2,000 per class. Their synthetic proportions are 0%, 25%, 50%, 75%, and 100%; the corresponding real/synthetic counts are 8,000/0, 6,000/2,000, 4,000/4,000, 2,000/6,000, and 0/8,000. Within each class, the script samples from the prepared real and synthetic pools, shuffles the mixture, and records selected indices and synthetic IDs in a manifest. The same 2,000 real validation articles and original 7,600 real test articles are used for every model. See [`src/train_mixtures.py`](../src/train_mixtures.py) and [`08_mixture_training.ipynb`](../notebooks/08_mixture_training.ipynb).

Accuracy is the fraction of correct predictions. Precision, recall, and F1 are calculated for each class. **Macro F1** is the arithmetic mean of the four class F1 scores, giving each category equal weight. Per-class recall is the fraction of articles in a given true category classified correctly. Training and validation loss are cross-entropy losses; validation loss measures fit on data excluded from training. Confusion matrices show each true-to-predicted category count.

### Original AG News test results

| Synthetic share | Real / synthetic training articles | Accuracy | Macro F1 | World recall | Sports recall | Business recall | Sci/Tech recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 8,000 / 0 | 0.9075 | 0.9073 | 0.8953 | 0.9826 | 0.8605 | 0.8916 |
| 25% | 6,000 / 2,000 | **0.9087** | **0.9085** | 0.8995 | 0.9789 | 0.8584 | 0.8979 |
| 50% | 4,000 / 4,000 | 0.8997 | 0.8998 | 0.8742 | 0.9826 | 0.8816 | 0.8605 |
| 75% | 2,000 / 6,000 | 0.8908 | 0.8906 | 0.8805 | 0.9758 | 0.8526 | 0.8542 |
| 100% | 0 / 8,000 | 0.7820 | 0.7641 | 0.9005 | 0.9842 | 0.8500 | 0.3932 |

The 25% mixture has the highest **observed test** accuracy and macro F1 in this five-run table. Its macro F1 exceeds the controlled real-only baseline by 0.00127 (0.127 percentage points), a very small difference from one training seed. Performance falls at 50% and 75%, and the 100% synthetic model has a large Sci/Tech recall loss. The 100% model's macro F1 is 0.1431 below the real-only baseline. Full metrics, per-class scores, confusion matrices, and plots are in [`09_performance_comparison.ipynb`](../notebooks/09_performance_comparison.ipynb) and [`results/mixture_comparison/comparison.csv`](../results/mixture_comparison/comparison.csv).

The original mixture logs record training runtime (about 60–68 seconds per run on this GPU), but they do **not** record peak GPU memory. These models were not retrained merely to obtain the missing memory measurement.

## Separate hyperparameter study

Before the main mixtures, a real-data baseline grid tested learning rates `2e-5`, `3e-5`, and `5e-5` crossed with weight decays `0`, `0.01`, and `0.1`: nine runs, five epochs each. For each run, the best epoch was selected by validation macro F1 and its model state saved. The strongest **validation** macro F1 was 0.9146 at learning rate `2e-5`, weight decay `0.01`, epoch 5. This is a validation result, not a test result or a mixture result. The main mixture experiment deliberately retained its fixed one-epoch, `5e-5` setup for comparability; it did not use the grid's best five-epoch checkpoint. Details are in [`src/tune_baseline.py`](../src/tune_baseline.py) and [`results/hyperparameter_tuning/best_variants.csv`](../results/hyperparameter_tuning/best_variants.csv).

## 50% synthetic ablation

The M4 study fixes the synthetic share at 50%: 4,000 real and 4,000 synthetic articles per run, balanced across the four classes. It crosses filtered versus unfiltered source data with two prompt compositions, and repeats each condition at seeds 42, 123, and 2026. Training architecture and settings match the main one-epoch setup. The filtered source is `full_filtered.csv`; the unfiltered source is `full_unfiltered.csv`.

The available basic-prompt pool cannot provide 4,000 basic articles. The **basic-enriched** condition therefore uses 400 basic and 3,600 advanced synthetic articles (100 basic per class), while **advanced-only** uses 4,000 advanced articles. Thus the prompting comparison changes 10% of the synthetic half, not the entire synthetic half. Results below are mean test scores across three seeds; the ± value is sample standard deviation.

| Filtering | Prompt composition | Mean test accuracy | Mean test macro F1 ± SD |
|---|---|---:|---:|
| Unfiltered | Advanced only | **0.9000** | **0.9001 ± 0.0029** |
| Filtered | Advanced only | 0.8986 | 0.8987 ± 0.0023 |
| Unfiltered | Basic enriched | 0.8976 | 0.8977 ± 0.0018 |
| Filtered | Basic enriched | 0.8976 | 0.8976 ± 0.0037 |

Unfiltered advanced-only has the highest observed mean in this design, but differences are small relative to variation across three seeds. The ablation does not show a clear benefit from filtering or basic enrichment under these conditions. Because sampling and seed also vary, these descriptive comparisons do not isolate every possible data-quality effect. Per-class recall, each run's metrics, paired differences, and graphs are in [`12_ablation_experiments.ipynb`](../notebooks/12_ablation_experiments.ipynb) and [`results/ablations/comparison.csv`](../results/ablations/comparison.csv).

## Error analysis and limitations

The controlled 0% and 25% checkpoints were run on the same 7,600 test articles. The 25% model corrects 144 articles missed by the real-only model, introduces 135 new errors, and leaves 559 articles wrong in both models. Its net gain is nine correct predictions. World and Sci/Tech recall rise slightly; Sports and Business recall fall slightly. Business↔Sci/Tech remains a common confusion, consistent with news articles that mix technical and financial angles. Among incorrect predictions, average maximum softmax confidence is 0.775 for the real-only model and 0.828 for the 25% model; these values are **not calibrated** probabilities of correctness. The paired predictions and 14 inspected examples are in [`13_error_analysis.ipynb`](../notebooks/13_error_analysis.ipynb).

The main mixtures use only one seed and one epoch, so the small 25% lead may reflect training or sample variation. The 25% model was identified by inspecting test scores; that test set therefore cannot provide an unbiased estimate of a model chosen this way. The M4 ablation uses only three seeds and a limited basic-prompt pool. Synthetic quality and label checks are imperfect, and some original test articles are short, noisy, or semantically ambiguous. Together, the results support a cautious conclusion: modest synthetic augmentation can match real-only performance in this setup, while replacing most or all real articles degrades performance, especially for Sci/Tech at 100% synthetic. A stronger claim would require validation-based model selection, more seeds, and an untouched final evaluation set.
