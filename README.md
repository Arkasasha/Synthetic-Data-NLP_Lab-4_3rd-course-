# Synthetic Data NLP: AG News classification

This project studies how synthetic news articles affect a four-class DistilBERT classifier. The controlled experiment trains on five 8,000-article mixtures containing 0%, 25%, 50%, 75%, or 100% synthetic data. It evaluates every model on the original 7,600-article AG News test split. A separate 50% synthetic ablation compares prompt composition and filtering across three random seeds.

## Results at a glance

| Synthetic share | Real / synthetic training articles | Test accuracy | Test macro F1 |
|---:|---:|---:|---:|
| 0% | 8,000 / 0 | 0.9075 | 0.9073 |
| 25% | 6,000 / 2,000 | **0.9087** | **0.9085** |
| 50% | 4,000 / 4,000 | 0.8997 | 0.8998 |
| 75% | 2,000 / 6,000 | 0.8908 | 0.8906 |
| 100% | 0 / 8,000 | 0.7820 | 0.7641 |

The 25% mixture has the highest observed test score, but its lead over 0% is small and comes from one seed. See the [training report](report/training_results.md), [data-generation report](report/data_generation.md), and [error-analysis notebook](notebooks/13_error_analysis.ipynb) for methods, ablations, and limitations.

![Performance versus synthetic share](results/final/performance_vs_synthetic.png)

## Project structure

- `src/`: data preparation, generation, filtering, training, evaluation, and final-result scripts.
- `prompts/`: basic and advanced synthetic-generation prompts.
- `data/synthetic/`: saved generated datasets, including the balanced `final_8000.csv` used for the main experiment.
- `data/processed/ag_news/`: reproducible prepared splits, created locally and excluded from Git.
- `models/`: saved model checkpoints created locally and excluded from Git.
- `notebooks/`: executed exploration, training, comparison, ablation, and error-analysis notebooks.
- `results/`: metrics, manifests, logs, and figures. `results/final/` contains the submission tables, figures, and experiment index.
- `report/`: the two scientific-report sections.

## Set up the environment

Use Python 3.11 from the repository root. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The training scripts require an NVIDIA CUDA GPU. Install a CUDA-enabled PyTorch build compatible with your GPU if the final command prints `False`. The completed runs used PyTorch 2.13.0+cu132 and an NVIDIA GeForce RTX 4060 Laptop GPU with about 8 GiB of VRAM. The declared dependencies were checked in the project environment with `pip check` and against the version bounds in `requirements.txt`.

For Jupyter, register this environment and open the notebooks:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name synthetic-data-nlp --display-name "Synthetic Data NLP"
.\.venv\Scripts\python.exe -m jupyter lab
```

Choose the **Synthetic Data NLP** kernel. The executed notebooks already contain their saved tables and plots.

## Reproduce the data and experiments

The original real AG News split is reconstructed with seed 42. The synthetic CSV sources needed for the experiments are included in `data/synthetic/`. Data preparation downloads `fancyzhx/ag_news` through Hugging Face Datasets; network access is needed on a new machine.

```powershell
.\.venv\Scripts\python.exe src/prepare_data.py
.\.venv\Scripts\python.exe src/train_mixtures.py
.\.venv\Scripts\python.exe src/run_ablations.py
.\.venv\Scripts\python.exe src/analyze_errors.py
.\.venv\Scripts\python.exe src/finalize_results.py
```

`train_mixtures.py` trains five one-epoch models or reuses completed checkpoints. `run_ablations.py` trains 12 one-epoch models or reuses completed checkpoints. `analyze_errors.py` performs GPU inference on the 0% and 25% checkpoints without retraining. `finalize_results.py` validates the saved datasets and tables, checks installed requirement versions, and rebuilds the submission figures and index. Model files are ignored by Git because of their size; reproducing predictions on a new clone requires running the training commands first.

The optional real-data hyperparameter study is `src/tune_baseline.py`: nine learning-rate/weight-decay combinations, five epochs each. The optional generation pipeline in `src/generate.py` and `src/filter_data.py` uses a local Ollama service with `qwen3:4b-instruct` and `all-minilm`; its stored CSV outputs are sufficient to reproduce the training experiments without regenerating articles. Prompt hashes, model metadata, and filtering counts are documented in the [data-generation report](report/data_generation.md).

## Final artifacts and verification

- [Main results table](results/final/main_results.csv) and [performance graph](results/final/performance_vs_synthetic.png).
- [Ablation results table](results/final/ablation_results.csv), [accuracy and macro-F1 graph](results/final/ablation_comparison.png), and [per-class recall graph](results/final/ablation_per_class_recall.png).
- [Experiment index](results/final/experiment_index.json): source-file SHA-256 hashes, hyperparameters, software versions, hardware, and runtime information.
- Detailed logs and manifests: `results/mixture_training/`, `results/ablations/`, and `results/hyperparameter_tuning/`.
- [Paired prediction audit](results/error_analysis/paired_predictions.csv) and [representative errors](results/error_analysis/representative_errors.csv).

The original five mixture runs recorded training time (60–68 seconds per run) but did **not** record peak GPU memory. The 12 later ablation runs did record peak PyTorch reserved GPU memory (about 1.44–1.53 GiB). No old models were retrained solely to fill that missing field.

To verify the local submission state and compare it with GitHub:

```powershell
git status --short
git branch --show-current
git fetch origin
git status -sb
```

A clean working tree and a branch with no ahead commits indicate that tracked local code and results match the remote branch. Git ignores local checkpoints and prepared dataset caches; their omission from GitHub is intentional and documented above.
