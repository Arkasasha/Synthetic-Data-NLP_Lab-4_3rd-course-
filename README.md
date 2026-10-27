# Synthetic Data NLP

Project 3: synthetic data for AG News classification.

## Project layout

- `data/raw/`: original AG News data when prepared.
- `data/processed/`: fixed train, validation, and test splits when prepared.
- `data/synthetic/`: generated articles; TASK-006 samples are in `task006/`.
- `notebooks/`: experiment notebooks, added as their tasks are completed.
- `prompts/`: reusable basic and advanced generation prompts.
- `src/`: reusable data, generation, filtering, training, and evaluation code.
- `models/`: saved model weights.
- `results/`: metrics, review sheets, experiment logs, and `figures/`.
- `report/` and `presentation/`: final write-up and slides.

Empty directories contain `.gitkeep`; future notebooks, scripts, metrics, and report files will be added when those experiments are done.

## TASK-006: Prompt engineering

The experiment in [notebooks/03_prompt_engineering.ipynb](notebooks/03_prompt_engineering.ipynb) generates 40 synthetic AG News briefs with two saved prompts. It uses the local Ollama `qwen3:4b-instruct` model (Qwen3-4B-Instruct-2507, Q4_K_M); the exact model digest, seeds, and decoding settings are recorded in the output manifest.

Run from the repository root with Ollama running:

```powershell
ollama pull qwen3:4b-instruct
python -m pip install numpy pandas scikit-learn requests jupyterlab ipykernel
jupyter lab notebooks/03_prompt_engineering.ipynb
```

The notebook loads `prompts/basic.txt` and `prompts/advanced.txt`. It saves raw generations to `data/synthetic/task006/samples.jsonl`, automatic metrics and a run manifest to `results/task006/`, and reads the scored review in `results/task006/review_scored.csv` for its quality comparison. The notebook resumes an interrupted run without regenerating completed samples; move the TASK-006 sample and result directories before starting a new experiment with different settings.

The generated articles are fictional. Check factual plausibility and category fit before using them to train a classifier.
