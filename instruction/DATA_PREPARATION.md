# AG News data preparation

Use Python 3.11. From the project root on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\prepare_data.py
```

The script uses seed 42 and downloads `fancyzhx/ag_news`. It creates 8,000 training
samples (2,000 per class), 2,000 validation samples (500 per class), and keeps
the original 7,600-sample test set (1,900 per class). It checks sizes, label
counts, duplicate text, and overlap between splits before saving to
`data/processed/ag_news/`. The generated data is excluded from Git.

To load the prepared splits:

```python
from datasets import load_from_disk

splits = load_from_disk("data/processed/ag_news")
```
