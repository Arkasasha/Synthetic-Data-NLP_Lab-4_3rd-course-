"""Generate a resumable, labeled AG News pilot set with local Qwen3-4B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "basic.txt"
DEFAULT_OUTPUT = ROOT / "data" / "synthetic" / "pilot_100.csv"
MODEL = "qwen3:4b-instruct"
API = "http://127.0.0.1:11434"
CATEGORIES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
FIELDS = (
    "sample_id", "label", "category", "headline", "summary", "text",
    "seed", "temperature", "model", "model_digest", "prompt_sha256",
)
SYSTEM_PROMPT = (
    "Write original, plausible, fictional English news briefs for a topic "
    "classification dataset. Do not copy existing articles. Return only "
    "Headline: and Summary: lines, without markdown or category labels."
)


def _model_digest(session: requests.Session) -> str:
    response = session.get(f"{API}/api/tags", timeout=10)
    response.raise_for_status()
    record = next(
        (item for item in response.json()["models"] if item["name"] == MODEL),
        None,
    )
    if record is None:
        raise RuntimeError(f"Model missing. Run: ollama pull {MODEL}")
    details = record["details"]
    if details["family"] != "qwen3" or details["parameter_size"] != "4.0B":
        raise RuntimeError(f"{MODEL} is not a Qwen3 4B model")
    return record["digest"]


def _parse_article(raw: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*Headline:[ \t]*([^\r\n]+)\r?\nSummary:[ \t]*(.+?)\s*",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError("Response did not contain Headline and Summary lines")
    headline, summary = match.group(1).strip(), match.group(2).strip()
    if not headline or not summary:
        raise ValueError("Headline or summary was empty")
    return headline, summary


def _load_existing(
    output: Path, expected: dict[str, tuple[int, str]],
    temperature: float, digest: str, prompt_hash: str,
) -> dict[str, dict[str, str]]:
    if not output.exists() or output.stat().st_size == 0:
        return {}
    with output.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected CSV columns in {output}")
        rows = {}
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id not in expected or sample_id in rows:
                raise ValueError(f"Unexpected or duplicate sample ID: {sample_id}")
            label, category = expected[sample_id]
            if (int(row["label"]) != label or row["category"] != category
                    or row["model"] != MODEL or row["model_digest"] != digest
                    or row["prompt_sha256"] != prompt_hash
                    or float(row["temperature"]) != temperature):
                raise ValueError(f"Existing sample has different settings: {sample_id}")
            rows[sample_id] = row
    return rows


def _save_atomic(output: Path, rows: dict[str, dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=".pilot_100_", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows.values())
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def generate_articles(
    samples_per_category: int = 25,
    temperature: float = 0.7,
    output_path: Path = DEFAULT_OUTPUT,
    base_seed: int = 42,
    max_retries: int = 3,
) -> list[dict[str, str]]:
    """Generate missing samples, assigning AG News labels 0-3 and saving each one."""
    if samples_per_category < 1 or max_retries < 1:
        raise ValueError("Sample count and retry count must be positive")
    if not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2")

    output = Path(output_path)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    expected = {
        f"{label}_{index:03d}": (label, category)
        for label, category in CATEGORIES.items()
        for index in range(1, samples_per_category + 1)
    }

    with requests.Session() as session:
        digest = _model_digest(session)
        rows = _load_existing(output, expected, temperature, digest, prompt_hash)
        failures = []
        for sample_id, (label, category) in expected.items():
            if sample_id in rows:
                continue
            index = int(sample_id.split("_")[1])
            prompt = prompt_template.format(category=category)
            for attempt in range(max_retries):
                seed = base_seed + label * 10000 + index * 10 + attempt
                try:
                    response = session.post(
                        f"{API}/api/chat",
                        json={
                            "model": MODEL,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            "stream": False,
                            "think": False,
                            "options": {
                                "seed": seed, "temperature": temperature,
                                "top_p": 0.8, "top_k": 20, "num_predict": 180,
                            },
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    headline, summary = _parse_article(
                        response.json()["message"]["content"]
                    )
                except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                    print(f"{sample_id}, attempt {attempt + 1}: {error}", file=sys.stderr)
                    if attempt + 1 < max_retries:
                        time.sleep(min(2 ** attempt, 4))
                    continue

                rows[sample_id] = {
                    "sample_id": sample_id,
                    "label": str(label),
                    "category": category,
                    "headline": headline,
                    "summary": summary,
                    "text": f"{headline} {summary}",
                    "seed": str(seed),
                    "temperature": str(temperature),
                    "model": MODEL,
                    "model_digest": digest,
                    "prompt_sha256": prompt_hash,
                }
                _save_atomic(output, rows)
                print(f"{len(rows)}/{len(expected)} {sample_id}", flush=True)
                break
            else:
                failures.append(sample_id)

    if failures:
        raise RuntimeError(
            f"{len(failures)} samples failed: {', '.join(failures)}. "
            "Run the same command again to resume."
        )
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-category", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    rows = generate_articles(
        samples_per_category=args.samples_per_category,
        temperature=args.temperature,
        output_path=args.output,
        base_seed=args.seed,
        max_retries=args.max_retries,
    )
    print(f"Complete: {len(rows)} samples in {args.output}")


if __name__ == "__main__":
    main()
