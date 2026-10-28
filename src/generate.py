"""Generate resumable AG News pilot, full, and replacement sets with Qwen3-4B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import os
import random
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
FULL_OUTPUT = ROOT / "data" / "synthetic" / "full_synthetic.csv"
ADVANCED_SCENARIOS = {
    "World": (
        "regional diplomatic talks", "election administration", "cross-border transport agreement",
        "flood relief coordination", "trade negotiations", "water-sharing talks",
        "public health cooperation", "migration policy", "port security agreement",
        "education reform", "wildfire response", "local government budget",
    ),
    "Sports": (
        "football league match", "tennis qualifying upset", "athletics championship selection",
        "basketball coaching change", "cricket tournament opener", "cycling stage race",
        "volleyball final", "swimming championship", "rugby league match",
        "marathon training program", "youth football tournament", "badminton semifinal",
    ),
    "Business": (
        "small manufacturer earnings", "retail chain expansion", "labor contract talks",
        "agricultural commodity prices", "competition review of a merger",
        "regional bank lending", "supply chain investment", "insurance market changes",
        "restaurant franchise growth", "shipping costs", "renewable energy financing",
        "housing construction demand",
    ),
    "Sci/Tech": (
        "medical device prototype", "software security update", "small satellite mission",
        "battery materials research", "AI tool deployment at a university",
        "ocean sensor trial", "robotics laboratory test", "public transit software",
        "crop monitoring research", "recycling technology pilot", "astronomy observation",
        "hospital data system study",
    ),
}
SCENARIO_PLACES = (
    "coastal Portugal", "northern Kenya", "southern Chile", "western Canada",
    "central India", "eastern Poland", "rural Japan", "northern Morocco",
    "southern Brazil", "western Australia", "central Mexico", "eastern Indonesia",
    "the Scottish Highlands", "southern Italy", "western France", "northern Vietnam",
    "eastern South Africa", "central Turkey", "southern Sweden", "western Argentina",
    "northern Thailand", "eastern Romania", "southern New Zealand", "central Colombia",
)
SCENARIO_ANGLES = (
    "a delayed launch", "a revised funding plan", "a disputed result",
    "a new local partnership", "an unexpected technical setback",
    "a small-scale trial", "a change in leadership", "an early evaluation",
    "a community response", "a revised timetable", "a first public report",
    "a decision after negotiations",
)
FIELDS = (
    "sample_id", "label", "category", "headline", "summary", "text",
    "seed", "temperature", "model", "model_digest", "prompt_sha256",
)
FULL_FIELDS = FIELDS + (
    "strategy", "scenario", "top_p", "top_k", "num_predict",
    "ollama_version", "system_prompt_sha256",
)
SYSTEM_PROMPT = (
    "Write original, plausible, fictional English news briefs for a topic "
    "classification dataset. Do not copy existing articles. Return only "
    "Headline: and Summary: lines, without markdown or category labels."
)
FULL_SYSTEM_PROMPT = (
    "Write original, plausible, fictional English news briefs for a topic "
    "classification dataset. Use invented people, teams, companies and "
    "institutions; real geographic place names are allowed. Avoid claims "
    "about real celebrities, actual company announcements and sweeping "
    "world-changing events. Do not copy existing articles. Return only "
    "Headline: and Summary: lines, without markdown or category labels."
)
REPLACEMENT_SYSTEM_PROMPT = (
    FULL_SYSTEM_PROMPT
    + " Keep the headline and summary together between 35 and 50 words. "
    "Keep every place in its real country and geographical setting. "
    "Do not conflate ODI and T20 cricket, or make impossible scientific "
    "and sports claims. Use ordinary, established crops, materials and "
    "sports formats rather than invented species or technical jargon. "
    "If unsure about a place or rule, use a simpler fictional local event."
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


def _full_plan(
    samples_per_category: int,
    base_seed: int,
    scenario_places: tuple[str, ...] = SCENARIO_PLACES,
):
    """Interleave categories so an interrupted run stays roughly balanced."""
    scenarios = {}
    for label, category in CATEGORIES.items():
        combinations = list(itertools.product(
            ADVANCED_SCENARIOS[category], scenario_places, SCENARIO_ANGLES
        ))
        random.Random(base_seed + label * 100_000).shuffle(combinations)
        scenarios[category] = combinations
    for index in range(1, samples_per_category + 1):
        for label, category in CATEGORIES.items():
            strategy = "basic" if index % 10 == 0 else "advanced"
            scenario = ""
            if strategy == "advanced":
                beat, place, angle = scenarios[category][
                    (index - 1) % len(scenarios[category])
                ]
                scenario = (
                    f"{beat} in {place}, involving {angle}"
                )
            yield f"{label}_{index:04d}", label, category, strategy, scenario


def _save_full_atomic(output: Path, rows: dict[str, dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=".full_synthetic_", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=FULL_FIELDS)
            writer.writeheader()
            writer.writerows(rows.values())
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def generate_full_dataset(
    samples_per_category: int = 2000,
    temperature: float = 0.85,
    output_path: Path = FULL_OUTPUT,
    base_seed: int = 42,
    max_retries: int = 4,
    save_every: int = 20,
    retry_offset: int = 0,
    start_index: int = 1,
    replacement: bool = False,
) -> list[dict[str, str]]:
    """Generate a balanced, resumable full set with both TASK-006 prompts."""
    if (samples_per_category < 1 or max_retries < 1 or save_every < 1
            or retry_offset < 0 or not 1 <= start_index <= samples_per_category):
        raise ValueError("Sample count, retry count and save interval must be positive")
    if not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2")

    output = Path(output_path)
    if output.resolve() == DEFAULT_OUTPUT.resolve():
        raise ValueError("Full dataset output must differ from the pilot output")
    prompt_paths = {
        strategy: ROOT / "prompts" / f"{strategy}.txt"
        for strategy in ("basic", "advanced")
    }
    prompts = {name: path.read_text(encoding="utf-8").strip()
               for name, path in prompt_paths.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in prompt_paths.items()}
    system_prompt = REPLACEMENT_SYSTEM_PROMPT if replacement else FULL_SYSTEM_PROMPT
    system_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    places = (tuple(place for place in SCENARIO_PLACES if place != "northern Morocco")
              if replacement else SCENARIO_PLACES)
    plan = [item for item in _full_plan(samples_per_category, base_seed, places)
            if int(item[0].split("_")[1]) >= start_index]
    expected = {item[0]: item for item in plan}
    top_p, top_k, num_predict = 0.8, 20, 180
    word_pattern = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

    with requests.Session() as session:
        digest = _model_digest(session)
        version_response = session.get(f"{API}/api/version", timeout=10)
        version_response.raise_for_status()
        ollama_version = version_response.json()["version"]
        rows: dict[str, dict[str, str]] = {}
        texts = set()
        if output.exists() and output.stat().st_size:
            with output.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != list(FULL_FIELDS):
                    raise ValueError(f"Unexpected CSV columns in {output}")
                for row in reader:
                    sample_id = row["sample_id"]
                    if sample_id not in expected or sample_id in rows:
                        raise ValueError(f"Unexpected or duplicate sample ID: {sample_id}")
                    _, label, category, strategy, scenario = expected[sample_id]
                    if (row["label"] != str(label) or row["category"] != category
                            or row["strategy"] != strategy or row["scenario"] != scenario
                            or row["model"] != MODEL or row["model_digest"] != digest
                            or row["prompt_sha256"] != hashes[strategy]
                            or row["system_prompt_sha256"] != system_hash
                            or float(row["temperature"]) != temperature
                            or float(row["top_p"]) != top_p
                            or int(row["top_k"]) != top_k
                            or int(row["num_predict"]) != num_predict
                            or row["ollama_version"] != ollama_version
                            or not row["headline"] or not row["summary"]
                            or row["text"] != f"{row['headline']} {row['summary']}"):
                        raise ValueError(f"Existing sample has different settings: {sample_id}")
                    normalized = " ".join(row["text"].casefold().split())
                    if normalized in texts:
                        raise ValueError(f"Duplicate text in existing dataset: {sample_id}")
                    texts.add(normalized)
                    rows[sample_id] = row

        print(f"Resuming at {len(rows)}/{len(plan)} articles", flush=True)
        unsaved = 0
        failures = []
        try:
            for sample_id, label, category, strategy, scenario in plan:
                if sample_id in rows:
                    continue
                prompt = prompts[strategy].format(category=category, scenario=scenario)
                for attempt in range(max_retries):
                    index = int(sample_id.split("_")[1])
                    seed = base_seed + label * 100_000 + index * 10 + retry_offset + attempt
                    try:
                        response = session.post(
                            f"{API}/api/chat",
                            json={
                                "model": MODEL,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt},
                                ],
                                "stream": False, "think": False,
                                "options": {
                                    "seed": seed, "temperature": temperature,
                                    "top_p": top_p, "top_k": top_k,
                                    "num_predict": num_predict,
                                },
                            },
                            timeout=180,
                        )
                        response.raise_for_status()
                        headline, summary = _parse_article(
                            response.json()["message"]["content"]
                        )
                        article = f"{headline} {summary}"
                        length = len(word_pattern.findall(article))
                        if not 35 <= length <= 80:
                            raise ValueError(f"Word count outside 35-80: {length}")
                        normalized = " ".join(article.casefold().split())
                        if normalized in texts:
                            raise ValueError("Duplicate text")
                    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                        print(f"{sample_id}, attempt {attempt + 1}: {error}", file=sys.stderr)
                        if attempt + 1 < max_retries:
                            time.sleep(min(2 ** attempt, 4))
                        continue

                    row = {
                        "sample_id": sample_id, "label": str(label),
                        "category": category, "headline": headline,
                        "summary": summary, "text": article,
                        "seed": str(seed), "temperature": str(temperature),
                        "model": MODEL, "model_digest": digest,
                        "prompt_sha256": hashes[strategy],
                        "strategy": strategy, "scenario": scenario,
                        "top_p": str(top_p), "top_k": str(top_k),
                        "num_predict": str(num_predict),
                        "ollama_version": ollama_version,
                        "system_prompt_sha256": system_hash,
                    }
                    rows[sample_id] = row
                    texts.add(normalized)
                    unsaved += 1
                    if unsaved >= save_every:
                        _save_full_atomic(output, rows)
                        unsaved = 0
                        print(f"Saved {len(rows)}/{len(plan)} articles", flush=True)
                    break
                else:
                    failures.append(sample_id)
        finally:
            if unsaved:
                _save_full_atomic(output, rows)
                print(f"Saved {len(rows)}/{len(plan)} articles", flush=True)

    if failures:
        raise RuntimeError(
            f"{len(failures)} samples failed. Rerun with the same arguments to resume: "
            + ", ".join(failures)
        )
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--full", action="store_true", help="Generate the balanced full dataset")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--replacements", action="store_true")
    args = parser.parse_args()
    if args.replacements and (not args.full or args.output is None):
        parser.error("--replacements requires --full and --output")
    generation = generate_full_dataset if args.full else generate_articles
    kwargs = {"base_seed": args.seed}
    if args.samples_per_category is not None:
        kwargs["samples_per_category"] = args.samples_per_category
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.output is not None:
        kwargs["output_path"] = args.output
    if args.max_retries is not None:
        kwargs["max_retries"] = args.max_retries
    if args.full:
        kwargs["start_index"] = args.start_index
        kwargs["replacement"] = args.replacements
    if args.full:
        for round_number in range(1, 5):
            try:
                rows = generation(
                    **kwargs,
                    retry_offset=(round_number - 1) * kwargs.get("max_retries", 4),
                )
                break
            except RuntimeError as error:
                if "samples failed" not in str(error) or round_number == 4:
                    raise
                print(f"Retrying failed samples, pass {round_number + 1}/4", flush=True)
    else:
        rows = generation(**kwargs)
    print(f"Complete: {len(rows)} samples in {args.output or (FULL_OUTPUT if args.full else DEFAULT_OUTPUT)}")


if __name__ == "__main__":
    main()
