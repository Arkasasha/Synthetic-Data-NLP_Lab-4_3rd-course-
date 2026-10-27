"""Filter the synthetic AG News pilot and record every removal reason."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "synthetic" / "pilot_100.csv"
DEFAULT_OUTPUT = ROOT / "data" / "synthetic" / "filtered.csv"
DEFAULT_STATS = ROOT / "results" / "filtering_stats.json"
OLLAMA_API = "http://127.0.0.1:11434"
EMBED_MODEL = "all-minilm"
CLASSIFIER_MODEL = "qwen3:4b-instruct"
CATEGORIES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
REQUIRED = ("sample_id", "label", "category", "headline", "summary", "text")
MANUAL_EXCLUSIONS = {
    "1_017": {
        "reason": "Fictitious Denver-Boston Game 7 presented as Denver's first NBA title",
        "source": "https://www.nba.com/playoffs/2023/series",
    }
}
LEBRON_ACTIVE_SOURCE = "https://www.nba.com/news/lebron-james-24th-season-jersey-patch-2026-27"
CLASSIFIER_INSTRUCTION = (
    "Classify the PRIMARY news angle into exactly one AG News category. "
    "World: diplomacy, government, international events. Sports: competitions "
    "and athletes. Business: earnings, investment, markets, company strategy "
    "and economic impact, including technology companies. Sci/Tech: scientific "
    "findings and technical product capabilities when the financial angle is "
    "secondary. Return only JSON with key category and one of World, Sports, "
    "Business, Sci/Tech."
)


def words(text: str) -> list[str]:
    return WORD.findall(str(text).lower())


def normalize(text: str) -> str:
    return " ".join(str(text).casefold().split())


def distinct_2(texts: list[str]) -> float | None:
    bigrams = [pair for text in texts
               for pair in zip(words(text)[:35], words(text)[1:35])]
    return len(set(bigrams)) / len(bigrams) if bigrams else None


def quality(frame: pd.DataFrame) -> dict:
    text = frame["text"].fillna("").astype(str)
    counts = text.map(lambda value: len(words(value)))
    return {
        "rows": len(frame),
        "class_counts": {
            name: int((frame["category"] == name).sum())
            for name in CATEGORIES.values()
        },
        "missing_text": int(frame["text"].isna().sum()),
        "blank_text": int(text.str.strip().eq("").sum()),
        "exact_duplicate_text": int(text.map(normalize).duplicated().sum()),
        "mean_words": round(float(counts.mean()), 2) if len(counts) else None,
        "distinct_2_first35": distinct_2(text.tolist()),
    }


def _installed_models(session: requests.Session) -> dict[str, dict]:
    response = session.get(f"{OLLAMA_API}/api/tags", timeout=10)
    response.raise_for_status()
    return {item["name"]: item for item in response.json()["models"]}


def _embed(session: requests.Session, texts: list[str]) -> np.ndarray:
    parts = []
    for start in range(0, len(texts), 50):
        response = session.post(
            f"{OLLAMA_API}/api/embed",
            json={"model": EMBED_MODEL, "input": texts[start:start + 50]},
            timeout=180,
        )
        response.raise_for_status()
        parts.append(np.asarray(response.json()["embeddings"], dtype=np.float32))
    vectors = np.vstack(parts)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def _classify(session: requests.Session, text: str) -> str:
    for attempt in range(3):
        try:
            response = session.post(
                f"{OLLAMA_API}/api/chat",
                json={
                    "model": CLASSIFIER_MODEL,
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_INSTRUCTION},
                        {"role": "user", "content": text},
                    ],
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": 50},
                },
                timeout=120,
            )
            response.raise_for_status()
            category = json.loads(response.json()["message"]["content"])["category"]
            if category not in CATEGORIES.values():
                raise ValueError(f"Unexpected category: {category}")
            return category
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"Category check failed: {error}") from error
    raise AssertionError("unreachable")


def _manual_exclusion(row: pd.Series) -> dict | None:
    sample_id = row["sample_id"]
    if (sample_id in MANUAL_EXCLUSIONS
            and "Denver Nuggets" in row["text"]
            and "Boston Celtics" in row["text"]):
        return MANUAL_EXCLUSIONS[sample_id]
    if re.search(r"^LeBron James Announces Retirement\b", row["headline"], re.I):
        return {
            "reason": "Fictitious retirement announcement about an active NBA player",
            "source": LEBRON_ACTIVE_SOURCE,
        }
    return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=".filter_stats_", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=".filtered_", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            frame.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def filter_data(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    stats_path: Path = DEFAULT_STATS,
    min_words: int = 35,
    max_words: int = 80,
    near_duplicate_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Filter invalid, repeated, extreme-length, and suspect-label articles."""
    input_path, output_path, stats_path = map(
        Path, (input_path, output_path, stats_path)
    )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ to preserve unfiltered data")
    if not 1 <= min_words <= max_words or not 0 < near_duplicate_threshold <= 1:
        raise ValueError("Invalid word-length bounds or similarity threshold")

    original = pd.read_csv(input_path)
    if not set(REQUIRED).issubset(original.columns):
        raise ValueError(f"Input must contain: {', '.join(REQUIRED)}")
    removed = {
        key: [] for key in (
            "invalid", "length_outlier", "exact_duplicate", "manual_exclusion",
            "near_duplicate", "potential_label_mismatch",
        )
    }
    seen_ids = set()
    valid_indices = []
    for index, row in original.iterrows():
        sample_id = str(row["sample_id"])
        label = pd.to_numeric(row["label"], errors="coerce")
        fields_present = all(
            isinstance(row[field], str) and row[field].strip()
            for field in ("sample_id", "category", "headline", "summary", "text")
        )
        valid = (
            fields_present and label in CATEGORIES
            and row["category"] == CATEGORIES.get(label)
            and normalize(row["text"]) == normalize(
                f"{row['headline']} {row['summary']}"
            ) and sample_id not in seen_ids
        )
        if valid:
            seen_ids.add(sample_id)
            valid_indices.append(index)
        else:
            removed["invalid"].append(sample_id)

    current = original.loc[valid_indices].copy()
    lengths = current["text"].map(lambda value: len(words(value)))
    in_range = lengths.between(min_words, max_words)
    removed["length_outlier"] = current.loc[~in_range, "sample_id"].tolist()
    current = current.loc[in_range].copy()

    duplicate = current["text"].map(normalize).duplicated(keep="first")
    removed["exact_duplicate"] = current.loc[duplicate, "sample_id"].tolist()
    current = current.loc[~duplicate].copy()

    manual_reasons = {}
    for _, row in current.iterrows():
        reason = _manual_exclusion(row)
        if reason is not None:
            manual_reasons[row["sample_id"]] = reason
    manual = current["sample_id"].isin(manual_reasons)
    removed["manual_exclusion"] = current.loc[manual, "sample_id"].tolist()
    current = current.loc[~manual].copy().reset_index(drop=True)
    if current.empty:
        raise ValueError("No valid articles remain after basic filtering")

    with requests.Session() as session:
        models = _installed_models(session)
        embed_record = models.get(EMBED_MODEL) or models.get(f"{EMBED_MODEL}:latest")
        classifier_record = models.get(CLASSIFIER_MODEL)
        if embed_record is None or classifier_record is None:
            raise RuntimeError("Install models: ollama pull all-minilm; ollama pull qwen3:4b-instruct")
        vectors = _embed(session, current["text"].tolist())
        similarities = vectors @ vectors.T
        high_similarity_before = int((np.triu(similarities, k=1) >= near_duplicate_threshold).sum())

        kept_indices = []
        near_matches = []
        for index, row in current.iterrows():
            if kept_indices:
                candidates = similarities[index, kept_indices]
                closest_position = int(candidates.argmax())
                closest = kept_indices[closest_position]
                score = float(candidates[closest_position])
                if score >= near_duplicate_threshold:
                    removed["near_duplicate"].append(row["sample_id"])
                    near_matches.append({
                        "removed": row["sample_id"],
                        "kept": current.at[closest, "sample_id"],
                        "cosine": round(score, 4),
                    })
                    continue
            kept_indices.append(index)

        deduplicated = current.loc[kept_indices].copy()
        predicted = []
        for number, row in enumerate(deduplicated.itertuples(index=False), start=1):
            category = _classify(session, row.text)
            predicted.append(category)
            if number % 10 == 0:
                print(f"Checked categories: {number}/{len(deduplicated)}", flush=True)

    deduplicated["predicted_category"] = predicted
    mismatch = deduplicated["predicted_category"] != deduplicated["category"]
    label_candidates = deduplicated.loc[
        mismatch, ["sample_id", "category", "predicted_category"]
    ].to_dict("records")
    removed["potential_label_mismatch"] = deduplicated.loc[
        mismatch, "sample_id"
    ].tolist()
    filtered = deduplicated.loc[~mismatch, original.columns].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No articles remain after filtering")

    kept_positions = [current.index[current.sample_id == sample_id][0]
                      for sample_id in filtered.sample_id]
    after_similarities = similarities[np.ix_(kept_positions, kept_positions)]
    high_similarity_after = int(
        (np.triu(after_similarities, k=1) >= near_duplicate_threshold).sum()
    )
    baseline_frame = original.loc[valid_indices]
    rng = np.random.default_rng(42)
    baseline_scores = [
        distinct_2(baseline_frame.iloc[
            rng.choice(len(baseline_frame), size=len(filtered), replace=False)
        ]["text"].tolist())
        for _ in range(200)
    ]
    before_pairs = len(current) * (len(current) - 1) // 2
    after_pairs = len(filtered) * (len(filtered) - 1) // 2
    statistics = {
        "input": str(input_path),
        "output": str(output_path),
        "settings": {
            "min_words": min_words, "max_words": max_words,
            "near_duplicate_cosine_threshold": near_duplicate_threshold,
            "embedding_model": embed_record["name"],
            "embedding_digest": embed_record["digest"],
            "label_check_model": classifier_record["name"],
            "label_check_digest": classifier_record["digest"],
        },
        "before": quality(original),
        "after": quality(filtered),
        "removed_counts": {key: len(value) for key, value in removed.items()},
        "removed_sample_ids": removed,
        "near_duplicate_pairs_before": high_similarity_before,
        "near_duplicate_pairs_after": high_similarity_after,
        "near_duplicate_pair_rate_before": (
            high_similarity_before / before_pairs if before_pairs else 0
        ),
        "near_duplicate_pair_rate_after": (
            high_similarity_after / after_pairs if after_pairs else 0
        ),
        "matched_size_lexical_comparison": {
            "sample_size": len(filtered), "random_seed": 42, "draws": 200,
            "unfiltered_random_mean_distinct_2": float(np.mean(baseline_scores)),
            "filtered_distinct_2": distinct_2(filtered["text"].tolist()),
        },
        "near_duplicate_matches": near_matches,
        "potential_label_mismatches": label_candidates,
        "manual_exclusion_reasons": manual_reasons,
    }
    if len(original) != len(filtered) + sum(statistics["removed_counts"].values()):
        raise AssertionError("Removal accounting does not match input and output")

    _write_csv_atomic(output_path, filtered)
    _write_json_atomic(stats_path, statistics)
    return filtered, statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--min-words", type=int, default=35)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    args = parser.parse_args()
    filtered, statistics = filter_data(
        input_path=args.input, output_path=args.output, stats_path=args.stats,
        min_words=args.min_words, max_words=args.max_words,
        near_duplicate_threshold=args.similarity_threshold,
    )
    print(f"Filtered: {len(filtered)}/{statistics['before']['rows']} articles")
    print("Removed:", statistics["removed_counts"])
    print("Saved:", args.output, "and", args.stats)


if __name__ == "__main__":
    main()
