"""Filter synthetic AG News and finalize a balanced training dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


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
CONSISTENCY_INSTRUCTION = (
    "Review a fictional news brief for CLEAR internal inconsistencies only. "
    "Invented people, organizations and events are allowed. Flag a brief only "
    "when it places a real location in the wrong country or geography, mixes "
    "incompatible sports formats or scores, or describes an impossible "
    "technical or logical claim. Do not flag merely unusual or unverified "
    "fictional events. Return JSON with obvious_issue (boolean) and reason "
    "(short string; empty when false)."
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
        for attempt in range(3):
            try:
                response = session.post(
                    f"{OLLAMA_API}/api/embed",
                    json={"model": EMBED_MODEL, "input": texts[start:start + 50]},
                    timeout=180,
                )
                response.raise_for_status()
                parts.append(np.asarray(response.json()["embeddings"], dtype=np.float32))
                break
            except (requests.RequestException, KeyError, ValueError) as error:
                if attempt == 2:
                    raise RuntimeError(f"Embedding batch at {start} failed: {error}") from error
                time.sleep(2 ** attempt)
        if (start // 50 + 1) % 40 == 0:
            print(f"Embedded {min(start + 50, len(texts))}/{len(texts)} texts", flush=True)
    vectors = np.vstack(parts)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def _real_screening_pool(session: requests.Session) -> pd.DataFrame:
    rows = []
    for offset in range(0, 120_000, 15_000):
        response = session.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": "fancyzhx/ag_news", "config": "default",
                    "split": "train", "offset": offset, "length": 100},
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()["rows"]
        if len(page) != 100 or any(item["truncated_cells"] for item in page):
            raise RuntimeError(f"Incomplete AG News page at offset {offset}")
        rows.extend({"row_idx": item["row_idx"], **item["row"]} for item in page)
    frame = pd.DataFrame(rows)
    if len(frame) != 800 or not frame["label"].isin(CATEGORIES).all():
        raise RuntimeError("Unexpected AG News screening pool")
    return frame


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
                    "format": {
                        "type": "object",
                        "properties": {"category": {
                            "type": "string", "enum": list(CATEGORIES.values()),
                        }},
                        "required": ["category"],
                        "additionalProperties": False,
                    },
                    "options": {
                        "temperature": 0 if attempt == 0 else 0.1,
                        "seed": 42 + attempt, "num_predict": 120,
                    },
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


def _review_consistency(session: requests.Session, text: str) -> tuple[bool, str]:
    for attempt in range(3):
        try:
            response = session.post(
                f"{OLLAMA_API}/api/chat",
                json={
                    "model": CLASSIFIER_MODEL,
                    "messages": [
                        {"role": "system", "content": CONSISTENCY_INSTRUCTION},
                        {"role": "user", "content": text},
                    ],
                    "stream": False,
                    "think": False,
                    "format": {
                        "type": "object",
                        "properties": {
                            "obvious_issue": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["obvious_issue", "reason"],
                        "additionalProperties": False,
                    },
                    "options": {
                        "temperature": 0 if attempt == 0 else 0.1,
                        "seed": 100 + attempt, "num_predict": 160,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            result = json.loads(response.json()["message"]["content"])
            if not isinstance(result["obvious_issue"], bool):
                raise ValueError("obvious_issue is not boolean")
            return result["obvious_issue"], str(result["reason"])
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"Consistency review failed: {error}") from error
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


def obvious_inconsistency(text: str) -> str | None:
    """Flag narrow, reviewable geography and sports-format contradictions."""
    text = str(text)
    if re.search(r"\bTizi Ouzou\b", text, re.I):
        if re.search(r"\bMorocc(?:o|an)\b", text, re.I):
            return "Tizi Ouzou/Morocco geographical conflict"
        if re.search(r"\bdesert\b", text, re.I):
            return "Tizi Ouzou described as desert"
    if re.search(r"\bODI\b", text, re.I) and re.search(r"\bT20\b", text, re.I):
        return "ODI and T20 formats conflated in one match"
    if re.search(r"\bFukushima(?: Prefecture)?[’']s Kamaishi\b", text, re.I):
        return "Kamaishi incorrectly placed in Fukushima instead of Iwate"
    if re.search(r"\bBra[șs]ov[’']s C[ăa]lim[ăa]ne[șs]ti district\b", text, re.I):
        return "Călimănești incorrectly placed in Brașov instead of Vâlcea"
    if (re.search(r"\bEast Nusa Tenggara\b", text, re.I)
            and re.search(r"\bMaluku\b", text, re.I)
            and re.search(r"\bbus route\b", text, re.I)):
        return "Direct bus route between separate Indonesian island regions"
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
    label_check_mode: str = "all",
) -> tuple[pd.DataFrame, dict]:
    """Filter invalid, repeated, extreme-length, and suspect-label articles."""
    input_path, output_path, stats_path = map(
        Path, (input_path, output_path, stats_path)
    )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ to preserve unfiltered data")
    if not 1 <= min_words <= max_words or not 0 < near_duplicate_threshold <= 1:
        raise ValueError("Invalid word-length bounds or similarity threshold")
    if label_check_mode not in ("all", "screen"):
        raise ValueError("label_check_mode must be 'all' or 'screen'")

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

        deduplicated = current.loc[kept_indices].copy().reset_index(drop=True)
        screen_statistics = None
        if label_check_mode == "screen":
            real_pool = _real_screening_pool(session)
            real_vectors = _embed(session, real_pool["text"].tolist())
            real_train, real_test = train_test_split(
                np.arange(len(real_pool)), test_size=0.2,
                random_state=42, stratify=real_pool["label"],
            )
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            classifier.fit(real_vectors[real_train], real_pool["label"].iloc[real_train])
            heldout_f1 = f1_score(
                real_pool["label"].iloc[real_test],
                classifier.predict(real_vectors[real_test]), average="macro",
            )
            classifier.fit(real_vectors, real_pool["label"])
            probabilities = classifier.predict_proba(vectors[kept_indices])
            predicted_labels = classifier.classes_[probabilities.argmax(axis=1)]
            assigned_labels = deduplicated["label"].astype(int).to_numpy()
            assigned_probability = probabilities[
                np.arange(len(deduplicated)),
                np.searchsorted(classifier.classes_, assigned_labels),
            ]
            disagreement = predicted_labels != assigned_labels
            candidates = np.flatnonzero(disagreement)
            predicted_probability = probabilities.max(axis=1)
            screen_statistics = {
                "source": "https://huggingface.co/datasets/fancyzhx/ag_news",
                "real_train_rows": len(real_pool),
                "real_heldout_macro_f1": round(float(heldout_f1), 4),
                "disagreements_before_threshold": int(disagreement.sum()),
                "candidate_rule": "all embedding label disagreements",
                "qwen_checked_candidates": len(candidates),
                "unchecked_disagreements": 0,
            }
            print(
                f"Label screen: {len(candidates)} classifier disagreements; "
                f"real held-out macro-F1={heldout_f1:.3f}", flush=True,
            )
        else:
            candidates = np.arange(len(deduplicated))

        predicted = deduplicated["category"].tolist()
        qwen_checks = []
        unresolved_checks = []
        for number, index in enumerate(candidates, start=1):
            row = deduplicated.iloc[int(index)]
            try:
                category = _classify(session, row["text"])
            except RuntimeError as error:
                if label_check_mode != "screen":
                    raise
                category = row["category"]
                unresolved_checks.append({"sample_id": row["sample_id"], "error": str(error)})
            predicted[int(index)] = category
            if label_check_mode == "screen":
                qwen_checks.append({
                    "sample_id": row["sample_id"],
                    "assigned_category": row["category"],
                    "embedding_prediction": CATEGORIES[int(predicted_labels[int(index)])],
                    "qwen_prediction": category,
                    "predicted_probability": round(float(predicted_probability[int(index)]), 4),
                    "confidence_gap": round(float(
                        predicted_probability[int(index)]
                        - assigned_probability[int(index)]), 4),
                })
            if number % 25 == 0:
                print(f"Checked categories: {number}/{len(candidates)}", flush=True)

    deduplicated["predicted_category"] = predicted
    mismatch = deduplicated["predicted_category"] != deduplicated["category"]
    if label_check_mode == "screen":
        qwen_agrees = {
            item["sample_id"] for item in qwen_checks
            if item["qwen_prediction"] == item["embedding_prediction"]
        }
        mismatch &= deduplicated["sample_id"].isin(qwen_agrees)
    label_candidates = deduplicated.loc[
        mismatch, ["sample_id", "category", "predicted_category"]
    ].to_dict("records")
    removed["potential_label_mismatch"] = deduplicated.loc[
        mismatch, "sample_id"
    ].tolist()
    filtered = deduplicated.loc[~mismatch, original.columns].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No articles remain after filtering")

    positions_by_id = dict(zip(current["sample_id"], current.index))
    kept_positions = [positions_by_id[sample_id] for sample_id in filtered.sample_id]
    after_similarities = similarities[np.ix_(kept_positions, kept_positions)]
    high_similarity_after = int(
        (np.triu(after_similarities, k=1) >= near_duplicate_threshold).sum()
    )
    baseline_frame = original.loc[valid_indices]
    rng = np.random.default_rng(42)
    baseline_draws = 200 if len(original) < 1_000 else 30
    baseline_scores = [
        distinct_2(baseline_frame.iloc[
            rng.choice(len(baseline_frame), size=len(filtered), replace=False)
        ]["text"].tolist())
        for _ in range(baseline_draws)
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
            "label_check_mode": label_check_mode,
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
            "sample_size": len(filtered), "random_seed": 42, "draws": baseline_draws,
            "unfiltered_random_mean_distinct_2": float(np.mean(baseline_scores)),
            "filtered_distinct_2": distinct_2(filtered["text"].tolist()),
        },
        "near_duplicate_matches": near_matches,
        "potential_label_mismatches": label_candidates,
        "manual_exclusion_reasons": manual_reasons,
    }
    if label_check_mode == "screen":
        screen_statistics["unresolved_qwen_checks"] = len(unresolved_checks)
        statistics["label_screen"] = screen_statistics
        statistics["label_screen_checks"] = qwen_checks
        statistics["unresolved_label_checks"] = unresolved_checks
    if len(original) != len(filtered) + sum(statistics["removed_counts"].values()):
        raise AssertionError("Removal accounting does not match input and output")

    _write_csv_atomic(output_path, filtered)
    _write_json_atomic(stats_path, statistics)
    return filtered, statistics


def finalize_full_dataset(
    replacements_path: Path,
    expected_replacements_per_category: int = 200,
    unfiltered_path: Path = ROOT / "data" / "synthetic" / "full_unfiltered.csv",
    user_cleaned_path: Path = ROOT / "data" / "synthetic" / "full_filtered_realism_cleaned.csv",
    filtered_path: Path = ROOT / "data" / "synthetic" / "full_filtered.csv",
    final_path: Path = ROOT / "data" / "synthetic" / "final_8000.csv",
    stats_path: Path = ROOT / "results" / "full_filtering_stats.json",
    near_duplicate_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Screen replacement articles against the reviewed set and balance classes."""
    original = pd.read_csv(unfiltered_path, dtype=str, keep_default_na=False)
    reviewed = pd.read_csv(user_cleaned_path, dtype=str, keep_default_na=False)
    base = pd.read_csv(filtered_path, dtype=str, keep_default_na=False)
    replacements = pd.read_csv(replacements_path, dtype=str, keep_default_na=False)
    statistics = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    if len(original) != 8_000 or any(len(frame.columns) != len(original.columns)
                                     or list(frame.columns) != list(original.columns)
                                     for frame in (reviewed, base, replacements)):
        raise ValueError("Unexpected dataset size or CSV schema")
    if any(not frame.sample_id.is_unique for frame in (original, reviewed, base, replacements)):
        raise ValueError("Duplicate sample IDs")
    if set(replacements.sample_id) & set(original.sample_id):
        raise ValueError("Replacement IDs overlap the original dataset")
    expected_replacement_ids = {
        f"{label}_{index:04d}"
        for label in CATEGORIES
        for index in range(2001, 2001 + expected_replacements_per_category)
    }
    if set(replacements.sample_id) != expected_replacement_ids:
        raise ValueError("Replacement generation is incomplete")

    original_by_id = original.set_index("sample_id")
    for name, frame in (("reviewed", reviewed), ("filtered", base)):
        if not set(frame.sample_id).issubset(set(original.sample_id)):
            raise ValueError(f"{name} contains unknown sample IDs")
        if not original_by_id.loc[frame.sample_id].reset_index(drop=True).equals(
            frame.drop(columns="sample_id").reset_index(drop=True)
        ):
            raise ValueError(f"{name} modifies original article content")
    if not set(base.sample_id).issubset(set(reviewed.sample_id)):
        raise ValueError("Filtered set must be a subset of the user's reviewed set")

    original_ids = set(original.sample_id)
    automatic_keys = (
        "invalid", "length_outlier", "exact_duplicate", "manual_exclusion",
        "near_duplicate", "potential_label_mismatch",
    )
    automatic_removed = {
        sample_id for key in automatic_keys
        for sample_id in statistics["removed_sample_ids"][key]
    }
    automatically_kept = original_ids - automatic_removed
    if not set(reviewed.sample_id).issubset(automatically_kept):
        raise ValueError("Reviewed set includes automatically removed articles")
    manually_removed = sorted(automatically_kept - set(reviewed.sample_id))
    rule_removed = sorted(set(reviewed.sample_id) - set(base.sample_id))
    reviewed_by_id = reviewed.set_index("sample_id")
    if any(obvious_inconsistency(reviewed_by_id.at[sample_id, "text"]) is None
           for sample_id in rule_removed):
        raise ValueError("Filtered set removed an unexplained reviewed article")
    if any(obvious_inconsistency(text) is not None for text in base.text):
        raise ValueError("Filtered set still contains a known inconsistency")
    if any(len(base.loc[base.category == category]) >= 2_000
           for category in CATEGORIES.values()):
        raise ValueError("Expected a class deficit before replacement")

    removed = {key: [] for key in (
        "invalid", "inconsistency", "exact_duplicate", "near_duplicate_base",
        "near_duplicate_replacement", "potential_label_mismatch", "unresolved_label_check",
        "consistency_review", "unresolved_consistency_review",
    )}
    candidate_indices = []
    seen_texts = set(base.text.map(normalize))
    for index, row in replacements.iterrows():
        sample_id = row["sample_id"]
        label = pd.to_numeric(row["label"], errors="coerce")
        required_present = all(row[field].strip() for field in REQUIRED)
        valid = (
            required_present and label in CATEGORIES
            and row["category"] == CATEGORIES.get(label)
            and normalize(row["text"]) == normalize(f"{row['headline']} {row['summary']}")
            and 35 <= len(words(row["text"])) <= 80
            and row["model"] == CLASSIFIER_MODEL
        )
        if not valid:
            removed["invalid"].append(sample_id)
            continue
        if obvious_inconsistency(row["text"]) is not None:
            removed["inconsistency"].append(sample_id)
            continue
        normalized = normalize(row["text"])
        if normalized in seen_texts:
            removed["exact_duplicate"].append(sample_id)
            continue
        seen_texts.add(normalized)
        candidate_indices.append(index)
    candidates = replacements.loc[candidate_indices].copy().reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No valid replacement candidates")

    with requests.Session() as session:
        models = _installed_models(session)
        embedding_model = models.get(EMBED_MODEL) or models.get(f"{EMBED_MODEL}:latest")
        classifier_model = models.get(CLASSIFIER_MODEL)
        if embedding_model is None or classifier_model is None:
            raise RuntimeError("Install all-minilm and qwen3:4b-instruct in Ollama")
        vectors = _embed(session, base.text.tolist() + candidates.text.tolist())
        base_vectors = vectors[:len(base)]
        candidate_vectors = vectors[len(base):]
        similarity_to_base = candidate_vectors @ base_vectors.T
        max_base_similarity = similarity_to_base.max(axis=1)
        candidate_similarity = candidate_vectors @ candidate_vectors.T
        kept_indices = []
        for index in np.argsort(max_base_similarity, kind="stable"):
            index = int(index)
            sample_id = candidates.at[index, "sample_id"]
            if max_base_similarity[index] >= near_duplicate_threshold:
                removed["near_duplicate_base"].append(sample_id)
            elif kept_indices and np.max(candidate_similarity[index, kept_indices]) >= near_duplicate_threshold:
                removed["near_duplicate_replacement"].append(sample_id)
            else:
                kept_indices.append(index)

        pool = candidates.iloc[kept_indices].copy().reset_index(drop=True)
        pool_vectors = candidate_vectors[kept_indices]
        pool["max_base_cosine"] = max_base_similarity[kept_indices]
        real = _real_screening_pool(session)
        real_vectors = _embed(session, real.text.tolist())
        train_indices, test_indices = train_test_split(
            np.arange(len(real)), test_size=0.2, random_state=42,
            stratify=real["label"],
        )
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(real_vectors[train_indices], real.label.iloc[train_indices])
        heldout_f1 = f1_score(
            real.label.iloc[test_indices], classifier.predict(real_vectors[test_indices]),
            average="macro",
        )
        classifier.fit(real_vectors, real.label)
        predicted_labels = classifier.predict(pool_vectors)
        checks = []
        for index, row in pool.iterrows():
            predicted_label = int(predicted_labels[index])
            if predicted_label == int(row["label"]):
                continue
            try:
                qwen_category = _classify(session, row["text"])
            except RuntimeError as error:
                removed["unresolved_label_check"].append(row["sample_id"])
                checks.append({"sample_id": row["sample_id"], "error": str(error)})
                continue
            checks.append({
                "sample_id": row["sample_id"],
                "assigned": row["category"],
                "embedding_prediction": CATEGORIES[predicted_label],
                "qwen_prediction": qwen_category,
            })
            if qwen_category == CATEGORIES[predicted_label]:
                removed["potential_label_mismatch"].append(row["sample_id"])
            if len(checks) % 25 == 0:
                print(f"Checked replacement labels: {len(checks)}", flush=True)

    rejected_ids = set().union(*(set(values) for values in removed.values()))
    eligible = pool.loc[~pool.sample_id.isin(rejected_ids)].copy()
    selected = []
    consistency_checks = []
    deficits = {category: 2_000 - int((base.category == category).sum())
                for category in CATEGORIES.values()}
    with requests.Session() as session:
        for category, need in deficits.items():
            choices = eligible.loc[eligible.category == category].sort_values(
                ["max_base_cosine", "sample_id"], kind="stable"
            )
            chosen = 0
            for row in choices.itertuples(index=False):
                if chosen >= need:
                    break
                try:
                    issue, reason = _review_consistency(session, row.text)
                except RuntimeError as error:
                    removed["unresolved_consistency_review"].append(row.sample_id)
                    consistency_checks.append({
                        "sample_id": row.sample_id, "error": str(error),
                    })
                    continue
                consistency_checks.append({
                    "sample_id": row.sample_id, "obvious_issue": issue,
                    "reason": reason,
                })
                if issue:
                    removed["consistency_review"].append(row.sample_id)
                    continue
                selected.append(row.sample_id)
                chosen += 1
                if len(selected) % 50 == 0:
                    print(f"Consistency-reviewed and selected {len(selected)} replacements", flush=True)
            if chosen < need:
                raise RuntimeError(
                    f"Not enough valid {category} replacements: {chosen} < {need}. "
                    "Generate more candidates and rerun finalization."
                )
    unused_eligible = (
        len(eligible) - len(selected) - len(removed["consistency_review"])
        - len(removed["unresolved_consistency_review"])
    )
    if sum(len(values) for values in removed.values()) + len(selected) + unused_eligible != len(replacements):
        raise AssertionError("Replacement filtering counts do not reconcile")
    selected_frame = replacements.loc[replacements.sample_id.isin(selected)].copy()
    final = pd.concat([base, selected_frame], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)
    if len(final) != 8_000 or final.category.value_counts().to_dict() != {
        category: 2_000 for category in CATEGORIES.values()
    }:
        raise AssertionError("Final dataset is not balanced at 8,000 articles")
    if final.sample_id.duplicated().any() or final.text.map(normalize).duplicated().any():
        raise AssertionError("Final dataset contains duplicate IDs or text")

    previous_after = statistics.get("automatic_filter_after", statistics["after"])
    statistics["automatic_filter_after"] = previous_after
    statistics["output"] = str(filtered_path)
    statistics["after"] = quality(base)
    rng = np.random.default_rng(42)
    baseline_scores = [
        distinct_2(original.iloc[
            rng.choice(len(original), size=len(base), replace=False)
        ].text.tolist())
        for _ in range(20)
    ]
    statistics["matched_size_lexical_comparison"] = {
        "sample_size": len(base), "random_seed": 42, "draws": 20,
        "unfiltered_random_mean_distinct_2": float(np.mean(baseline_scores)),
        "filtered_distinct_2": distinct_2(base.text.tolist()),
    }
    statistics["removed_counts"]["user_realism_review"] = len(manually_removed)
    statistics["removed_counts"]["geographic_technical_review"] = len(rule_removed)
    statistics["removed_sample_ids"]["user_realism_review"] = manually_removed
    statistics["removed_sample_ids"]["geographic_technical_review"] = rule_removed
    if len(base) + sum(statistics["removed_counts"].values()) != len(original):
        raise AssertionError("Original filtering counts do not reconcile")
    statistics["inconsistency_rule_sources"] = [
        "https://www.interieur.gov.dz/images/le-nombre-de-siges--pourvoir-pour-llection-des-membres-APW-et--APC.pdf",
        "https://www.city.kamaishi.iwate.jp/docs/2010100400028/",
        "https://primaria-calimanesti.ro/monitorul-oficial-local/statutul-orasului-calimanesti/16886-a%C8%99ezare-geografic%C4%83.html",
        "https://fiskal.kemenkeu.go.id/files/berita-kajian/file/1738297418_sesa_report.pdf",
        "https://www.icc-cricket.com/about/cricket/game-formats/the-three-formats",
    ]
    statistics["inconsistency_reasons"] = {
        sample_id: obvious_inconsistency(reviewed_by_id.at[sample_id, "text"])
        for sample_id in rule_removed
    }
    statistics["finalization"] = {
        "unfiltered_original": str(unfiltered_path),
        "user_reviewed_source": str(user_cleaned_path),
        "replacement_generation": {
            "command": (
                "python src/generate.py --full --replacements --start-index 2001 "
                f"--samples-per-category {2000 + expected_replacements_per_category} "
                "--output <temporary CSV>"
            ),
            "model": replacements.model.unique().tolist(),
            "model_digest": replacements.model_digest.unique().tolist(),
            "strategies": replacements.strategy.value_counts().to_dict(),
            "system_prompt_sha256": replacements.system_prompt_sha256.unique().tolist(),
            "first_index": 2001,
            "last_index": 2000 + expected_replacements_per_category,
            "scenario_places_excluded": ["northern Morocco"],
        },
        "candidate_count": len(replacements),
        "replacement_removed_counts": {key: len(values) for key, values in removed.items()},
        "replacement_removed_sample_ids": removed,
        "replacement_label_checks": checks,
        "replacement_consistency_checks": consistency_checks,
        "real_ag_news_heldout_macro_f1": round(float(heldout_f1), 4),
        "embedding_model": embedding_model["name"],
        "embedding_digest": embedding_model["digest"],
        "label_check_model": classifier_model["name"],
        "label_check_digest": classifier_model["digest"],
        "near_duplicate_cosine_threshold": near_duplicate_threshold,
        "selection": "lowest maximum embedding cosine to original filtered set",
        "required_by_category": deficits,
        "eligible_by_category": {
            category: int((eligible.category == category).sum())
            for category in CATEGORIES.values()
        },
        "selected_sample_ids": selected,
        "selected_by_category": {
            category: int((selected_frame.category == category).sum())
            for category in CATEGORIES.values()
        },
        "unused_eligible_candidates": unused_eligible,
        "near_duplicate_pairs_in_final_at_threshold": 0,
    }
    statistics["final_output"] = str(final_path)
    statistics["final"] = quality(final)
    _write_csv_atomic(final_path, final)
    _write_json_atomic(stats_path, statistics)
    return final, statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--min-words", type=int, default=35)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--label-check-mode", choices=("all", "screen"), default="all")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--replacements-path", type=Path)
    parser.add_argument("--expected-replacements-per-category", type=int, default=200)
    args = parser.parse_args()
    if args.finalize:
        if args.replacements_path is None:
            parser.error("--finalize requires --replacements-path")
        final, statistics = finalize_full_dataset(
            replacements_path=args.replacements_path,
            expected_replacements_per_category=args.expected_replacements_per_category,
            near_duplicate_threshold=args.similarity_threshold,
        )
        print(f"Finalized: {len(final)} balanced articles")
        print("Replacement counts:", statistics["finalization"]["selected_by_category"])
        return
    filtered, statistics = filter_data(
        input_path=args.input, output_path=args.output, stats_path=args.stats,
        min_words=args.min_words, max_words=args.max_words,
        near_duplicate_threshold=args.similarity_threshold,
        label_check_mode=args.label_check_mode,
    )
    print(f"Filtered: {len(filtered)}/{statistics['before']['rows']} articles")
    print("Removed:", statistics["removed_counts"])
    print("Saved:", args.output, "and", args.stats)


if __name__ == "__main__":
    main()
