"""Create reproducible, balanced AG News training and validation splits."""

import random
from pathlib import Path

from datasets import DatasetDict, load_dataset


SEED = 42
TRAIN_PER_CLASS = 2_000
VALIDATION_PER_CLASS = 500
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "ag_news"


def main() -> None:
    source = load_dataset("fancyzhx/ag_news")
    labels = source["train"].features["label"].names
    rng = random.Random(SEED)
    train_indices = []
    validation_indices = []

    for label_id, label_name in enumerate(labels):
        indices = [
            index
            for index, label in enumerate(source["train"]["label"])
            if label == label_id
        ]
        if len(indices) < TRAIN_PER_CLASS + VALIDATION_PER_CLASS:
            raise ValueError(f"Too few samples for class {label_name}: {len(indices)}")
        rng.shuffle(indices)
        train_indices.extend(indices[:TRAIN_PER_CLASS])
        validation_indices.extend(
            indices[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VALIDATION_PER_CLASS]
        )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    splits = DatasetDict(
        {
            "train": source["train"].select(train_indices),
            "validation": source["train"].select(validation_indices),
            "test": source["test"],
        }
    )

    expected_sizes = {"train": 8_000, "validation": 2_000, "test": 7_600}
    expected_per_class = {"train": 2_000, "validation": 500, "test": 1_900}
    for name, dataset in splits.items():
        if len(dataset) != expected_sizes[name]:
            raise ValueError(f"Unexpected {name} size: {len(dataset)}")
        counts = [dataset["label"].count(label_id) for label_id in range(len(labels))]
        if any(count != expected_per_class[name] for count in counts):
            raise ValueError(f"Unexpected {name} class counts: {counts}")
        print(f"{name}: {len(dataset)} samples; class counts: {dict(zip(labels, counts))}")

    texts = {name: set(dataset["text"]) for name, dataset in splits.items()}
    names = list(splits)
    for i, left in enumerate(names):
        if len(texts[left]) != len(splits[left]):
            raise ValueError(f"Duplicate text within {left}")
        for right in names[i + 1 :]:
            if texts[left] & texts[right]:
                raise ValueError(f"Overlapping text between {left} and {right}")
    print("No duplicate or overlapping text across splits.")

    splits.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
