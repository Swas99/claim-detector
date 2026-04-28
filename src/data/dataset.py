"""Dataset loading and preprocessing for claim detection.

Loads the composite dataset from VeritaResearch/claim-extraction:
- Train: ~10,400 sentences (48% claims)
- Test: ~2,600 sentences (47% claims)
- OOD: 911 tweets from CheckThat (63% claims)
"""

import csv
import subprocess
from pathlib import Path

from datasets import Dataset, DatasetDict

from src.config import settings

REPO_API_BASE = "https://api.github.com/repos/VeritaResearch/claim-extraction/contents/data"
DATA_FILES = {
    "train": ("ours/train.csv", "train.csv", ","),
    "test": ("ours/test.csv", "test.csv", ","),
    "ood": ("CheckThat/CT22_english_1B_claim_dev_test.tsv", "checkthat_ood.tsv", "\t"),
}


def download_dataset(data_dir: Path | None = None) -> None:
    """Download dataset files from GitHub using gh CLI."""
    data_dir = data_dir or settings.data_dir

    for split, (repo_path, local_name, _) in DATA_FILES.items():
        local_path = data_dir / local_name
        if local_path.exists():
            print(f"  {split}: {local_path} already exists, skipping")
            continue

        print(f"  {split}: downloading {repo_path}...")
        result = subprocess.run(
            ["gh", "api", f"repos/VeritaResearch/claim-extraction/contents/data/{repo_path}",
             "--jq", ".download_url"],
            capture_output=True, text=True, check=True,
        )
        download_url = result.stdout.strip()
        subprocess.run(
            ["curl", "-sL", "-o", str(local_path), download_url],
            check=True,
        )
        print(f"  {split}: saved to {local_path}")


def _load_csv(path: Path, delimiter: str = ",") -> list[dict]:
    """Load a CSV/TSV file and return list of {text, label} dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # CSV format: text,label (0/1)
            # TSV format: topic, tweet_id, tweet_url, tweet_text, class_label
            if "tweet_text" in row:
                text = row["tweet_text"]
                label = int(row["class_label"])
            else:
                text = row["text"]
                label = int(row["label"])

            if text and text.strip():
                rows.append({"text": text.strip(), "label": label})
    return rows


def load_claim_dataset(data_dir: Path | None = None) -> DatasetDict:
    """Load train/test splits as a HuggingFace DatasetDict.

    Returns:
        DatasetDict with 'train' and 'test' splits.
        Each example has 'text' (str) and 'label' (int: 0=not claim, 1=claim).
    """
    data_dir = data_dir or settings.data_dir
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print("Dataset not found. Downloading...")
        download_dataset(data_dir)

    train_rows = _load_csv(train_path)
    test_rows = _load_csv(test_path)

    return DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows),
    })


def load_ood_dataset(data_dir: Path | None = None) -> Dataset:
    """Load out-of-domain CheckThat tweets dataset.

    Returns:
        Dataset with 'text' (str) and 'label' (int: 0=not claim, 1=claim).
    """
    data_dir = data_dir or settings.data_dir
    ood_path = data_dir / "checkthat_ood.tsv"

    if not ood_path.exists():
        print("OOD dataset not found. Downloading...")
        download_dataset(data_dir)

    rows = _load_csv(ood_path, delimiter="\t")
    return Dataset.from_list(rows)


def print_dataset_stats(ds: DatasetDict) -> None:
    """Print summary statistics for the dataset."""
    for split_name, split in ds.items():
        labels = split["label"]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"  {split_name}: {len(labels)} samples | "
              f"claims: {n_pos} ({n_pos/len(labels)*100:.1f}%) | "
              f"not claims: {n_neg} ({n_neg/len(labels)*100:.1f}%)")


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        print("Downloading dataset...")
        download_dataset()
        print("Done.")
    else:
        print("Loading dataset...")
        ds = load_claim_dataset()
        print_dataset_stats(ds)

        print("\nSample train examples:")
        for i in range(3):
            ex = ds["train"][i]
            label = "CLAIM" if ex["label"] == 1 else "NOT CLAIM"
            print(f'  [{label}] "{ex["text"][:80]}..."')

        print("\nLoading OOD dataset...")
        ood = load_ood_dataset()
        n_pos = sum(ood["label"])
        print(f"  OOD: {len(ood)} tweets | claims: {n_pos} ({n_pos/len(ood)*100:.1f}%)")
