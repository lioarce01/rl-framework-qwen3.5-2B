"""
Download the Opus-4.6-Reasoning-3300x dataset from HuggingFace.

HuggingFace page: https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x

Schema (real field names):
    problem   — question/problem statement
    thinking  — step-by-step reasoning trace (from Claude Opus 4.6)
    solution  — final answer
    category  — "math" | "code"
    difficulty — "medium" | "hard"
    id, timestamp, hash

Usable rows after cleaning: ~2,160

Saves in two formats:
  - HuggingFace Arrow format (fast loading via load_from_disk)
  - JSONL (human-readable, also loadable with datasets)

Usage:
    python download_dataset.py
    python download_dataset.py --output-dir ./data/opus_reasoning
    python download_dataset.py --preview       # print first 3 rows and stats, no save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def download(
    dataset_id: str = "Crownelius/Opus-4.6-Reasoning-3300x",
    output_dir: str = None,
    token: str = None,
) -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    from config import LOCAL_DATASET_DIR
    output_dir = output_dir or LOCAL_DATASET_DIR
    os.makedirs(output_dir, exist_ok=True)

    hf_kwargs = {}
    if token:
        hf_kwargs["token"] = token
    elif os.environ.get("HF_TOKEN"):
        hf_kwargs["token"] = os.environ["HF_TOKEN"]

    print(f"Downloading {dataset_id} → {output_dir}\n")

    ds = load_dataset(dataset_id, **hf_kwargs)

    # Use 'train' split (the only split in this dataset)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    print(f"Rows downloaded: {len(split)}")
    print(f"Columns: {split.column_names}\n")

    # Save as Arrow (fast, default format for load_from_disk)
    arrow_path = str(Path(output_dir) / "arrow")
    split.save_to_disk(arrow_path)
    print(f"Saved Arrow format → {arrow_path}")

    # Save as JSONL (human-readable backup)
    jsonl_path = str(Path(output_dir) / "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in split:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved JSONL format → {jsonl_path}")

    _print_stats(split)
    return output_dir


def _print_stats(dataset):
    from collections import Counter

    print("\n--- Dataset Statistics ---")
    print(f"Total rows: {len(dataset)}")

    if "category" in dataset.column_names:
        cats = Counter(dataset["category"])
        print(f"\nCategory distribution:")
        for cat, n in cats.most_common():
            print(f"  {cat}: {n} ({100*n/len(dataset):.1f}%)")

    if "difficulty" in dataset.column_names:
        diffs = Counter(dataset["difficulty"])
        print(f"\nDifficulty distribution:")
        for diff, n in diffs.most_common():
            print(f"  {diff}: {n} ({100*n/len(dataset):.1f}%)")

    # Field length stats
    for field in ["problem", "thinking", "solution"]:
        if field in dataset.column_names:
            lengths = [len(str(row[field])) for row in dataset]
            print(f"\n{field} length (chars):")
            print(f"  min={min(lengths)}, max={max(lengths)}, "
                  f"avg={sum(lengths)/len(lengths):.0f}, "
                  f"median={sorted(lengths)[len(lengths)//2]}")


def preview(dataset_id: str = "Crownelius/Opus-4.6-Reasoning-3300x"):
    """Print first 3 rows and stats without saving."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    _print_stats(split)

    print("\n--- First 3 rows ---")
    for i in range(min(3, len(split))):
        row = split[i]
        print(f"\n[Row {i}]")
        for k, v in row.items():
            val = str(v)
            truncated = val[:200] + "..." if len(val) > 200 else val
            print(f"  {k}: {truncated}")


def load_local(output_dir: str = None):
    """
    Load the dataset from local disk (after downloading).
    Use this in training scripts instead of re-downloading each time.

    Example:
        from download_dataset import load_local
        dataset = load_local()
    """
    from datasets import load_from_disk
    from config import LOCAL_DATASET_DIR

    arrow_path = str(Path(output_dir or LOCAL_DATASET_DIR) / "arrow")
    if not Path(arrow_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {arrow_path}. Run: python download_dataset.py"
        )
    return load_from_disk(arrow_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Opus-4.6-Reasoning-3300x dataset")
    parser.add_argument(
        "--dataset-id", type=str, default="Crownelius/Opus-4.6-Reasoning-3300x",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Local directory to save the dataset (default: ./data/opus_reasoning)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print sample rows and stats without saving",
    )
    args = parser.parse_args()

    if args.preview:
        preview(args.dataset_id)
    else:
        download(
            dataset_id=args.dataset_id,
            output_dir=args.output_dir,
            token=args.token,
        )
