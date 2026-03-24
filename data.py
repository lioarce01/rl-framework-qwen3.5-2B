"""
Dataset loading and formatting for the Qwen3.5-2B RL pipeline.

Dataset: Crownelius/Opus-4.6-Reasoning-3300x
Real field names: problem, thinking, solution  (NOT question/reasoning/answer)
Usable rows after cleaning: ~2,160
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from config import cfg


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_raw_dataset(dataset_id: str = None) -> Dataset:
    """
    Load and return the raw dataset (train split).

    Prefers local disk cache (downloaded via download_dataset.py) over
    streaming from HuggingFace, which avoids re-downloading each run.
    """
    from pathlib import Path
    from config import LOCAL_DATASET_DIR

    arrow_path = Path(LOCAL_DATASET_DIR) / "arrow"
    if arrow_path.exists():
        from datasets import load_from_disk
        split = load_from_disk(str(arrow_path))
        print(f"Loaded {len(split)} rows from local cache ({arrow_path})")
    else:
        did = dataset_id or cfg.model.dataset_id
        ds = load_dataset(did)
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        print(f"Loaded {len(split)} rows from HuggingFace ({did})")

    print(f"Columns: {split.column_names}")
    return split


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------

def format_for_sft(example: dict) -> dict:
    """
    Convert a raw dataset row into a chat-formatted example for SFT.

    Wraps the thinking trace in <think>...</think> tags, which is the native
    format Qwen3.5 uses to separate reasoning from the final answer.
    """
    thinking = example["thinking"].strip()
    solution = example["solution"].strip()

    assistant_content = f"<think>\n{thinking}\n</think>\n\n{solution}"

    return {
        "messages": [
            {"role": "user", "content": example["problem"].strip()},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def prepare_sft_dataset(raw_dataset: Dataset = None) -> Dataset:
    """Load + format dataset for SFT training."""
    ds = raw_dataset or load_raw_dataset()
    formatted = ds.map(
        format_for_sft,
        remove_columns=ds.column_names,
        desc="Formatting for SFT",
    )
    print(f"SFT dataset: {len(formatted)} examples")
    return formatted


# ---------------------------------------------------------------------------
# GRPO formatting
# ---------------------------------------------------------------------------

def format_for_grpo(example: dict) -> dict:
    """
    Convert a raw dataset row into a GRPO-compatible format.

    GRPOTrainer expects:
    - 'prompt': list of message dicts (the input the model will respond to)
    - Any other fields are passed as kwargs to reward functions.

    We keep 'solution' and 'category' so reward functions can access them.
    """
    return {
        "prompt": [
            {"role": "user", "content": example["problem"].strip()}
        ],
        "solution": example["solution"].strip(),
        "category": example.get("category", "unknown"),
    }


def prepare_grpo_dataset(raw_dataset: Dataset = None) -> Dataset:
    """Load + format dataset for GRPO training."""
    ds = raw_dataset or load_raw_dataset()
    formatted = ds.map(
        format_for_grpo,
        remove_columns=ds.column_names,
        desc="Formatting for GRPO",
    )
    print(f"GRPO dataset: {len(formatted)} examples")
    return formatted


# ---------------------------------------------------------------------------
# Parsing model completions
# ---------------------------------------------------------------------------

def extract_thinking(text: str) -> str:
    """Extract content inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_solution(text: str) -> str:
    """Extract the final answer (content after </think> tag, or full text if no tags)."""
    parts = text.split("</think>")
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Dataset augmentation (self-improvement loop)
# ---------------------------------------------------------------------------

def load_generated_samples(gen_dir: str = None) -> list[dict]:
    """Load all JSONL files from the generated output directory."""
    directory = Path(gen_dir or cfg.sft.output_dir).parent / "generated"
    files = list(directory.glob("*.jsonl"))
    if not files:
        return []

    samples = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    print(f"Loaded {len(samples)} generated samples from {len(files)} files")
    return samples


def augment_dataset(
    original: Dataset,
    generated_samples: list[dict],
    for_grpo: bool = False,
) -> Dataset:
    """
    Merge original dataset with self-generated samples.

    generated_samples must have: problem, thinking, solution
    """
    if not generated_samples:
        return original

    gen_ds = Dataset.from_list(generated_samples)
    combined = original.concatenate(gen_ds) if hasattr(original, "concatenate") else original

    # Re-apply formatting
    fmt_fn = format_for_grpo if for_grpo else format_for_sft
    keep_cols = [c for c in gen_ds.column_names if c not in original.column_names]
    combined = combined.map(
        fmt_fn,
        remove_columns=[c for c in combined.column_names if c not in ("problem", "thinking", "solution", "category")],
        desc="Formatting augmented dataset",
    )
    print(f"Augmented dataset: {len(combined)} examples "
          f"({len(original)} original + {len(generated_samples)} generated)")
    return combined


# ---------------------------------------------------------------------------
# CLI: print dataset stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = load_raw_dataset()

    print("\n--- Sample ---")
    row = ds[0]
    for k, v in row.items():
        val = str(v)[:120] + "..." if len(str(v)) > 120 else str(v)
        print(f"  {k}: {val}")

    print("\n--- Category distribution ---")
    from collections import Counter
    cats = Counter(ds["category"])
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    print("\n--- Difficulty distribution ---")
    diffs = Counter(ds["difficulty"])
    for diff, count in diffs.most_common():
        print(f"  {diff}: {count}")
