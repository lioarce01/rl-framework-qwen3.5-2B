"""
Download Qwen3.5-2B weights from HuggingFace to a local directory.

HuggingFace page: https://huggingface.co/Qwen/Qwen3.5-2B

Downloads safetensors weights, tokenizer, and config files.
Resumes automatically if interrupted (snapshot_download is idempotent).

Usage:
    python download_model.py
    python download_model.py --token hf_xxxx   # if you have a HF token
    python download_model.py --output-dir ./models/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download(
    model_id: str = "Qwen/Qwen3.5-2B",
    output_dir: str = None,
    token: str = None,
):
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    from config import LOCAL_MODEL_DIR
    output_dir = output_dir or LOCAL_MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    if token:
        login(token=token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    print(f"Downloading {model_id} → {output_dir}")
    print("(This is ~4GB in BF16 safetensors. Resumes if interrupted.)\n")

    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        ignore_patterns=["*.pt", "*.bin", "original/*"],  # prefer safetensors, skip legacy formats
    )

    print(f"\nDownload complete: {local_path}")
    _print_contents(local_path)
    return local_path


def _print_contents(directory: str):
    files = sorted(Path(directory).rglob("*"))
    total_bytes = 0
    print("\nFiles downloaded:")
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            total_bytes += size
            print(f"  {f.relative_to(directory)}  ({size / 1e6:.1f} MB)")
    print(f"\nTotal: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen3.5-2B weights")
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen3.5-2B",
        help="HuggingFace model ID (default: Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Local directory to save weights (default: ./models/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token (or set HF_TOKEN env var). Not required for Qwen3.5-2B.",
    )
    args = parser.parse_args()

    download(
        model_id=args.model_id,
        output_dir=args.output_dir,
        token=args.token,
    )
