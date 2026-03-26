"""
Side-by-side inference comparison: base Qwen3.5-2B vs GRPO-trained model.

Generates a response from each model for the same problem and writes a
structured log file to outputs/comparisons/.

Usage:
    python compare.py --prompt "What is 17 * 23?"
    python compare.py --prompt "Solve: x^2 - 5x + 6 = 0"
    python compare.py                          # interactive prompt
    python compare.py --grpo-path ./outputs/grpo --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig

from config import cfg, LOCAL_MODEL_DIR, GRPO_OUTPUT_DIR, patch_qwen35_config

COMPARISONS_DIR = str(Path(__file__).parent / "outputs" / "comparisons")


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.model.use_double_quant,
    )


def load_base_model(model_source: str):
    """Load the raw base model (no adapter)."""
    hf_config = patch_qwen35_config(
        AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_source,
        config=hf_config,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if hasattr(model, "visual"):
        model.visual.to("cpu")
        model.visual.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_grpo_model(grpo_path: str, base_source: str):
    """Load GRPO-trained model. Auto-detects PEFT adapter vs full saved model."""
    hf_config = patch_qwen35_config(
        AutoConfig.from_pretrained(base_source, trust_remote_code=True)
    )

    is_peft = (Path(grpo_path) / "adapter_config.json").exists()

    if is_peft:
        from peft import PeftModel
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_source,
            config=hf_config,
            quantization_config=_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if hasattr(base, "visual"):
            base.visual.to("cpu")
            base.visual.requires_grad_(False)
        model = PeftModel.from_pretrained(base, grpo_path)
        model = model.merge_and_unload()
    else:
        # Full merged model saved directly
        grpo_hf_config = patch_qwen35_config(
            AutoConfig.from_pretrained(grpo_path, trust_remote_code=True)
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            grpo_path,
            config=grpo_hf_config,
            quantization_config=_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if hasattr(model, "visual"):
            model.visual.to("cpu")
            model.visual.requires_grad_(False)

    model.eval()

    # Tokenizer: prefer the one saved with the GRPO checkpoint, fall back to base
    try:
        tokenizer = AutoTokenizer.from_pretrained(grpo_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def free_model(model):
    """Delete model and reclaim VRAM."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(model, tokenizer, problem: str, max_new_tokens: int, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": problem}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizer versions don't support enable_thinking
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # device_map="auto" can spread layers across devices; use the embedding device
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=cfg.gen.temperature,
            top_p=cfg.gen.top_p,
            do_sample=cfg.gen.do_sample,
            repetition_penalty=cfg.gen.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _split_think(text: str) -> tuple[str, str]:
    """Return (thinking, answer). Falls back gracefully if no <think> tags."""
    import re
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # No </think> close — everything is thinking
    if re.search(r"<think>", text, re.IGNORECASE):
        return re.sub(r"<think>", "", text, flags=re.IGNORECASE).strip(), "(no answer extracted)"
    return "", text.strip()


def format_block(label: str, problem: str, raw: str) -> str:
    thinking, answer = _split_think(raw)
    lines = [
        "=" * 70,
        f"  {label}",
        "=" * 70,
        "",
        f"PROBLEM:",
        f"  {problem}",
        "",
        "THINKING:",
    ]
    if thinking:
        for line in thinking.splitlines():
            lines.append(f"  {line}")
    else:
        lines.append("  (no <think> block)")
    lines += [
        "",
        "ANSWER:",
        f"  {answer}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(problem: str, grpo_path: str, max_new_tokens: int, enable_thinking: bool = True):
    base_source = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else cfg.model.model_id

    if not Path(grpo_path).exists():
        print(f"ERROR: GRPO checkpoint not found at '{grpo_path}'")
        print("       Run `python train_grpo.py` first, or pass --grpo-path <path>.")
        sys.exit(1)

    os.makedirs(COMPARISONS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(COMPARISONS_DIR) / f"compare_{timestamp}.log"

    header = [
        "=" * 70,
        "  INFERENCE COMPARISON — Qwen3.5-2B: Base vs GRPO-trained",
        f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Base   : {base_source}",
        f"  GRPO   : {grpo_path}",
        f"  Tokens : {max_new_tokens}",
        "=" * 70,
        "",
    ]

    print("\n".join(header))

    # ── 1. Base model ────────────────────────────────────────────────────────
    print("Loading BASE model...")
    model, tokenizer = load_base_model(base_source)
    print("Generating (base)...")
    base_raw = generate(model, tokenizer, problem, max_new_tokens, enable_thinking)
    free_model(model)
    del tokenizer
    base_block = format_block("BASE MODEL  (Qwen3.5-2B, no training)", problem, base_raw)

    # ── 2. GRPO model ────────────────────────────────────────────────────────
    print("Loading GRPO model...")
    model, tokenizer = load_grpo_model(grpo_path, base_source)
    print("Generating (GRPO)...")
    grpo_raw = generate(model, tokenizer, problem, max_new_tokens, enable_thinking)
    free_model(model)
    del tokenizer
    grpo_block = format_block("GRPO MODEL  (SFT → GRPO on Opus-distilled data)", problem, grpo_raw)

    # ── 3. Output ────────────────────────────────────────────────────────────
    full_log = "\n".join(header) + base_block + "\n" + grpo_block

    print(base_block)
    print(grpo_block)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_log)

    print(f"Log saved to: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare base vs GRPO Qwen3.5-2B on a problem")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Problem to solve (if omitted, will ask interactively)",
    )
    parser.add_argument(
        "--grpo-path", type=str, default=GRPO_OUTPUT_DIR,
        help=f"Path to GRPO checkpoint (default: {GRPO_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=768,
        help="Max tokens to generate per model (default: 768)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable <think> blocks (enable_thinking=False in chat template)",
    )
    args = parser.parse_args()

    problem = args.prompt
    if not problem:
        print("Enter the problem to solve (Ctrl+D / Ctrl+Z to submit):")
        try:
            problem = sys.stdin.read().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

    if not problem:
        print("No problem provided. Exiting.")
        sys.exit(1)

    run(
        problem,
        grpo_path=args.grpo_path,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=not args.no_thinking,
    )
