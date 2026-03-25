"""
Self-improvement generation script.

Loads a trained model (GRPO checkpoint), generates multiple completions per
problem, scores them using reward functions, filters the best ones, and saves
a new JSONL file ready for the next SFT round.

Usage:
    python generate.py
    python generate.py --model-path ./outputs/grpo
    python generate.py --model-path ./outputs/grpo --top-k 0.3 --num-sequences 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

from config import Config, cfg, GEN_OUTPUT_DIR, LOCAL_MODEL_DIR, patch_qwen35_config
from data import extract_thinking, extract_solution, load_raw_dataset
from rewards import score_completion


# ---------------------------------------------------------------------------
# Model loading for inference
# ---------------------------------------------------------------------------

def load_for_inference(model_path: str, config: Config):
    """Load model + tokenizer in inference mode (merged weights if possible)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_source = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else config.model.model_id

    # Try loading as merged model; fall back to direct load
    try:
        from peft import PeftModel
        hf_config = patch_qwen35_config(
            AutoConfig.from_pretrained(base_source, trust_remote_code=True)
        )
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_source,
            config=hf_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if hasattr(base, "visual"):
            base.visual.to("cpu")
            base.visual.requires_grad_(False)
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"Loaded and merged PEFT adapter from: {model_path}")
    except Exception:
        hf_config = patch_qwen35_config(
            AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if hasattr(model, "visual"):
            model.visual.to("cpu")
            model.visual.requires_grad_(False)
        print(f"Loaded model from: {model_path}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    prompt: str,
    gen_config,
) -> list[str]:
    """Generate multiple completions for a single prompt."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.max_new_tokens,
            num_return_sequences=gen_config.num_return_sequences,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            do_sample=gen_config.do_sample,
            repetition_penalty=gen_config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input tokens)
    input_len = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        generated_ids = output[input_len:]
        completions.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return completions


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_best(
    completions: list[str],
    solution: str,
    category: str,
    top_k: float,
) -> list[dict]:
    """Score all completions, keep top top_k fraction."""
    scored = [
        (score_completion(c, solution, category), c)
        for c in completions
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    keep_n = max(1, int(len(scored) * top_k))
    best = scored[:keep_n]

    return [
        {"score": score, "completion": completion}
        for score, completion in best
    ]


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run(
    config: Config = None,
    model_path: str = None,
    output_path: str = None,
) -> str:
    """
    Generate self-improvement samples and save to JSONL.

    Returns:
        Path to the saved JSONL file.
    """
    config = config or cfg
    model_path = model_path or config.grpo.output_dir
    os.makedirs(GEN_OUTPUT_DIR, exist_ok=True)

    loop_num = _get_loop_number(GEN_OUTPUT_DIR)
    output_path = output_path or str(Path(GEN_OUTPUT_DIR) / f"generated_loop_{loop_num}.jsonl")

    print("=" * 60)
    print("Self-Improvement Generation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print()

    # Load model
    model, tokenizer = load_for_inference(model_path, config)

    # Load dataset
    raw = load_raw_dataset(config.model.dataset_id)

    gen_config = config.gen
    top_k = gen_config.top_k_filter
    total_generated = 0
    total_kept = 0

    with open(output_path, "w") as out_f:
        for i, example in enumerate(raw):
            problem  = example["problem"]
            solution = example["solution"]
            category = example.get("category", "unknown")

            completions = generate_completions(model, tokenizer, problem, gen_config)
            total_generated += len(completions)

            best = filter_best(completions, solution, category, top_k)
            total_kept += len(best)

            for item in best:
                record = {
                    "problem":  problem,
                    "thinking": extract_thinking(item["completion"]),
                    "solution": extract_solution(item["completion"]),
                    "category": category,
                    "reward_score": round(item["score"], 4),
                    "source": "self_generated",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(raw)}] generated={total_generated}, kept={total_kept}")

    print(f"\nDone. Generated {total_generated}, kept {total_kept} "
          f"({100*total_kept/total_generated:.1f}%)")
    print(f"Saved to: {output_path}")
    return output_path


def _get_loop_number(directory: str) -> int:
    """Infer next loop number from existing files."""
    existing = list(Path(directory).glob("generated_loop_*.jsonl"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-improvement generation for Qwen3.5-2B")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to GRPO model checkpoint (default: outputs/grpo)",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--top-k", type=float, default=None,
        help="Fraction of completions to keep per problem (default: 0.3)",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=None,
        help="Number of completions to generate per problem (default: 5)",
    )
    args = parser.parse_args()

    if args.top_k is not None:
        cfg.gen.top_k_filter = args.top_k
    if args.num_sequences is not None:
        cfg.gen.num_return_sequences = args.num_sequences

    run(cfg, model_path=args.model_path, output_path=args.output_path)
