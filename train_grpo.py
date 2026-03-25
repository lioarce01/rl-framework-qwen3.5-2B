"""
GRPO (Group Relative Policy Optimization) RL training for Qwen3.5-2B.

Loads the SFT checkpoint and further trains via RL using reward functions
defined in rewards.py. All GRPOTrainer hyperparameters go in GRPOConfig —
NOT in the GRPOTrainer constructor.

VRAM constraint notes (12GB):
- num_generations=4  (default 8 will OOM or crash with effective_batch=4)
- beta=0.0           (KL disabled — no reference model copy in VRAM)
- max_completion_length=512  (reduce to 256 if OOM)
- effective_batch = per_device_batch * grad_accum = 1 * 4 = 4
- 4 % 4 == 0  ✓  (required constraint)

Usage:
    python train_grpo.py
    python train_grpo.py --from-base   # skip SFT checkpoint, use base model
    python train_grpo.py --model-path ./outputs/sft
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from config import Config, cfg, patch_qwen35_config
from data import load_raw_dataset, prepare_grpo_dataset
from rewards import REWARD_FUNCS


def _offload_vision(model) -> None:
    """Move the vision encoder to CPU — text-only training never uses it."""
    if hasattr(model, "visual"):
        model.visual.to("cpu")
        model.visual.requires_grad_(False)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_grpo(
    model_path: str,
    config: Config,
    is_peft_checkpoint: bool = True,
):
    """
    Load model for GRPO training.

    Args:
        model_path: Path to SFT adapter checkpoint, or base model ID.
        config: Config instance.
        is_peft_checkpoint: If True, loads as PEFT adapter on top of base model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.model.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.model.use_double_quant,
    )

    from pathlib import Path
    from transformers import AutoConfig
    from config import LOCAL_MODEL_DIR
    base_source = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else config.model.model_id

    if is_peft_checkpoint:
        # Load base + SFT adapter
        hf_config = patch_qwen35_config(
            AutoConfig.from_pretrained(base_source, trust_remote_code=True)
        )
        base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_source,
            config=hf_config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        base_model.config.use_cache = False
        _offload_vision(base_model)
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
        print(f"Loaded SFT adapter from: {model_path}")
    else:
        # Start GRPO from scratch (base model or model_id directly)
        hf_config = patch_qwen35_config(
            AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=hf_config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        _offload_vision(model)
        print(f"Loaded base model from: {model_path}")

    return model


def load_tokenizer(model_path: str, config: Config):
    # Try loading tokenizer from checkpoint first; fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-pad for generation
    return tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(
    config: Config = None,
    model_path: str = None,
    dataset: Dataset = None,
    is_peft_checkpoint: bool = True,
) -> str:
    """
    Run GRPO training. Returns the path to the saved model.

    Args:
        config: Config instance. Uses global cfg if None.
        model_path: Path to SFT checkpoint. Uses config.sft.output_dir if None.
        dataset: Pre-formatted GRPO dataset. Loads from HuggingFace if None.
        is_peft_checkpoint: Whether model_path is a PEFT adapter directory.

    Returns:
        Path to the saved adapter/model directory.
    """
    config = config or cfg
    config.grpo.validate()  # Enforce batch divisibility constraint

    model_path = model_path or config.sft.output_dir
    os.makedirs(config.grpo.output_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO RL Training — Qwen3.5-2B")
    print("=" * 60)

    # Dataset
    if dataset is None:
        raw = load_raw_dataset(config.model.dataset_id)
        dataset = prepare_grpo_dataset(raw)

    # Model + tokenizer
    print(f"\nLoading model from: {model_path}")
    model = load_model_for_grpo(model_path, config, is_peft_checkpoint)
    tokenizer = load_tokenizer(model_path, config)

    # LoRA config for the new GRPO adapters
    grpo_params = config.grpo
    lc = config.lora_grpo
    peft_config = LoraConfig(
        r=lc.r,
        lora_alpha=lc.lora_alpha,
        lora_dropout=lc.lora_dropout,
        bias=lc.bias,
        task_type=lc.task_type,
        target_modules=lc.target_modules,
    )

    # GRPOConfig — ALL hyperparams go HERE, not in GRPOTrainer()
    effective_batch = grpo_params.per_device_train_batch_size * grpo_params.gradient_accumulation_steps
    print(f"\nGRPO config:")
    print(f"  effective_batch={effective_batch}, num_generations={grpo_params.num_generations} "
          f"({'✓' if effective_batch % grpo_params.num_generations == 0 else '✗ WILL CRASH'})")
    print(f"  max_completion_length={grpo_params.max_completion_length}")
    print(f"  beta={grpo_params.beta} ({'no ref model' if grpo_params.beta == 0.0 else 'KL active'})")
    print(f"  loss_type={grpo_params.loss_type}")
    print()

    grpo_config = GRPOConfig(
        output_dir=grpo_params.output_dir,
        per_device_train_batch_size=grpo_params.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_params.gradient_accumulation_steps,
        num_generations=grpo_params.num_generations,
        max_completion_length=grpo_params.max_completion_length,
        learning_rate=grpo_params.learning_rate,
        num_train_epochs=grpo_params.num_train_epochs,
        beta=grpo_params.beta,
        loss_type=grpo_params.loss_type,
        num_iterations=grpo_params.num_iterations,
        epsilon=grpo_params.epsilon,
        temperature=grpo_params.temperature,
        top_p=grpo_params.top_p,
        repetition_penalty=grpo_params.repetition_penalty,
        gradient_checkpointing=grpo_params.gradient_checkpointing,
        bf16=grpo_params.bf16,
        logging_steps=grpo_params.logging_steps,
        save_strategy=grpo_params.save_strategy,
        report_to=grpo_params.report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,             # list of callables from rewards.py
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"Training on {len(dataset)} examples\n")
    trainer.train()
    trainer.save_model(grpo_params.output_dir)
    tokenizer.save_pretrained(grpo_params.output_dir)

    print(f"\nGRPO model saved to: {grpo_params.output_dir}")
    return grpo_params.output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO RL training for Qwen3.5-2B")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to SFT checkpoint (default: outputs/sft)",
    )
    parser.add_argument(
        "--from-base", action="store_true",
        help="Start GRPO from the base model instead of SFT checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=None,
        help="Override max completion length (reduce to 256 if OOM)",
    )
    parser.add_argument(
        "--num-generations", type=int, default=None,
        help="Override num_generations (must divide effective batch size)",
    )
    args = parser.parse_args()

    if args.epochs:
        cfg.grpo.num_train_epochs = args.epochs
    if args.max_completion_length:
        cfg.grpo.max_completion_length = args.max_completion_length
    if args.num_generations:
        cfg.grpo.num_generations = args.num_generations

    model_path = args.model_path or (cfg.model.model_id if args.from_base else cfg.sft.output_dir)
    is_peft = not args.from_base and args.model_path != cfg.model.model_id

    run(cfg, model_path=model_path, is_peft_checkpoint=is_peft)
