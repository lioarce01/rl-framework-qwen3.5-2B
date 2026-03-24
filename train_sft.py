"""
Supervised Fine-Tuning (SFT) for Qwen3.5-2B.

Trains the model to imitate reasoning traces from the Opus-4.6 distillation
dataset using QLoRA (4-bit quantization + LoRA adapters).

Usage:
    python train_sft.py
    python train_sft.py --dataset-path ./outputs/generated/augmented.jsonl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import Config, cfg, patch_qwen35_config
from data import prepare_sft_dataset, load_raw_dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: Config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.model.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.model.use_double_quant,
    )

    from pathlib import Path
    from transformers import AutoConfig
    from config import LOCAL_MODEL_DIR
    model_source = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else config.model.model_id
    print(f"Model source: {model_source}")

    hf_config = patch_qwen35_config(
        AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        config=hf_config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Prevents issues with flash attention

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def build_lora_config(config: Config) -> LoraConfig:
    lc = config.lora_sft
    return LoraConfig(
        r=lc.r,
        lora_alpha=lc.lora_alpha,
        lora_dropout=lc.lora_dropout,
        bias=lc.bias,
        task_type=lc.task_type,
        target_modules=lc.target_modules,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(config: Config = None, dataset: Dataset = None) -> str:
    """
    Run SFT training. Returns the path to the saved model.

    Args:
        config: Config instance. Uses global cfg if None.
        dataset: Pre-formatted SFT dataset. Loads from HuggingFace if None.

    Returns:
        Path to the saved adapter/model directory.
    """
    config = config or cfg
    config.grpo.validate()  # Sanity-check sibling config too

    os.makedirs(config.sft.output_dir, exist_ok=True)

    print("=" * 60)
    print("SFT Training — Qwen3.5-2B")
    print("=" * 60)

    # Dataset
    if dataset is None:
        raw = load_raw_dataset(config.model.dataset_id)
        dataset = prepare_sft_dataset(raw)

    # Model
    print(f"\nLoading model: {config.model.model_id}")
    model, tokenizer = load_model_and_tokenizer(config)

    # LoRA
    peft_config = build_lora_config(config)

    # SFT config
    sft_params = config.sft
    sft_config = SFTConfig(
        output_dir=sft_params.output_dir,
        per_device_train_batch_size=sft_params.per_device_train_batch_size,
        gradient_accumulation_steps=sft_params.gradient_accumulation_steps,
        num_train_epochs=sft_params.num_train_epochs,
        learning_rate=sft_params.learning_rate,
        max_seq_length=sft_params.max_seq_length,
        warmup_ratio=sft_params.warmup_ratio,
        lr_scheduler_type=sft_params.lr_scheduler_type,
        gradient_checkpointing=sft_params.gradient_checkpointing,
        bf16=sft_params.bf16,
        logging_steps=sft_params.logging_steps,
        save_strategy=sft_params.save_strategy,
        report_to=sft_params.report_to,
        # Use loss on assistant tokens only (more efficient for SFT)
        dataset_text_field=None,  # Use 'messages' format
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,    # NOTE: 'tokenizer=' was removed in TRL; use 'processing_class='
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print(f"\nTraining on {len(dataset)} examples")
    print(f"Batch size: {sft_params.per_device_train_batch_size} × "
          f"{sft_params.gradient_accumulation_steps} grad accum = "
          f"{sft_params.per_device_train_batch_size * sft_params.gradient_accumulation_steps} effective")
    print(f"Epochs: {sft_params.num_train_epochs}")
    print(f"Max seq length: {sft_params.max_seq_length}")
    print()

    trainer.train()
    trainer.save_model(sft_params.output_dir)
    tokenizer.save_pretrained(sft_params.output_dir)

    print(f"\nSFT model saved to: {sft_params.output_dir}")
    return sft_params.output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training for Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a local JSONL dataset file to use instead of HuggingFace",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    # Apply CLI overrides
    if args.epochs:
        cfg.sft.num_train_epochs = args.epochs
    if args.output_dir:
        cfg.sft.output_dir = args.output_dir

    # Load local dataset if specified
    dataset = None
    if args.dataset_path:
        from datasets import load_dataset as ld
        raw = ld("json", data_files=args.dataset_path, split="train")
        from data import prepare_sft_dataset
        dataset = prepare_sft_dataset(raw)

    run(cfg, dataset)
