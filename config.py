"""
Central configuration for the Qwen3.5-2B reasoning RL pipeline.
All hyperparameters and paths live here — import this in every script.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"

SFT_OUTPUT_DIR    = str(OUTPUTS / "sft")
GRPO_OUTPUT_DIR   = str(OUTPUTS / "grpo")
GEN_OUTPUT_DIR    = str(OUTPUTS / "generated")
LOOP_STATE_FILE   = str(OUTPUTS / "loop_state.json")

# Local download paths (set by download_model.py / download_dataset.py)
LOCAL_MODEL_DIR   = str(ROOT / "models" / "Qwen3.5-2B")
LOCAL_DATASET_DIR = str(ROOT / "data" / "opus_reasoning")


# ---------------------------------------------------------------------------
# Model & Dataset
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3.5-2B"
    base_model_id: str = "Qwen/Qwen3.5-2B-Base"  # not used in pipeline, for reference
    dataset_id: str = "Crownelius/Opus-4.6-Reasoning-3300x"

    # QLoRA quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

@dataclass
class LoraConfig_SFT:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Qwen3.5 Gated DeltaNet hybrid — these are the attention + FFN projections.
    # Run `python config.py --verify-modules` to confirm against the loaded model.
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class LoraConfig_GRPO:
    r: int = 8           # Smaller rank for RL phase to save VRAM
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])


# ---------------------------------------------------------------------------
# SFT Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class SFTHyperparams:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8     # effective batch = 16
    num_train_epochs: int = 5
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    output_dir: str = SFT_OUTPUT_DIR
    report_to: str = "none"


# ---------------------------------------------------------------------------
# GRPO Hyperparameters
# ---------------------------------------------------------------------------
# CONSTRAINT: (per_device_train_batch_size * gradient_accumulation_steps) % num_generations == 0
# With defaults: 1 * 4 = 4; 4 % 4 == 0  ✓

@dataclass
class GRPOHyperparams:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4     # effective batch = 4
    num_generations: int = 4                 # Must divide effective batch. Default 8 OOMs on 12GB.
    max_completion_length: int = 768         # Reduce to 512 or 256 if OOM
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    beta: float = 0.0                        # KL penalty OFF — no reference model in VRAM
    loss_type: str = "grpo"                  # "grpo" | "dapo" — "dapo" is TRL default
    num_iterations: int = 1                  # PPO-style updates per generation
    epsilon: float = 0.2                     # PPO clip ratio
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.1         # Helps prevent Qwen3.5-2B thinking loops
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    output_dir: str = GRPO_OUTPUT_DIR
    report_to: str = "none"

    def validate(self):
        effective = self.per_device_train_batch_size * self.gradient_accumulation_steps
        assert effective % self.num_generations == 0, (
            f"effective_batch ({effective}) must be divisible by num_generations "
            f"({self.num_generations}). Got remainder {effective % self.num_generations}."
        )


# ---------------------------------------------------------------------------
# Self-improvement / Generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    num_return_sequences: int = 5    # Completions per problem
    max_new_tokens: int = 768
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.1
    top_k_filter: float = 0.3       # Keep top 30% by reward score


# ---------------------------------------------------------------------------
# Loop orchestration
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    num_loops: int = 3               # How many SFT → GRPO → generate cycles to run
    start_from_sft: bool = True      # If False, skip SFT in first iteration (e.g. resume)


# ---------------------------------------------------------------------------
# Aggregate config (convenience import)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora_sft: LoraConfig_SFT = field(default_factory=LoraConfig_SFT)
    lora_grpo: LoraConfig_GRPO = field(default_factory=LoraConfig_GRPO)
    sft: SFTHyperparams = field(default_factory=SFTHyperparams)
    grpo: GRPOHyperparams = field(default_factory=GRPOHyperparams)
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)


# Default instance — import and use directly
cfg = Config()


# ---------------------------------------------------------------------------
# CLI helper: verify LoRA target modules against a loaded model
# ---------------------------------------------------------------------------

def verify_lora_modules(model_id: str = None):
    """
    Print expected LoRA module names by reading config.json directly.
    Avoids loading model weights entirely — works around the Qwen3.5 nested
    text_config bug in transformers where config.vocab_size is not accessible
    at the top level of Qwen3_5Config.
    """
    import json

    local = Path(LOCAL_MODEL_DIR)
    config_path = local / "config.json"

    if not config_path.exists():
        print(f"config.json not found at {config_path}. Run: python download_model.py")
        return

    with open(config_path) as f:
        raw = json.load(f)

    text = raw.get("text_config", raw)

    print(f"Model type     : {raw.get('model_type', '?')}")
    print(f"Num layers     : {text.get('num_hidden_layers', '?')}")
    print(f"Hidden size    : {text.get('hidden_size', '?')}")
    print(f"Attention heads: {text.get('num_attention_heads', '?')}")
    layer_types = text.get("layer_types", [])
    print(f"Layer types    : {dict((t, layer_types.count(t)) for t in set(layer_types))}")
    print()

    print("Expected LoRA target modules (PEFT matches these by suffix):")
    print("  Attention : q_proj, k_proj, v_proj, o_proj")
    print("  FFN       : gate_proj, up_proj, down_proj")
    print()
    print(f"Configured SFT  target_modules : {cfg.lora_sft.target_modules}")
    print(f"Configured GRPO target_modules : {cfg.lora_grpo.target_modules}")
    print()
    print("Note: PEFT matches by suffix — short names like 'q_proj' work regardless")
    print("      of the full path (e.g. language_model.model.layers.0.self_attn.q_proj).")


def patch_qwen35_config(hf_config):
    """
    Workaround for a transformers bug where Qwen3_5Config stores model attributes
    under a nested 'text_config' sub-config, but the model's __init__ tries to
    read them directly from the top-level config (e.g. config.vocab_size).

    Call this on the AutoConfig object before passing it to from_pretrained().
    Safe no-op if the attributes are already present at the top level.
    """
    tc = getattr(hf_config, "text_config", None)
    if tc is None:
        return hf_config
    for key in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "intermediate_size", "max_position_embeddings",
                "rms_norm_eps", "hidden_act", "tie_word_embeddings", "use_cache",
                "pad_token_id", "eos_token_id", "head_dim", "layer_types",
                "full_attention_interval", "attn_output_gate", "mtp_num_hidden_layers",
                "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
                "linear_value_head_dim", "linear_conv_kernel_dim"]:
        if hasattr(tc, key) and not hasattr(hf_config, key):
            setattr(hf_config, key, getattr(tc, key))
    return hf_config


if __name__ == "__main__":
    import sys
    if "--verify-modules" in sys.argv:
        verify_lora_modules()
    else:
        from dataclasses import asdict
        import json
        print(json.dumps(asdict(cfg), indent=2, default=str))
