# Qwen3.5-2B Reasoning RL Pipeline

Fine-tunes **Qwen3.5-2B** for step-by-step reasoning using SFT + GRPO reinforcement learning on the [Opus-4.6-Reasoning-3300x](https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x) dataset. Targets **12 GB VRAM** (RTX 5070).

---

## Setup

```bash
pip install -r requirements.txt
python download_model.py    # → models/Qwen3.5-2B/
python download_dataset.py  # → data/opus_reasoning/
```

---

## Run

```bash
# 1. Smoke tests (no GPU needed)
python config.py --verify-modules
python data.py
python rewards.py

# 2. SFT (~1-3 hours)
python train_sft.py

# 3. GRPO RL (~12-36 hours)
python train_grpo.py

# 4. Full loop — SFT → GRPO → self-generate → repeat
python loop.py
```

---

## Pipeline

```
Qwen3.5-2B
  → SFT on 2,160 math/code problems with <think>...</think> traces
  → GRPO with correctness + format reward functions
  → Self-generate new traces, filter top 30%, augment dataset
  → Repeat 2-3 times
```

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters — edit here |
| `data.py` | Dataset loading and formatting |
| `rewards.py` | Reward functions for GRPO |
| `train_sft.py` | Supervised fine-tuning |
| `train_grpo.py` | GRPO reinforcement learning |
| `generate.py` | Self-improvement generation |
| `loop.py` | Full pipeline orchestration |

---

## Hardware

Runs on a consumer GPU. Uses QLoRA 4-bit quantization to keep VRAM usage low.

> See `CONCEPTS.md` for a full plain-English explanation of what this does and why.
