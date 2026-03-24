# What Is This Project? A Plain-English Guide

This project trains a small AI language model to reason better — step by step, like a person working through a hard problem. It combines two training techniques (imitation learning + reinforcement learning) in a loop that keeps improving the model over time.

No prior AI research background needed to read this. We'll build up from scratch.

---

## The Big Picture

We start with **Qwen3.5-2B**, a pre-trained language model made by Alibaba. It already knows how to generate text, write code, and solve basic math. But we want it to be *better at reasoning* — specifically, to think through problems step by step before giving an answer, the way a person would write on scratch paper before writing the final answer.

To do that, we run this pipeline:

```
Pre-trained model
      ↓
 SFT Training        ← teach it how to reason by example
      ↓
 GRPO Training       ← reinforce behaviors that lead to correct answers
      ↓
 Self-Generation     ← let the model produce new training data from itself
      ↓
[repeat 2-3 times]
      ↓
Final model with stronger reasoning
```

---

## Core Concepts

### 1. Language Model

A language model is a program that predicts the next word (or token) given everything that came before. Modern large language models (LLMs) like GPT-4, Claude, or Qwen can answer questions, write code, and hold conversations because they were trained on massive amounts of text.

**Qwen3.5-2B** is a relatively small model (2 billion parameters). Small models are cheaper to train and run, but need more careful training to perform well on specific tasks.

### 2. Parameters and Weights

A model's "knowledge" is stored in millions (or billions) of numbers called **parameters** or **weights**. Training a model means adjusting these numbers so the model gets better at a task. With 2 billion parameters, Qwen3.5-2B is small enough to fit on a consumer GPU but large enough to be useful.

### 3. Tokens

Models don't read words — they read **tokens**, which are chunks of text (roughly 3-4 characters each on average). The sentence "Step 1: solve for x" might be split into tokens like `["Step", " 1", ":", " solve", " for", " x"]`. When we talk about a model's "context length" or "max sequence length," we mean the maximum number of tokens it can process at once.

### 4. Reasoning / Chain-of-Thought

Rather than jumping straight to an answer, a model can be trained to first "think aloud" — writing out intermediate steps. This is called **chain-of-thought reasoning**. It dramatically improves accuracy on hard problems.

Qwen3.5-2B has a built-in format for this using special tags:

```
<think>
Step 1: We need to find x.
From the equation 2x + 4 = 10:
2x = 6
Therefore x = 3.
</think>

x = 3
```

The `<think>` block is the scratchpad. The text after `</think>` is the final answer.

### 5. Fine-Tuning

A pre-trained model knows a lot about language in general, but it doesn't automatically know how to format its thinking in the `<think>` structure, or how to reason well on the specific types of problems in our dataset.

**Fine-tuning** is the process of continuing to train a pre-trained model on a smaller, specialized dataset. Instead of training from scratch (which requires months and enormous compute), fine-tuning takes a few hours and adapts the model to behave the way we want.

---

## Training Phase 1 — SFT (Supervised Fine-Tuning)

**File:** `train_sft.py`

### What it does

SFT teaches the model by example. We show it thousands of solved problems in the `<think>...</think>` format and train it to imitate that style.

Think of it like teaching a student by showing them worked examples: "Here's a problem, here's how you should think through it, here's the answer." The model learns to copy that pattern.

### The dataset

We use **Opus-4.6-Reasoning-3300x** — a collection of ~2,160 math and code problems, each with:
- `problem` — the question
- `thinking` — a detailed step-by-step reasoning trace (written by Claude Opus 4.6)
- `solution` — the final answer

The model learns to produce reasoning traces that look like these examples.

### Loss function

During training, the model makes predictions and we measure how wrong it is using a **loss function**. The optimizer then nudges the model's weights in the direction that reduces loss. After thousands of steps, the model gets better at producing text that looks like the training examples.

### QLoRA — training a big model on limited hardware

Training all 2 billion parameters at once requires a lot of GPU memory. We use two tricks to make it feasible on 12 GB of VRAM:

**Quantization (4-bit / QLoRA):** Instead of storing each number at full precision (32 or 16 bits), we compress them to 4 bits. This cuts memory usage by ~75% with only minor quality loss.

**LoRA (Low-Rank Adaptation):** Instead of updating all 2 billion parameters, we freeze the original weights and add tiny "adapter" layers alongside them. Only the adapters (a few million parameters) are trained. This is much cheaper and can be saved separately from the base model.

Together these techniques are called **QLoRA**. The result: we can fine-tune a 2B model on a single consumer GPU.

---

## Training Phase 2 — GRPO (Reinforcement Learning)

**File:** `train_grpo.py`

### The problem with SFT alone

SFT teaches the model to *imitate* examples. But imitation has limits: the model can learn the *style* of reasoning without actually learning to *get things right*. It might look like it's reasoning but produce wrong answers.

### What is Reinforcement Learning?

**Reinforcement learning (RL)** is a training paradigm where an agent learns by trial and error. Instead of showing examples, we let the model try things and reward it when it does well, penalize it when it doesn't. Over time, it learns to do more of what gets rewarded.

This is how humans learn many skills — not by copying, but by doing and getting feedback.

### GRPO — Group Relative Policy Optimization

**GRPO** is a specific RL algorithm designed for language models. Here's how it works in plain terms:

1. Give the model a problem from the dataset.
2. Have it generate multiple different answers (we use 4 per problem).
3. Score each answer using **reward functions** (see below).
4. Compare the scores within the group — answers that scored higher than average get positive reinforcement; answers that scored lower get negative reinforcement.
5. Update the model's weights to make it more likely to produce the better answers.

Compared to older RL methods (like PPO), GRPO is simpler because it doesn't need a separate "critic" model to estimate value — it just compares answers within the batch.

### Reward Functions

**File:** `rewards.py`

Reward functions are the "grade" we give each answer. We use four:

| Reward | What it measures | Range |
|--------|-----------------|-------|
| `reward_correct` | Does the final answer match the ground truth? | -1.0 to +1.0 |
| `reward_format` | Does the response use `<think>...</think>` correctly? | -0.2 to +0.3 |
| `reward_length_penalty` | Is the response too long (potential reasoning loop)? | ≤ 0.0 |
| `reward_reasoning_quality` | Does the thinking block contain structured steps? | 0.0 to +0.2 |

The model receives all four scores and learns to optimize for all of them simultaneously. The most important signal is correctness — getting the answer right.

### Why RL after SFT?

SFT + RL is a common pattern in modern AI training (it's part of what makes ChatGPT, Claude, etc. behave well):

- **SFT** gives the model a good starting point — it already knows roughly how to format reasoning.
- **RL** then pushes it to actually be *correct*, not just to *look right*.

---

## Phase 3 — Self-Improvement Loop

**Files:** `generate.py`, `loop.py`

### The idea

After RL training, the model is better at reasoning than it was before. We can use that improved model to generate *new* training data — problems it solved correctly, with the reasoning traces it actually used. We add this data to the original dataset and train again.

This is called **Bootstrapped RL Distillation**. The model is essentially teaching its future self.

### The loop in detail

Each iteration does three things:

```
Loop N:
  1. SFT  — train on [original dataset + data generated in previous loops]
  2. GRPO — RL fine-tuning on the SFT checkpoint
  3. Generate — for every problem:
       a. Generate 5 different answers
       b. Score each answer with reward functions
       c. Keep the top 30% by score
       d. Save to a new JSONL file
```

After 3 loops, the dataset has grown significantly with high-quality, self-generated reasoning traces, and the model has been trained on progressively better data each time.

### Why does this work?

The original dataset has ~2,160 examples from Claude Opus 4.6. After one loop, we might add ~3,000 more from our model's own best outputs. The key insight: the model only saves its *best* answers (filtered by reward score), so the new data is at least as good as what the model can produce on average, and likely better than its average since we're keeping the top 30%.

---

## The Dataset

**Crownelius/Opus-4.6-Reasoning-3300x** is a dataset of math and programming problems where each answer includes a detailed reasoning trace generated by Claude Opus 4.6, one of Anthropic's most capable models.

By training our small 2B model to imitate these traces and then reinforcing correct answers, we're doing a form of **knowledge distillation** — transferring reasoning ability from a large, expensive model to a small, cheap one.

The dataset has:
- ~2,160 usable examples after cleaning
- Two categories: `math` and `code`
- Two difficulty levels: `medium` and `hard`

---

## The Hardware Setup

Everything is designed to run on a **consumer GPU**, not a data center server. The QLoRA + GRPO configuration is carefully tuned to keep VRAM usage low enough for consumer hardware.

Key constraints that flow from this:
- 4-bit quantization is required (not optional)
- `num_generations=4` (higher values use too much memory)
- `max_completion_length=768` tokens (longer would exhaust VRAM)
- No KL penalty (`beta=0.0`) — this saves ~2-4 GB by not keeping a reference model copy in memory

---

## File Map

| File | What it does |
|------|-------------|
| `config.py` | All hyperparameters in one place. Import `cfg` in any script. |
| `data.py` | Loads the dataset, formats it for SFT or GRPO. |
| `rewards.py` | The four reward functions used during RL training. |
| `train_sft.py` | Runs supervised fine-tuning. ~1-3 hours. |
| `train_grpo.py` | Runs GRPO reinforcement learning. ~12-36 hours. |
| `generate.py` | Generates new training samples from a trained model. |
| `loop.py` | Orchestrates the full multi-loop pipeline with checkpointing. |
| `download_model.py` | Downloads Qwen3.5-2B weights from HuggingFace. |
| `download_dataset.py` | Downloads the reasoning dataset from HuggingFace. |

---

## Glossary

| Term | Plain English |
|------|--------------|
| **LLM** | Large Language Model — an AI that generates text |
| **Parameter / weight** | A number inside the model that stores learned knowledge |
| **Token** | A small chunk of text (word piece) that models process |
| **Fine-tuning** | Continuing training on a smaller, specialized dataset |
| **SFT** | Supervised Fine-Tuning — learning by imitation |
| **RL** | Reinforcement Learning — learning by trial, error, and reward |
| **GRPO** | The specific RL algorithm used here |
| **Reward function** | A function that scores the model's output |
| **LoRA** | Adds small trainable layers without changing the original weights |
| **QLoRA** | LoRA + 4-bit quantization for memory efficiency |
| **Quantization** | Compressing numbers to use less memory |
| **VRAM** | GPU memory — the scarce resource in training |
| **Chain-of-thought** | Making the model write out reasoning steps before answering |
| **Distillation** | Teaching a small model to behave like a larger one |
| **`<think>` tags** | Qwen3.5's built-in format to separate reasoning from the answer |
