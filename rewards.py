"""
Reward functions for GRPO training.

All functions follow the GRPOTrainer signature:
    fn(completions: list[str], **kwargs) -> list[float]

The `**kwargs` receives all non-'prompt' fields from the GRPO dataset as
parallel lists (e.g. kwargs["solution"] is a list[str] of ground truths).

These functions are also used by generate.py to score and filter
self-generated completions during the self-improvement loop.
"""

from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_OPEN  = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_BLANK_LINE  = re.compile(r"\n\s*\n")


def _extract_final_answer(text: str) -> str:
    """Return the text after </think>, stripped. Falls back to full text."""
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def _normalize(text: str) -> str:
    """Basic normalization for answer comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\d\s\.\+\-\*\/\=\(\)\[\]\{\}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _numeric_match(pred: str, gold: str, tol: float = 1e-4) -> bool:
    """Try to compare as floats; return False if conversion fails."""
    try:
        return math.isclose(float(pred), float(gold), rel_tol=tol, abs_tol=tol)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Reward: correctness
# ---------------------------------------------------------------------------

def reward_correct(completions: list[str], **kwargs) -> list[float]:
    """
    Primary reward: +1.0 if model's final answer matches ground truth, -1.0 otherwise.

    Tries exact string match first, then numeric match for math problems.
    'solution' field from the dataset is passed via kwargs.
    """
    solutions: list[str] = kwargs.get("solution", [""] * len(completions))
    rewards = []

    for completion, gold in zip(completions, solutions):
        pred = _extract_final_answer(completion)
        gold = str(gold).strip()

        exact = _normalize(pred) == _normalize(gold)
        numeric = _numeric_match(pred, gold)

        rewards.append(1.0 if (exact or numeric) else -1.0)

    return rewards


# ---------------------------------------------------------------------------
# Reward: format compliance
# ---------------------------------------------------------------------------

def reward_format(completions: list[str], **kwargs) -> list[float]:
    """
    Format reward: checks for proper <think>...</think> structure.

    +0.3  both <think> and </think> present, and content after </think>
    +0.1  only </think> present (partial)
     0.0  neither tag present
    -0.2  <think> present but no </think> (open loop — bad for Qwen3.5-2B)
    """
    rewards = []
    for text in completions:
        has_open  = bool(_THINK_OPEN.search(text))
        has_close = bool(_THINK_CLOSE.search(text))
        after_close = _extract_final_answer(text) if has_close else ""

        if has_open and has_close and after_close:
            rewards.append(0.3)
        elif has_close and after_close:
            rewards.append(0.1)
        elif not has_open and not has_close:
            rewards.append(0.0)
        else:
            # Open <think> without closing — thinking loop risk
            rewards.append(-0.2)

    return rewards


# ---------------------------------------------------------------------------
# Reward: length penalty
# ---------------------------------------------------------------------------

def reward_length_penalty(completions: list[str], **kwargs) -> list[float]:
    """
    Soft penalty for excessive length.

    No penalty for completions under 4,000 chars (~1,000 tokens).
    Linear penalty above that threshold. Helps prevent thinking loops
    which are a documented issue with Qwen3.5-2B.
    """
    threshold = 4000
    scale = 0.0002  # penalty per char above threshold
    rewards = []
    for text in completions:
        excess = max(0, len(text) - threshold)
        rewards.append(-scale * excess)
    return rewards


# ---------------------------------------------------------------------------
# Reward: reasoning quality (lightweight heuristic)
# ---------------------------------------------------------------------------

def reward_reasoning_quality(completions: list[str], **kwargs) -> list[float]:
    """
    Heuristic reward for reasoning quality inside <think> block.

    Checks for structural indicators: numbered steps, "therefore", "because",
    equation-like content, etc. Not a substitute for a trained reward model,
    but provides a weak signal for step-by-step reasoning structure.
    """
    rewards = []
    for text in completions:
        # Extract thinking content only
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        thinking = match.group(1) if match else ""

        if not thinking:
            rewards.append(0.0)
            continue

        score = 0.0
        # Numbered steps
        if re.search(r"(step\s*\d+|^\s*\d+[\.\)]\s)", thinking, re.MULTILINE | re.IGNORECASE):
            score += 0.05
        # Logical connectors
        if re.search(r"\b(therefore|because|since|thus|hence|so we|this means)\b", thinking, re.IGNORECASE):
            score += 0.05
        # Math equations
        if re.search(r"[\=\+\-\*\/]\d", thinking):
            score += 0.05
        # Multiple paragraphs (shows structured thought)
        if len(_BLANK_LINE.findall(thinking)) >= 2:
            score += 0.05

        rewards.append(round(score, 4))

    return rewards


# ---------------------------------------------------------------------------
# Composite scorer (used by generate.py for filtering)
# ---------------------------------------------------------------------------

REWARD_WEIGHTS = {
    "correct":           1.0,
    "format":            0.3,
    "length_penalty":    1.0,
    "reasoning_quality": 0.5,
}


def score_completion(
    completion: str,
    solution: str,
    category: str = "unknown",
) -> float:
    """
    Compute a scalar score for a single completion.
    Used by the self-improvement loop to rank and filter candidates.
    """
    kwargs = {"solution": [solution], "category": [category]}
    comps  = [completion]

    score  = 0.0
    score += REWARD_WEIGHTS["correct"]           * reward_correct(comps, **kwargs)[0]
    score += REWARD_WEIGHTS["format"]            * reward_format(comps, **kwargs)[0]
    score += REWARD_WEIGHTS["length_penalty"]    * reward_length_penalty(comps, **kwargs)[0]
    score += REWARD_WEIGHTS["reasoning_quality"] * reward_reasoning_quality(comps, **kwargs)[0]
    return score


# Default reward list for GRPOTrainer
REWARD_FUNCS = [reward_correct, reward_format, reward_length_penalty, reward_reasoning_quality]


# ---------------------------------------------------------------------------
# CLI: test rewards on a dummy example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_completion = (
        "<think>\nStep 1: We need to find x.\n"
        "Step 2: From the equation 2x + 4 = 10, therefore x = 3.\n"
        "</think>\n\nx = 3"
    )
    sol = "x = 3"

    print("Test completion:")
    print(test_completion)
    print(f"\nreward_correct:           {reward_correct([test_completion], solution=[sol])}")
    print(f"reward_format:            {reward_format([test_completion])}")
    print(f"reward_length_penalty:    {reward_length_penalty([test_completion])}")
    print(f"reward_reasoning_quality: {reward_reasoning_quality([test_completion])}")
    print(f"\nComposite score: {score_completion(test_completion, sol):.4f}")
