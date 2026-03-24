"""
Orchestrates the full SFT → GRPO → Self-improvement loop.

Each iteration:
  1. SFT  — train on current dataset (original + previously generated)
  2. GRPO — RL fine-tuning on SFT checkpoint
  3. Generate — produce new reasoning traces from GRPO model
  4. Filter + augment dataset for next iteration

Loop state is saved to outputs/loop_state.json so interrupted runs can resume.

Usage:
    python loop.py              # Run all loops from scratch
    python loop.py --resume     # Resume from last checkpoint
    python loop.py --loops 2    # Run only 2 loops
    python loop.py --skip-sft   # Start at GRPO (e.g., SFT already done)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import Config, cfg, LOOP_STATE_FILE, SFT_OUTPUT_DIR, GRPO_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Loop state (checkpoint/resume)
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_loops": 0, "current_stage": None, "generated_files": []}


def save_state(path: str, state: dict):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Per-loop output directories
# ---------------------------------------------------------------------------

def loop_sft_dir(loop_num: int) -> str:
    return str(Path(SFT_OUTPUT_DIR).parent / f"sft_loop_{loop_num}")

def loop_grpo_dir(loop_num: int) -> str:
    return str(Path(GRPO_OUTPUT_DIR).parent / f"grpo_loop_{loop_num}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(config: Config = None, resume: bool = False, num_loops: int = None):
    config = config or cfg
    num_loops = num_loops or config.loop.num_loops

    os.makedirs("outputs", exist_ok=True)
    state = load_state(LOOP_STATE_FILE) if resume else {
        "completed_loops": 0,
        "current_stage": None,
        "generated_files": [],
        "sft_checkpoints": [],
        "grpo_checkpoints": [],
    }

    print("=" * 60)
    print(f"Bootstrapped RL Distillation — {num_loops} loop(s)")
    print(f"Model: {config.model.model_id}")
    print(f"Dataset: {config.model.dataset_id}")
    print("=" * 60)

    # Import here to avoid loading torch at module level
    from data import load_raw_dataset, prepare_sft_dataset, prepare_grpo_dataset, load_generated_samples
    from datasets import concatenate_datasets, Dataset
    import train_sft
    import train_grpo
    import generate as gen_module

    start_loop = state["completed_loops"] + 1
    raw_dataset = load_raw_dataset(config.model.dataset_id)

    for loop_num in range(start_loop, num_loops + 1):
        print(f"\n{'='*60}")
        print(f"LOOP {loop_num} / {num_loops}")
        print(f"{'='*60}")

        # Set loop-specific output dirs
        config.sft.output_dir  = loop_sft_dir(loop_num)
        config.grpo.output_dir = loop_grpo_dir(loop_num)

        # ----------------------------------------------------------------
        # Build augmented dataset (original + all previously generated)
        # ----------------------------------------------------------------
        generated_files = state.get("generated_files", [])
        all_generated = []
        for gen_file in generated_files:
            if Path(gen_file).exists():
                with open(gen_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_generated.append(json.loads(line))

        if all_generated:
            gen_ds = Dataset.from_list(all_generated)
            # Keep only fields present in the original dataset
            common_cols = [c for c in gen_ds.column_names if c in raw_dataset.column_names]
            gen_ds_filtered = gen_ds.select_columns(common_cols)
            current_raw = concatenate_datasets([raw_dataset, gen_ds_filtered])
            print(f"Augmented dataset: {len(raw_dataset)} original + {len(all_generated)} generated = {len(current_raw)} total")
        else:
            current_raw = raw_dataset
            print(f"Using original dataset: {len(current_raw)} rows")

        # ----------------------------------------------------------------
        # Stage 1: SFT
        # ----------------------------------------------------------------
        stage_key = f"loop_{loop_num}_sft"
        if state.get("current_stage") == stage_key and resume:
            print(f"\nSkipping SFT (already completed in previous run)")
            sft_path = config.sft.output_dir
        elif loop_num == 1 and not config.loop.start_from_sft and resume:
            print(f"\nSkipping SFT for loop 1 (--skip-sft)")
            sft_path = config.sft.output_dir
        else:
            state["current_stage"] = stage_key
            save_state(LOOP_STATE_FILE, state)
            print(f"\n--- Stage 1: SFT (loop {loop_num}) ---")
            sft_dataset = prepare_sft_dataset(current_raw)
            sft_path = train_sft.run(config, dataset=sft_dataset)
            state.setdefault("sft_checkpoints", []).append(sft_path)
            save_state(LOOP_STATE_FILE, state)

        # ----------------------------------------------------------------
        # Stage 2: GRPO
        # ----------------------------------------------------------------
        stage_key = f"loop_{loop_num}_grpo"
        state["current_stage"] = stage_key
        save_state(LOOP_STATE_FILE, state)
        print(f"\n--- Stage 2: GRPO (loop {loop_num}) ---")
        grpo_dataset = prepare_grpo_dataset(current_raw)
        grpo_path = train_grpo.run(
            config,
            model_path=sft_path,
            dataset=grpo_dataset,
            is_peft_checkpoint=True,
        )
        state.setdefault("grpo_checkpoints", []).append(grpo_path)
        save_state(LOOP_STATE_FILE, state)

        # ----------------------------------------------------------------
        # Stage 3: Self-improvement generation
        # ----------------------------------------------------------------
        stage_key = f"loop_{loop_num}_generate"
        state["current_stage"] = stage_key
        save_state(LOOP_STATE_FILE, state)
        print(f"\n--- Stage 3: Self-improvement generation (loop {loop_num}) ---")
        gen_path = gen_module.run(config, model_path=grpo_path)
        state["generated_files"].append(gen_path)

        # ----------------------------------------------------------------
        # Mark loop complete
        # ----------------------------------------------------------------
        state["completed_loops"] = loop_num
        state["current_stage"] = None
        save_state(LOOP_STATE_FILE, state)
        print(f"\nLoop {loop_num} complete.")

    print(f"\n{'='*60}")
    print(f"All {num_loops} loops complete.")
    print(f"Final GRPO model: {state['grpo_checkpoints'][-1]}")
    print(f"Generated data files: {len(state['generated_files'])}")
    print(f"{'='*60}")
    return state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full SFT → GRPO → generate loop")
    parser.add_argument(
        "--loops", type=int, default=None,
        help=f"Number of loops to run (default: {cfg.loop.num_loops})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last saved loop state",
    )
    parser.add_argument(
        "--skip-sft", action="store_true",
        help="Skip SFT in the first loop (e.g., already have an SFT checkpoint)",
    )
    args = parser.parse_args()

    if args.skip_sft:
        cfg.loop.start_from_sft = False

    run(cfg, resume=args.resume, num_loops=args.loops)
