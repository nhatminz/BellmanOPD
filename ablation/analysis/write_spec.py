#!/usr/bin/env python3
"""Write a small reproducibility record before an ablation run starts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    arm = sys.argv[2]
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    payload = {
        "ablation_kind": os.environ.get("ABLATION_KIND", "arm"),
        "arm": arm,
        "seed": int(os.environ.get("SEED", "42")),
        "rollout_seed": int(os.environ.get("ROLLOUT_SEED", "42")),
        "top_k": int(os.environ.get("TOP_K", "16")),
        "cmt_gain_support": "student_topk",
        "cmt_allocation_kl": float(os.environ.get("CMT_ALLOCATION_KL", "0.5")),
        "cmt_allocation_mode": os.environ.get(
            "CMT_ALLOCATION_MODE", "direct_bounded_gibbs"
        ),
        "cmt_weight_min": float(os.environ.get("CMT_WEIGHT_MIN", "0.5")),
        "cmt_weight_max": float(os.environ.get("CMT_WEIGHT_MAX", "2.0")),
        "cmt_correction_mode": os.environ.get("CMT_CORRECTION_MODE", "tanh_q99"),
        "cmt_correction_quantile": float(
            os.environ.get("CMT_CORRECTION_QUANTILE", "0.99")
        ),
        "cmt_final_allocation_kl": float(
            os.environ.get("CMT_FINAL_ALLOCATION_KL", "0.02")
        ),
        "cmt_gamma": float(os.environ.get("CMT_GAMMA", "1.0")),
        "cmt_successor_lambda": float(os.environ.get("CMT_SUCCESSOR_LAMBDA", "1.0")),
        "learning_rate": float(
            os.environ.get("LR", os.environ.get("LEARNING_RATE", "5e-6"))
        ),
        "global_batch_size": int(os.environ.get("GLOBAL_BATCH_SIZE", "64")),
        "num_responses": int(os.environ.get("NUM_RESPONSES", "4")),
        "ppo_mini_batch_size": int(os.environ.get("PPO_MINI_BATCH_SIZE", "64")),
        "micro_batch_size_per_gpu": int(
            os.environ.get("MICRO_BATCH_SIZE_PER_GPU", "16")
        ),
        "num_epochs": int(os.environ.get("NUM_EPOCHS", "3")),
        "save_interval": int(os.environ.get("SAVE_INTERVAL", "150")),
        "eval_interval": int(os.environ.get("TRAIN_EVAL_INTERVAL", "150")),
        "train_max_new_tokens": int(os.environ.get("MAX_RESPONSE_LEN", "4096")),
        "eval_max_new_tokens": int(os.environ.get("TRAIN_EVAL_MAX_NEW_TOKENS", "7168")),
        "eval_seed": int(os.environ.get("TRAIN_EVAL_SEED", "1234")),
        "eval_benchmarks": [
            item.strip()
            for item in os.environ.get(
                "TRAIN_EVAL_BENCHMARKS",
                "Competition-MATH,MATH-500,AIME24,AIME25,GPQA-Diamond,AMC23",
            ).replace(",", " ").split()
            if item.strip()
        ],
        "rollout_temperature": float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0")),
        "rollout_top_p": float(os.environ.get("ROLLOUT_TOP_P", "1.0")),
        "rollout_vllm_gpu_memory_utilization": float(
            os.environ.get("ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION", "0.60")
        ),
        "rollout_vllm_max_model_len": int(
            os.environ.get("ROLLOUT_VLLM_MAX_MODEL_LEN", "5200")
        ),
        "student_model": os.environ.get("STUDENT_MODEL", ""),
        "teacher_model": os.environ.get("TEACHER_MODEL", ""),
        "train_dataset": os.environ.get("TRAIN_DATASET", ""),
        "git_commit": commit,
        "command": os.environ.get("ABLATION_COMMAND", ""),
    }
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "ablation_spec.json"
    # A resume must retain the original run identity and tuned values.
    if not destination.exists():
        destination.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
