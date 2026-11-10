"""Training ablation: reassign CMT's final weights, preserving their multiset.

This module never computes/changes CMT scores or solves an allocation. Its input
is the FINAL global valid-token allocation for one PPO optimizer update.
"""

from __future__ import annotations

import hashlib
import time

import torch

CONTROL_NUMERIC_FIELDS = (
    "cmt_random_weight_control",
    "cmt_weight_mean",
    "cmt_weight_clip_min",
    "cmt_weight_clip_max",
    "cmt_effective_sample_size",
    "cmt_permutation_token_count",
    "cmt_permutation_fixed_points",
    "cmt_permutation_fixed_point_fraction",
    "cmt_permutation_moved_fraction",
    "cmt_permutation_changed_weight_fraction",
    "cmt_permutation_changed_raw_weight_fraction",
    "cmt_permutation_mean_abs_displacement",
    "cmt_permutation_time_sec",
)
CONTROL_FIELDS = (
    *CONTROL_NUMERIC_FIELDS,
    "cmt_weight_assignment",
    "cmt_permutation_seed",
    "cmt_permutation_sha256",
)


def random_weight_control_enabled(config) -> bool:
    enabled = config.get("selector", {}).get("cmt_random_weight_control", False)
    if not isinstance(enabled, bool):
        raise ValueError("selector.cmt_random_weight_control must be a boolean")
    return enabled


def permutation_seed(experiment_seed: int, optimizer_step: int) -> int:
    """Stable, independent RNG namespace; no rank, rollout or microbatch offset."""
    if int(optimizer_step) <= 0:
        raise ValueError("CMT permutation needs a positive global optimizer step")
    payload = (
        f"cmt-final-weight-permutation-v1:{int(experiment_seed)}:{int(optimizer_step)}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") & (
        (1 << 63) - 1
    )


@torch.no_grad()
def permute_final_cmt_weights(
    raw_weights, final_weights, *, experiment_seed: int, optimizer_step: int
):
    """Pure reindexing of aligned raw/final weights; no clipping/renormalization.

    Every rank already owns the same gathered vector. An independent CPU
    generator makes the permutation identical on all ranks without touching
    the global CPU/CUDA RNGs. The caller scatters only after this operation.
    """
    if (
        final_weights.ndim != 1
        or final_weights.numel() == 0
        or raw_weights.shape != final_weights.shape
        or raw_weights.device != final_weights.device
    ):
        raise ValueError(
            "Expected aligned, non-empty global 1-D raw/final CMT weights on the same device"
        )
    started = time.perf_counter()
    seed = permutation_seed(experiment_seed, optimizer_step)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    count = final_weights.numel()
    permutation = torch.randperm(count, generator=generator, device="cpu")
    identity = torch.arange(count, device="cpu")
    indices = permutation.to(device=final_weights.device)
    raw = raw_weights.index_select(0, indices)
    final = final_weights.index_select(0, indices)
    fixed = int(permutation.eq(identity).sum())
    diagnostics = {
        "cmt_permutation_seed": seed,
        "cmt_permutation_sha256": hashlib.sha256(
            permutation.numpy().tobytes()
        ).hexdigest(),
        "cmt_permutation_token_count": int(count),
        "cmt_permutation_fixed_points": fixed,
        "cmt_permutation_fixed_point_fraction": fixed / count,
        "cmt_permutation_moved_fraction": 1 - fixed / count,
        "cmt_permutation_changed_weight_fraction": float(
            final.ne(final_weights).double().mean()
        ),
        "cmt_permutation_changed_raw_weight_fraction": float(
            raw.ne(raw_weights).double().mean()
        ),
        "cmt_permutation_mean_abs_displacement": float(
            (permutation - identity).abs().double().mean()
        ),
        "cmt_permutation_time_sec": time.perf_counter() - started,
    }
    return raw, final, diagnostics
