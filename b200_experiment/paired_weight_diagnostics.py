"""Detached, paired PROW/G-only diagnostics on random-trained on-policy tokens.

No model calls, RNG operations, score modifications or training weights live
here. Global PPO-group inputs must already have identical token ordering.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

TRAJECTORY_SOURCE = "random_weight_trained_student_on_policy_trajectories"
COUNT_FIELDS = (
    "valid_token_count",
    "zero_prospective_count",
    "out_of_support_count",
    "out_of_support_zero_count",
    "out_of_support_nonzero_count",
    "in_support_count",
    "in_support_zero_count",
)
FRACTION_DENOMINATORS = {
    "zero_prospective_fraction": ("zero_prospective_count", "valid_token_count"),
    "out_of_support_fraction": ("out_of_support_count", "valid_token_count"),
    "out_of_support_zero_fraction": ("out_of_support_zero_count", "valid_token_count"),
    "zero_attributable_to_out_of_support_fraction": (
        "out_of_support_zero_count",
        "zero_prospective_count",
    ),
    "in_support_zero_fraction": ("in_support_zero_count", "valid_token_count"),
    "in_support_zero_conditional_fraction": (
        "in_support_zero_count",
        "in_support_count",
    ),
}
SUBSETS = ("zero_prospective", "out_of_support")
PAIR_SUM_FIELDS = (
    "count",
    "prow_weight_sum",
    "g_only_weight_sum",
    "signed_diff_sum",
    "abs_diff_sum",
    "different_count",
    "different_gt_1e6_count",
)
PAIR_DERIVED_FIELDS = (
    "prow_weight_mean",
    "g_only_weight_mean",
    "signed_diff_mean",
    "abs_diff_mean",
    "abs_diff_median",
    "abs_diff_p95",
    "different_fraction",
    "different_gt_1e6_fraction",
)
ALLOCATION_SUM_FIELDS = ("count", "weight_sum", "squared_weight_sum", "w_log_w_sum")
ALLOCATION_DERIVED_FIELDS = (
    "mean_weight",
    "min_weight",
    "max_weight",
    "effective_sample_size",
    "realized_kl",
    "mean_one_passed",
    "clipping_passed",
)
ALLOCATIONS = ("prow", "g_only", "random")
CHECK_FIELDS = (
    "prow_random_identical_multiset",
    "prow_random_identical_realized_kl",
    "all_mean_one_passed",
    "all_clipping_passed",
)
PAIRED_FIELDS = (
    "paired_weight_diagnostics_enabled",
    "paired_weight_diagnostics_time_sec",
    *(f"paired_{name}" for name in (*COUNT_FIELDS, *FRACTION_DENOMINATORS)),
    *(
        f"paired_{subset}_{name}"
        for subset in SUBSETS
        for name in (*PAIR_SUM_FIELDS, *PAIR_DERIVED_FIELDS)
    ),
    *(
        f"paired_{allocation}_{name}"
        for allocation in ALLOCATIONS
        for name in (*ALLOCATION_SUM_FIELDS, *ALLOCATION_DERIVED_FIELDS)
    ),
    *(f"paired_check_{name}" for name in CHECK_FIELDS),
)


def paired_diagnostics_enabled(config):
    enabled = config.get("selector", {}).get("cmt_paired_weight_diagnostics", False)
    if not isinstance(enabled, bool):
        raise ValueError("selector.cmt_paired_weight_diagnostics must be a boolean")
    return enabled


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _fractions(counts):
    return {
        name: _ratio(counts[num], counts[den])
        for name, (num, den) in FRACTION_DENOMINATORS.items()
    }


def _pair_means(values):
    count = values["count"]
    return {
        "prow_weight_mean": _ratio(values["prow_weight_sum"], count),
        "g_only_weight_mean": _ratio(values["g_only_weight_sum"], count),
        "signed_diff_mean": _ratio(values["signed_diff_sum"], count),
        "abs_diff_mean": _ratio(values["abs_diff_sum"], count),
        "different_fraction": _ratio(values["different_count"], count),
        "different_gt_1e6_fraction": _ratio(values["different_gt_1e6_count"], count),
    }


def _moments(values):
    count, total = values["count"], values["weight_sum"]
    return {
        "mean_weight": total / count,
        "effective_sample_size": total**2 / values["squared_weight_sum"],
        # Normalize p=w/sum(w), also correct for tiny mean-one roundoff.
        "realized_kl": values["w_log_w_sum"] / total + math.log(count / total),
    }


@torch.no_grad()
def measure_paired_weights(
    local_gain,
    sequential_gain_raw,
    coverage_correction,
    prow_weights,
    g_only_weights,
    random_weights,
    *,
    weight_min,
    weight_max,
    allocation_mode,
    tolerance=1e-6,
):
    """Measure before update; detached PROW/G-only vectors never enter the loss."""
    vectors = [
        local_gain,
        sequential_gain_raw,
        coverage_correction,
        prow_weights,
        g_only_weights,
        random_weights,
    ]
    shape = prow_weights.shape
    if (
        prow_weights.ndim != 1
        or not prow_weights.numel()
        or any(v.shape != shape for v in vectors)
    ):
        raise ValueError(
            "Paired diagnostics require aligned, non-empty global 1-D token vectors"
        )
    if any(not bool(torch.isfinite(v).all()) for v in vectors):
        raise FloatingPointError("Non-finite paired diagnostic inputs")
    zero = sequential_gain_raw.detach().eq(0)  # exact zero, never a tolerance
    outside = coverage_correction.detach().eq(0)
    counts = {
        "valid_token_count": int(zero.numel()),
        "zero_prospective_count": int(zero.sum()),
        "out_of_support_count": int(outside.sum()),
        "out_of_support_zero_count": int((zero & outside).sum()),
        "out_of_support_nonzero_count": int((~zero & outside).sum()),
        "in_support_count": int((~outside).sum()),
        "in_support_zero_count": int((zero & ~outside).sum()),
    }
    counts.update(_fractions(counts))
    prow, gain_only = prow_weights.detach().double(), g_only_weights.detach().double()
    diff = prow - gain_only
    pairs, payload = {}, {}
    for name, mask in (("zero_prospective", zero), ("out_of_support", outside)):
        signed = diff[mask]
        absolute = signed.abs()
        count = int(mask.sum())
        entry = {
            "count": count,
            "prow_weight_sum": float(prow[mask].sum()),
            "g_only_weight_sum": float(gain_only[mask].sum()),
            "signed_diff_sum": float(signed.sum()),
            "abs_diff_sum": float(absolute.sum()),
            "different_count": int(signed.ne(0).sum()),
            "different_gt_1e6_count": int(absolute.gt(1e-6).sum()),
            "abs_diff_median": float(torch.quantile(absolute, 0.5)) if count else None,
            "abs_diff_p95": float(torch.quantile(absolute, 0.95)) if count else None,
        }
        entry.update(_pair_means(entry))
        pairs[name] = entry
        payload[name] = absolute  # only the rank-0 logger copies this to CPU
    allocations, sorted_weights = {}, {}
    bounded = allocation_mode != "gibbs"
    for name, weights in zip(
        ALLOCATIONS, (prow_weights, g_only_weights, random_weights)
    ):
        # Canonical reduction order makes identical multisets produce exactly
        # identical measured KL/ESS, independently of token permutation order.
        values = weights.detach().double().sort().values
        if bool((values < 0).any()) or float(values.sum()) <= 0:
            raise ValueError("Invalid nonnegative allocation weights")
        sorted_weights[name] = values
        entry = {
            "count": values.numel(),
            "weight_sum": float(values.sum()),
            "squared_weight_sum": float(values.square().sum()),
            "w_log_w_sum": float(
                torch.where(
                    values > 0, values * values.clamp_min(1e-300).log(), 0
                ).sum()
            ),
            "min_weight": float(values[0]),
            "max_weight": float(values[-1]),
        }
        entry.update(_moments(entry))
        entry["mean_one_passed"] = abs(entry["mean_weight"] - 1) <= tolerance
        entry["clipping_passed"] = (
            (
                entry["min_weight"] >= weight_min - tolerance
                and entry["max_weight"] <= weight_max + tolerance
            )
            if bounded
            else None
        )
        allocations[name] = entry
    checks = {
        "prow_random_identical_multiset": torch.equal(
            sorted_weights["prow"], sorted_weights["random"]
        ),
        "prow_random_identical_realized_kl": allocations["prow"]["realized_kl"]
        == allocations["random"]["realized_kl"],
        "all_mean_one_passed": all(
            row["mean_one_passed"] for row in allocations.values()
        ),
        "all_clipping_passed": all(
            row["clipping_passed"] for row in allocations.values()
        )
        if bounded
        else None,
    }
    if any(value is False for value in checks.values()):
        raise AssertionError(f"Paired weight allocation invariants failed: {checks}")
    return {
        "trajectory_source": TRAJECTORY_SOURCE,
        "zero_score_statistics": counts,
        "paired_comparison": pairs,
        "allocations": allocations,
        "checks": checks,
        "allocation_mode": allocation_mode,
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
    }, payload


def flatten_paired(diagnostics):
    result = {"paired_weight_diagnostics_enabled": True}
    result.update(
        {f"paired_{k}": v for k, v in diagnostics["zero_score_statistics"].items()}
    )
    for subset, values in diagnostics["paired_comparison"].items():
        result.update({f"paired_{subset}_{k}": v for k, v in values.items()})
    for allocation, values in diagnostics["allocations"].items():
        result.update({f"paired_{allocation}_{k}": v for k, v in values.items()})
    result.update({f"paired_check_{k}": v for k, v in diagnostics["checks"].items()})
    return result


def _write_json(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class PairedWeightDiagnosticsLogger:
    """Deduplicate optimizer steps; pool counts/sums, and exact quantiles at end.

    Per-step lossless NPZ samples allow *pooled* median/p95 at finalize, rather
    than averaging group quantiles. No extra rollouts/model evaluations occur.
    """

    def __init__(self, output_dir, *, enabled, resume_step=0):
        self.output = Path(output_dir)
        self.enabled = enabled
        self.records = {}
        if not enabled:
            return
        self.samples = self.output / "rebuttal_weight_diagnostics_samples"
        self.samples.mkdir(parents=True, exist_ok=True)
        self.journal = self.output / "rebuttal_weight_diagnostics_steps.jsonl"
        if self.journal.is_file():
            for line in self.journal.read_text().splitlines():
                record = json.loads(line)
                if record["trajectory_source"] != TRAJECTORY_SOURCE:
                    raise ValueError("Cannot mix paired diagnostics trajectory sources")
                if record["step"] <= resume_step:
                    self.records[record["step"]] = record
            temporary = self.journal.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(
                    json.dumps(row, allow_nan=False) + "\n"
                    for _, row in sorted(self.records.items())
                ),
                encoding="utf-8",
            )
            temporary.replace(self.journal)
        self._publish(final=False)

    def write(
        self, step, diagnostics, payload, *, allocation_id, rollout_id, ppo_group_index
    ):
        if not self.enabled:
            return
        if diagnostics["trajectory_source"] != TRAJECTORY_SOURCE:
            raise ValueError("Unexpected diagnostics trajectory source")
        arrays = {
            name: value.detach().double().cpu().numpy()
            for name, value in payload.items()
        }
        digest = hashlib.sha256()
        for name in SUBSETS:
            digest.update(name.encode())
            digest.update(arrays[name].tobytes())
        destination = (
            self.samples / f"step-{int(step):09d}-{digest.hexdigest()[:20]}.npz"
        )
        temporary = destination.with_suffix(".npz.tmp")
        for name in SUBSETS:
            if arrays[name].size != diagnostics["paired_comparison"][name]["count"]:
                raise ValueError("Paired quantile samples do not match subset counts")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
        record = {
            "step": int(step),
            "allocation_id": allocation_id,
            "rollout_id": int(rollout_id),
            "ppo_group_index": int(ppo_group_index),
            "samples": str(destination.relative_to(self.output)),
            **diagnostics,
        }
        self.records[int(step)] = record
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        self._publish(final=False)

    def _overall(self, final):
        records = list(self.records.values())
        counts = {
            key: sum(row["zero_score_statistics"][key] for row in records)
            for key in COUNT_FIELDS
        }
        counts.update(_fractions(counts))
        comparisons = {}
        for subset in SUBSETS:
            values = {
                key: math.fsum(row["paired_comparison"][subset][key] for row in records)
                for key in PAIR_SUM_FIELDS
            }
            for key in ("count", "different_count", "different_gt_1e6_count"):
                values[key] = int(values[key])
            values.update(_pair_means(values))
            values.update(abs_diff_median=None, abs_diff_p95=None)
            if final and values["count"]:
                parts = []
                for row in records:
                    with np.load(
                        self.output / row["samples"], allow_pickle=False
                    ) as samples:
                        parts.append(samples[subset])
                pooled = np.concatenate(parts)
                if pooled.size != values["count"]:
                    raise ValueError("Incomplete pooled paired quantile samples")
                values["abs_diff_median"], values["abs_diff_p95"] = map(
                    float, np.quantile(pooled, [0.5, 0.95])
                )
                del pooled, parts
            comparisons[subset] = values
        allocations = {}
        if records:
            for name in ALLOCATIONS:
                rows = [row["allocations"][name] for row in records]
                values = {
                    key: math.fsum(row[key] for row in rows)
                    for key in ALLOCATION_SUM_FIELDS
                }
                values["count"] = int(values["count"])
                values.update(_moments(values))
                values.update(
                    min_weight=min(row["min_weight"] for row in rows),
                    max_weight=max(row["max_weight"] for row in rows),
                    mean_one_passed=all(row["mean_one_passed"] for row in rows),
                    clipping_passed=all(row["clipping_passed"] for row in rows)
                    if all(row["clipping_passed"] is not None for row in rows)
                    else None,
                )
                allocations[name] = values
        checks = {
            name: all(row["checks"][name] for row in records)
            if records and all(row["checks"][name] is not None for row in records)
            else None
            for name in CHECK_FIELDS
        }
        return {
            "zero_score_statistics": counts,
            "paired_comparison": comparisons,
            "allocations": allocations,
            "checks": checks,
        }

    def _publish(self, *, final):
        overall = self._overall(final)
        result = {
            "schema_version": 1,
            "trajectory_source": TRAJECTORY_SOURCE,
            "status": "complete" if final else "partial",
            "quantiles_finalized": bool(final),
            "aggregation": "One record per global optimizer step; counts/sums token-weighted; pooled exact linear quantiles; no rollout duplication.",
            "different_weights_definition": "exact inequality; also report abs difference > 1e-6 separately",
            "zero_definition": "sequential_gain_raw == 0 exactly; out of support is coverage_correction == 0",
            "observed_optimizer_steps": len(self.records),
            "overall": overall,
            "per_step": [row for _, row in sorted(self.records.items())],
        }
        _write_json(self.output / "rebuttal_weight_diagnostics.json", result)
        destination = self.output / "rebuttal_weight_diagnostics.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["scope", "step", "trajectory_source", *PAIRED_FIELDS],
                extrasaction="ignore",
            )
            writer.writeheader()
            if self.records:
                writer.writerow(
                    {
                        "scope": "overall",
                        "step": "",
                        "trajectory_source": TRAJECTORY_SOURCE,
                        **flatten_paired(
                            {
                                "zero_score_statistics": overall[
                                    "zero_score_statistics"
                                ],
                                "paired_comparison": overall["paired_comparison"],
                                "allocations": overall["allocations"],
                                "checks": overall["checks"],
                            }
                        ),
                    }
                )
            for step, record in sorted(self.records.items()):
                writer.writerow(
                    {
                        "scope": "step",
                        "step": step,
                        "trajectory_source": TRAJECTORY_SOURCE,
                        **flatten_paired(record),
                    }
                )
        temporary.replace(destination)
        return result

    def finalize(self):
        return self._publish(final=True) if self.enabled else None
