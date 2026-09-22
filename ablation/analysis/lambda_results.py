#!/usr/bin/env python3
"""Validate and aggregate CMT successor-lambda ablation outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_ORDER = (
    "Competition-MATH",
    "MATH-500",
    "AIME24",
    "AIME25",
    "GPQA-Diamond",
    "AMC23",
)

# These fields describe every intended training/config difference relevant to
# this ablation. Lambda is deliberately absent and is validated separately.
INVARIANT_FIELDS = (
    "arm",
    "cmt_gain_support",
    "top_k",
    "cmt_allocation_kl",
    "cmt_allocation_mode",
    "cmt_weight_min",
    "cmt_weight_max",
    "cmt_correction_mode",
    "cmt_correction_quantile",
    "cmt_final_allocation_kl",
    "cmt_gamma",
    "learning_rate",
    "global_batch_size",
    "num_responses",
    "ppo_mini_batch_size",
    "micro_batch_size_per_gpu",
    "num_epochs",
    "save_interval",
    "eval_interval",
    "train_max_new_tokens",
    "eval_max_new_tokens",
    "eval_seed",
    "eval_benchmarks",
    "rollout_temperature",
    "rollout_top_p",
    "rollout_vllm_gpu_memory_utilization",
    "rollout_vllm_max_model_len",
    "student_model",
    "teacher_model",
    "train_dataset",
    "git_commit",
)

TRAINING_FIELDS = {
    "allocation_kl_final": ("allocation_kl_final",),
    "normalized_ess": ("normalized_ess",),
    "fraction_at_weight_min": ("fraction_at_weight_min",),
    "fraction_at_weight_max": ("fraction_at_weight_max",),
    "weight_final_min": ("weight_final_min",),
    "weight_final_max": ("weight_final_max",),
    "weight_mean": ("selector", "w", "mean"),
    "weight_std": ("selector", "w", "std"),
    "weight_max": ("selector", "w", "max"),
    "correction_kappa": ("correction_kappa",),
    "sequential_gain_raw_abs_q95": ("sequential_gain_raw_abs_q95",),
    "sequential_gain_raw_abs_q99": ("sequential_gain_raw_abs_q99",),
    "sequential_gain_robust_abs_q95": ("sequential_gain_robust_abs_q95",),
    "sequential_gain_robust_abs_q99": ("sequential_gain_robust_abs_q99",),
    "saturation_fraction_gt_1": ("correction_saturation_rate_1kappa",),
    "saturation_fraction_gt_2": ("correction_saturation_rate_2kappa",),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_file(run_dir: Path, name: str) -> Path | None:
    direct = run_dir / name
    if direct.is_file():
        return direct
    nested = run_dir / "cmt_opd" / name
    if nested.is_file():
        return nested
    matches = sorted(run_dir.glob(f"*/{name}"))
    return matches[0] if len(matches) == 1 else None


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _benchmark_value(row: dict[str, Any], benchmark: str, metric: str) -> float | None:
    values = row.get("benchmarks", {})
    item = values.get(benchmark)
    if item is None:
        normalized = benchmark.lower().replace("-", "").replace("_", "")
        for name, candidate in values.items():
            if name.lower().replace("-", "").replace("_", "") == normalized:
                item = candidate
                break
    return None if item is None else _finite_float(item.get(metric))


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "mean": statistics.fmean(items),
        "std": statistics.stdev(items) if len(items) > 1 else 0.0,
        "n": len(items),
    }


def _group_summary(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row["value"]))
    output = []
    for group, values in sorted(grouped.items()):
        item = dict(zip(keys, group))
        item.update(_summary(values))
        output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_lambda_results(
    input_root: Path,
    *,
    expected_lambdas: tuple[float, ...],
    metric: str,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for spec_path in sorted(input_root.rglob("ablation_spec.json")):
        spec = _read_json(spec_path)
        if spec.get("ablation_kind") != "lambda_successor":
            continue
        run_dir = spec_path.parent
        runs.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir.resolve()),
                "seed": int(spec["seed"]),
                "lambda": float(spec["cmt_successor_lambda"]),
                "spec": spec,
            }
        )
    if not runs:
        raise ValueError(f"No lambda_successor ablation specs found below {input_root}")

    seen: set[tuple[int, float]] = set()
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (run["seed"], run["lambda"])
        if key in seen:
            raise ValueError(f"Duplicate run for seed={key[0]}, lambda={key[1]}")
        seen.add(key)
        by_seed[run["seed"]].append(run)
        spec = run["spec"]
        if spec.get("arm") != "g_d":
            raise ValueError(f"{run['run_name']} is not canonical g_d CMT")
        if spec.get("cmt_gain_support") != "student_topk":
            raise ValueError(f"{run['run_name']} does not use Student Top-K g_t")
        if spec.get("cmt_correction_mode") != "tanh_q99":
            raise ValueError(f"{run['run_name']} does not use tanh_q99")
        if spec.get("cmt_allocation_mode") != "direct_bounded_gibbs":
            raise ValueError(f"{run['run_name']} does not use direct_bounded_gibbs")

    expected = {float(value) for value in expected_lambdas}
    invariant_by_seed: dict[str, dict[str, Any]] = {}
    for seed, seed_runs in sorted(by_seed.items()):
        actual = {float(run["lambda"]) for run in seed_runs}
        if actual != expected:
            raise ValueError(
                f"Seed {seed} has lambda values {sorted(actual)}, expected {sorted(expected)}"
            )
        reference = {
            key: seed_runs[0]["spec"].get(key) for key in INVARIANT_FIELDS
        }
        for run in seed_runs[1:]:
            current = {key: run["spec"].get(key) for key in INVARIANT_FIELDS}
            if current != reference:
                changed = [key for key in INVARIANT_FIELDS if current[key] != reference[key]]
                raise ValueError(
                    f"Seed {seed}: {run['run_name']} changes non-lambda fields: {changed}"
                )
        invariant_by_seed[str(seed)] = reference

    eval_records: list[dict[str, Any]] = []
    final_per_run: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        history_path = _run_file(run_dir, "eval_history.jsonl")
        eval_rows = _read_jsonl(history_path) if history_path else []
        per_benchmark: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in eval_rows:
            step = int(row.get("step", 0))
            for benchmark in BENCHMARK_ORDER:
                value = _benchmark_value(row, benchmark, metric)
                if value is None:
                    continue
                record = {
                    "run_name": run["run_name"],
                    "seed": run["seed"],
                    "lambda": run["lambda"],
                    "step": step,
                    "benchmark": benchmark,
                    "value": value,
                }
                eval_records.append(record)
                per_benchmark[benchmark].append((step, value))
        for benchmark, curve in per_benchmark.items():
            step, value = max(curve, key=lambda item: item[0])
            final_per_run.append(
                {
                    "run_name": run["run_name"],
                    "seed": run["seed"],
                    "lambda": run["lambda"],
                    "step": step,
                    "benchmark": benchmark,
                    "value": value,
                }
            )

        metrics_path = _run_file(run_dir, "metrics.jsonl")
        for row in _read_jsonl(metrics_path) if metrics_path else []:
            step = int(row.get("step", row.get("optimizer_step", 0)))
            extracted: dict[str, float] = {}
            for name, path in TRAINING_FIELDS.items():
                value = _finite_float(_nested(row, path))
                if value is not None:
                    extracted[name] = value
            kappa = extracted.get("correction_kappa")
            if kappa is not None and kappa > 0.0:
                for quantile in ("q95", "q99"):
                    raw = extracted.get(f"sequential_gain_raw_abs_{quantile}")
                    if raw is not None:
                        extracted[f"lambda_d_over_kappa_{quantile}"] = raw / kappa
            for name, value in extracted.items():
                training_records.append(
                    {
                        "run_name": run["run_name"],
                        "seed": run["seed"],
                        "lambda": run["lambda"],
                        "step": step,
                        "diagnostic": name,
                        "value": value,
                    }
                )

    public_runs = [{key: value for key, value in run.items() if key != "spec"} for run in runs]
    return {
        "schema_version": 1,
        "metric": metric,
        "expected_lambdas": sorted(expected),
        "benchmarks": list(BENCHMARK_ORDER),
        "runs": public_runs,
        "invariant_config_by_seed": invariant_by_seed,
        "config_validation": (
            "Within every seed, all tracked configuration fields are identical; "
            "only cmt_successor_lambda differs."
        ),
        "final_eval_summary": _group_summary(
            final_per_run, ("lambda", "benchmark")
        ),
        "eval_curve_summary": _group_summary(
            eval_records, ("lambda", "step", "benchmark")
        ),
        "training_summary": _group_summary(
            training_records, ("lambda", "step", "diagnostic")
        ),
        "raw_final_eval": final_per_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-lambdas", default="0.25 0.5 1.0 2.0")
    parser.add_argument("--metric", default="accuracy")
    args = parser.parse_args()
    expected = tuple(float(item) for item in args.expected_lambdas.replace(",", " ").split())
    summary = collect_lambda_results(
        args.input_root.resolve(), expected_lambdas=expected, metric=args.metric
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lambda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "lambda_runs.csv", summary["runs"])
    _write_csv(
        args.output_dir / "lambda_final_eval_summary.csv",
        summary["final_eval_summary"],
    )
    _write_csv(
        args.output_dir / "lambda_eval_curves_summary.csv",
        summary["eval_curve_summary"],
    )
    _write_csv(
        args.output_dir / "lambda_training_diagnostics_summary.csv",
        summary["training_summary"],
    )
    print(summary["config_validation"])
    print(f"Aggregated {len(summary['runs'])} runs into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
