"""Non-destructive evaluation of explicitly selected checkpoints across seeds.

The old histories are read-only inputs. Each experiment tag owns separate
histories and immutable, resumable per-checkpoint/per-seed prediction artifacts.
Heavy model imports are deferred so planning and summary inspection work offline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import load_config, resolve_runtime_paths, save_config
from .evaluation_context import resolve_evaluation_max_model_len

BENCHMARKS = (
    "Competition-MATH",
    "MATH-500",
    "AIME24",
    "AIME25",
    "GPQA-Diamond",
    "AMC23",
)
METRICS = {"avg@8": "avg_at_8", "pass@8": "pass_at_8"}
LEGACY_FILES = {
    "avg@8": "eval_history.jsonl",
    "pass@8": "eval_history_pass_at_8.jsonl",
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_run_output(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (
        [root]
        if (root / "resolved_config.yaml").is_file()
        else sorted(root.glob("*/resolved_config.yaml"))
    )
    candidates = [p.parent if p.is_file() else p for p in candidates]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one method run with resolved_config.yaml under {root}; found {candidates}. Pass the method directory explicitly."
        )
    return candidates[0]


def checkpoint_step(checkpoint: Path, run: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if match:
        return int(match.group(1))
    if checkpoint.name == "final":
        latest = run / "latest.json"
        if latest.is_file():
            value = json.loads(latest.read_text(encoding="utf-8"))
            if (
                Path(str(value.get("checkpoint", ""))).name == "final"
                and "step" in value
            ):
                return int(value["step"])
        summary = run / "summary.json"
        if summary.is_file():
            value = json.loads(summary.read_text(encoding="utf-8"))
            if "steps" in value:
                return int(value["steps"])
        rows = _read_rows(run / "metrics.jsonl")
        if rows:
            return int(rows[-1]["step"])
    raise ValueError(
        f"Cannot infer optimizer step for {checkpoint}; use checkpoint-N or a final/ with latest.json/summary.json/metrics.jsonl"
    )


def _legacy_row(
    run: Path, filename: str, step: int, method: str
) -> dict[str, Any] | None:
    matches = [
        row
        for row in _read_rows(run / filename)
        if int(row.get("step", -1)) == step and row.get("method", method) == method
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous duplicate step={step}, method={method} in {run / filename}"
        )
    return matches[0] if matches else None


def _metric_value(row: dict[str, Any], benchmark: str, metric: str) -> float | None:
    result = row.get("benchmarks", {}).get(benchmark, {})
    value = result.get(METRICS[metric])
    parameters = row.get("parameters", {})
    selected = result.get("metric", parameters.get("metric"))
    samples = result.get("samples_per_problem", parameters.get("num_responses"))
    if value is None and selected == metric and int(samples or 0) == 8:
        value = result.get("accuracy")
    if value is None:
        return None
    if int(samples or 8) != 8:
        raise ValueError(f"{benchmark}: expected eight responses, got {samples}")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {benchmark} {metric} score: {value}")
    return value


def _row_parameters(row: dict[str, Any], run: Path) -> dict[str, Any]:
    """Older history rows sometimes point at richer per-step summaries."""
    parameters: dict[str, Any] = {}
    details = row.get("details")
    if details:
        path = Path(details)
        if not path.is_absolute():
            path = run / path
        if path.is_file():
            parameters.update(
                json.loads(path.read_text(encoding="utf-8")).get("parameters", {})
            )
    parameters.update(row.get("parameters", {}))
    return parameters


def _fingerprint(checkpoint: Path) -> str:
    # Metadata only: do not read many GiB of model weights just to resume eval.
    files = sorted(
        p
        for p in checkpoint.iterdir()
        if p.is_file()
        and (
            p.suffix in {".json", ".safetensors", ".bin"}
            or p.name.startswith("tokenizer")
        )
    )
    payload = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def build_plan(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    run = resolve_run_output(args.run_output)
    config = resolve_runtime_paths(load_config(run / "resolved_config.yaml"))
    # Import only for a real run configuration, never rewrite the old YAML.
    from .evaluation import ensure_extended_benchmark_specs

    config = ensure_extended_benchmark_specs(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        raise ValueError("--tag must be a simple filename component")
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds or min(args.seeds) < 0:
        raise ValueError("--seeds must contain distinct non-negative integers")
    method = str(config["experiment"]["method"])
    fresh_seeds = bool(getattr(args, "fresh_seeds", False))
    if fresh_seeds and args.baseline_seed is not None:
        raise ValueError("--fresh-seeds cannot be combined with --baseline-seed")
    if fresh_seeds and len(args.seeds) < 2:
        raise ValueError(
            "--fresh-seeds requires at least two seeds for sample standard deviation"
        )
    plan = []
    seen_steps: set[int] = set()
    defaults = config.get("training_evaluation", config.get("evaluation", {}))
    for selected in args.checkpoints:
        candidate = Path(selected)
        if not candidate.is_absolute():
            if str(selected).isdigit():
                candidate = Path(f"checkpoint-{int(selected):06d}")
            candidate = run / candidate
        checkpoint = candidate.resolve()
        if checkpoint.parent != run or not (checkpoint / "config.json").is_file():
            raise ValueError(
                f"Checkpoint must be a saved model directly under {run}: {checkpoint}"
            )
        step = checkpoint_step(checkpoint, run)
        if step in seen_steps:
            raise ValueError(
                f"Two selected checkpoints represent step {step}; choose one of checkpoint-N and final"
            )
        seen_steps.add(step)
        baseline: dict[str, Any] = {}
        protocols = []
        if fresh_seeds:
            protocol = {
                key: defaults.get(key)
                for key in ("temperature", "top_p", "max_new_tokens")
            }
            if any(value is None for value in protocol.values()):
                raise ValueError(
                    f"Incomplete resolved evaluation sampling protocol: {protocol}"
                )
            if (
                not math.isfinite(float(protocol["temperature"]))
                or float(protocol["temperature"]) <= 0
            ):
                raise ValueError(
                    "Fresh evaluation seeds require finite temperature > 0"
                )
            if (
                not 0 < float(protocol["top_p"]) <= 1
                or int(protocol["max_new_tokens"]) <= 0
            ):
                raise ValueError(
                    "Evaluation requires top_p in (0, 1] and max_new_tokens > 0"
                )
            protocols = [protocol, protocol]
        for metric, filename in ({} if fresh_seeds else LEGACY_FILES).items():
            row = _legacy_row(run, filename, step, method)
            # A modern history may contain both metrics from one generation.
            if row is None or any(
                _metric_value(row, name, metric) is None for name in BENCHMARKS
            ):
                other = _legacy_row(run, "eval_history.jsonl", step, method)
                if other and all(
                    _metric_value(other, name, metric) is not None
                    for name in BENCHMARKS
                ):
                    row, filename = other, "eval_history.jsonl"
            if row is None or any(
                _metric_value(row, name, metric) is None for name in BENCHMARKS
            ):
                raise ValueError(
                    f"Missing old {metric} results for one or more of the six datasets at step {step} in {run}. Complete the baseline history before aggregating three seeds."
                )
            params = _row_parameters(row, run)
            declared_seed = params.get("seed", params.get("vllm", {}).get("seed"))
            if declared_seed is None and args.baseline_seed is None:
                raise ValueError(
                    f"Old {metric} at step {step} has no recorded evaluation seed. Pass --baseline-seed with the actual old EVAL seed (not the training seed)."
                )
            if (
                declared_seed is not None
                and args.baseline_seed is not None
                and int(declared_seed) != args.baseline_seed
            ):
                raise ValueError(
                    f"--baseline-seed={args.baseline_seed} conflicts with recorded evaluation seed={declared_seed} at step {step}"
                )
            old_seed = int(
                declared_seed if declared_seed is not None else args.baseline_seed
            )
            if old_seed in args.seeds:
                raise ValueError(
                    f"New seeds must differ from old evaluation seed {old_seed}"
                )
            protocol = {
                key: params.get(key, defaults.get(key))
                for key in ("temperature", "top_p", "max_new_tokens")
            }
            if any(v is None for v in protocol.values()):
                raise ValueError(
                    f"Incomplete old sampling protocol for step {step}: {protocol}"
                )
            if float(protocol["temperature"]) <= 0:
                raise ValueError(
                    "Multiple evaluation seeds require sampling temperature > 0"
                )
            if params.get("limit") is not None:
                raise ValueError(
                    "Old history used an evaluation limit; cannot combine it with a full-dataset evaluation"
                )
            if int(params.get("num_responses", 8)) != 8:
                raise ValueError("Baseline must have num_responses=8")
            baseline[metric] = {
                "seed": old_seed,
                "history": str(run / filename),
                "scores": {
                    name: _metric_value(row, name, metric) for name in BENCHMARKS
                },
                "problems": {
                    name: row["benchmarks"][name].get("problems") for name in BENCHMARKS
                },
            }
            protocols.append(protocol)
        if protocols[0] != protocols[1]:
            raise ValueError(
                f"Old avg@8 and pass@8 use different sampling protocols at step {step}: {protocols}. They cannot share one new generation and be compared as the same experiment."
            )
        settings = {
            **protocols[0],
            "backend": "vllm",
            "num_responses": 8,
            "metric": "avg@8",
            "limit": None,
            "benchmark_names": list(BENCHMARKS),
            "vllm": dict(defaults.get("vllm", {})),
        }
        for key in ("max_num_seqs", "gpu_memory_utilization", "max_model_len"):
            value = getattr(args, key, None)
            if value is not None:
                settings["vllm"][key] = value
        settings["vllm"]["tensor_parallel_size"] = 1
        settings["vllm"]["max_model_len"] = resolve_evaluation_max_model_len(
            checkpoint, settings["vllm"].get("max_model_len", 4096)
        )
        plan.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_fingerprint": _fingerprint(checkpoint),
                "method": method,
                "baseline": baseline,
                "settings": settings,
            }
        )
    return run, config, plan


def _history_row(
    item: dict, suite: dict, seed: int, metric: str, details: Path
) -> dict:
    results = copy.deepcopy(suite["benchmarks"])
    for result in results.values():
        result["accuracy"] = result["avg_at_n"] = float(result[METRICS[metric]])
        result["metric"] = metric
        if metric == "pass@8":
            result["pass_at_k"] = result["pass_at_8"]
    return {
        "step": item["step"],
        "method": item["method"],
        "seed": seed,
        "checkpoint": item["checkpoint"],
        "benchmarks": results,
        "parameters": {**suite["parameters"], "seed": seed, "metric": metric},
        "details": str(details),
    }


def _persist_history(path: Path, row: dict) -> None:
    rows = [
        r
        for r in _read_rows(path)
        if (r.get("step"), r.get("method")) != (row["step"], row["method"])
    ]
    rows.append(row)
    rows.sort(key=lambda r: int(r["step"]))
    temporary = path.with_suffix(".jsonl.tmp")
    try:
        temporary.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(
    plan: list[dict], suites: dict[tuple[int, int], dict], seeds: list[int]
) -> dict:
    rows = []
    for item in plan:
        for benchmark in BENCHMARKS:
            for metric, field in METRICS.items():
                baseline = item["baseline"].get(metric)
                values = [float(baseline["scores"][benchmark])] if baseline else []
                expected_count = baseline["problems"][benchmark] if baseline else None
                for seed in seeds:
                    result = suites[(item["step"], seed)]["benchmarks"][benchmark]
                    if expected_count is not None and int(expected_count) != int(
                        result["problems"]
                    ):
                        raise ValueError(
                            f"Dataset problem count changed for {benchmark} at step {item['step']}"
                        )
                    expected_count = int(result["problems"])
                    values.append(float(result[field]))
                mean, std = statistics.mean(values), statistics.stdev(values)
                rows.append(
                    {
                        "checkpoint": item["checkpoint"],
                        "step": item["step"],
                        "method": item["method"],
                        "benchmark": benchmark,
                        "metric": metric,
                        "seeds": [baseline["seed"], *seeds]
                        if baseline
                        else list(seeds),
                        "values": values,
                        "n": len(values),
                        "mean": mean,
                        "std": std,
                        "mean_percent": 100 * mean,
                        "std_percentage_points": 100 * std,
                        "mean_pm_std": f"{100 * mean:.2f} ± {100 * std:.2f}",
                        "baseline_history": baseline["history"] if baseline else None,
                    }
                )
    return {
        "schema_version": 1,
        "variation": "evaluation sampling seeds; fixed trained checkpoint",
        "standard_deviation": "sample standard deviation (ddof=1); not standard error or confidence interval",
        "display_units": "percent; ± is percentage points",
        "results": rows,
    }


def _write_summary(run: Path, tag: str, summary: dict) -> None:
    stem = run / f"eval_summary_{tag}"
    _write_json(Path(str(stem) + ".json"), summary)
    fields = (
        "checkpoint",
        "step",
        "method",
        "benchmark",
        "metric",
        "n",
        "seeds",
        "values",
        "mean_percent",
        "std_percentage_points",
        "mean_pm_std",
    )
    temporary = Path(str(stem) + ".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["results"])
    temporary.replace(Path(str(stem) + ".csv"))
    lines = [
        "Mean ± sample standard deviation over evaluation seeds, in percent.",
        "The trained checkpoint is fixed; this does not measure training-seed variation.",
        "",
        "| Step | Dataset | avg@8 | pass@8 |",
        "|---|---|---|---|",
    ]
    for avg, passed in zip(summary["results"][::2], summary["results"][1::2]):
        lines.append(
            f"| {avg['step']} | {avg['benchmark']} | {avg['mean_pm_std']} | {passed['mean_pm_std']} |"
        )
    Path(str(stem) + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@contextmanager
def _run_lock(path: Path):
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluation owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(args: argparse.Namespace, evaluator=None) -> dict:
    run, config, plan = build_plan(args)
    if args.dry_run:
        return {
            "run_output": str(run),
            "tag": args.tag,
            "new_seeds": args.seeds,
            "plan": plan,
            "dry_run": True,
        }
    if evaluator is None:
        from .vllm_evaluation import evaluate_vllm_distributed

        evaluator = evaluate_vllm_distributed
    world_size = args.world_size
    if world_size is None:
        devices = [
            device
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if devices:
            world_size = len(devices)
        else:
            import torch

            world_size = max(torch.cuda.device_count(), 1)
    if world_size <= 0:
        raise ValueError("--world-size must be positive")
    suites = {}
    root = run / "eval_multiseed" / args.tag
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root / ".lock"):
        for item in plan:
            for seed in args.seeds:
                settings = copy.deepcopy(item["settings"])
                settings["vllm"]["seed"] = seed
                destination = root / Path(item["checkpoint"]).name / f"seed_{seed}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                request = {
                    "schema_version": 1,
                    "checkpoint": item["checkpoint"],
                    "checkpoint_fingerprint": item["checkpoint_fingerprint"],
                    "settings": settings,
                    "models": config.get("models"),
                    "data": config.get("data"),
                    "benchmarks": config["evaluation"]["benchmarks"],
                    "world_size": world_size,
                }
                if destination.exists():
                    if (
                        not (destination / "request.json").is_file()
                        or json.loads(
                            (destination / "request.json").read_text(encoding="utf-8")
                        )
                        != request
                    ):
                        raise ValueError(
                            f"Existing result has a different request: {destination}. Use a new --tag."
                        )
                    suite = json.loads(
                        (destination / "summary.json").read_text(encoding="utf-8")
                    )
                    print(f"Reusing committed result: {destination}", flush=True)
                else:
                    staged = Path(
                        tempfile.mkdtemp(prefix=".seed-eval-", dir=destination.parent)
                    )
                    try:
                        resolved = staged / "resolved_eval_config.yaml"
                        save_config(config, resolved)
                        print(
                            f"Evaluating {item['checkpoint']} with seed={seed} on all six datasets",
                            flush=True,
                        )
                        started = time.perf_counter()
                        suite = evaluator(
                            f"{item['method']} step {item['step']}",
                            item["checkpoint"],
                            config,
                            staged,
                            settings,
                            resolved,
                            world_size=world_size,
                        )
                        suite["evaluation_time_sec"] = time.perf_counter() - started
                        if set(suite["benchmarks"]) != set(BENCHMARKS):
                            raise ValueError(
                                "Evaluator did not return all six datasets"
                            )
                        parameters = suite.get("parameters", {})
                        for key in (
                            "temperature",
                            "top_p",
                            "max_new_tokens",
                            "num_responses",
                        ):
                            if parameters.get(key) != settings[key]:
                                raise ValueError(
                                    f"Evaluator did not honor requested {key}={settings[key]}"
                                )
                        if int(parameters.get("seed", -1)) != seed:
                            raise ValueError(
                                f"Evaluator did not honor requested seed={seed}"
                            )
                        for name, result in suite["benchmarks"].items():
                            for field in METRICS.values():
                                if (
                                    field not in result
                                    or not math.isfinite(float(result[field]))
                                    or not 0 <= float(result[field]) <= 1
                                ):
                                    raise ValueError(
                                        f"Missing or invalid {field} for {name}"
                                    )
                            if int(
                                result.get("samples_per_problem", 0)
                            ) != 8 or not int(result.get("problems", 0)):
                                raise ValueError(
                                    f"Expected all problems with eight responses for {name}"
                                )
                            prediction = result.get("predictions")
                            if prediction:
                                result["predictions"] = str(
                                    destination / Path(prediction).name
                                )
                        if suite.get("detailed_outputs"):
                            suite["detailed_outputs"] = str(
                                destination / Path(suite["detailed_outputs"]).name
                            )
                        _write_json(staged / "summary.json", suite)
                        _write_json(staged / "request.json", request)
                        staged.rename(destination)
                    finally:
                        if staged.exists():
                            shutil.rmtree(staged)
                suites[(item["step"], seed)] = suite
                for metric in METRICS:
                    suffix = "" if metric == "avg@8" else "_pass_at_8"
                    history = run / f"eval_history{suffix}_seed_{seed}_{args.tag}.jsonl"
                    _persist_history(
                        history,
                        _history_row(
                            item, suite, seed, metric, destination / "summary.json"
                        ),
                    )
        summary = summarize(plan, suites, args.seeds)
        summary["tag"] = args.tag
        _write_summary(run, args.tag, summary)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-output",
        required=True,
        help="Method directory or its parent run directory; may be in a different checkout",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Selected checkpoint-N names, step numbers, or final",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument(
        "--fresh-seeds",
        action="store_true",
        help="Evaluate and summarize only the requested seeds; do not read or require old baseline histories. Sampling protocol comes from the run's resolved evaluation config.",
    )
    parser.add_argument(
        "--baseline-seed",
        type=int,
        help="Actual old EVALUATION seed, only needed if absent from old artifacts",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Separate artifact namespace; defaults to seeds_43_44 etc.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="One TP=1 evaluator per visible GPU by default",
    )
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.tag = args.tag or "seeds_" + "_".join(map(str, args.seeds))
    print(json.dumps(execute(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
