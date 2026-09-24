"""Rebuild metric-specific checkpoint histories from committed step summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config


_STEP_DIRECTORY_PATTERN = re.compile(r"^step-(\d+)$")
_METHOD_ALIASES = {
    "opd": "opd",
    "ta": "ta",
    "ta-opd": "ta",
    "ta_opd": "ta",
    "cmt": "cmt",
    "cmt-opd": "cmt",
    "cmt_opd": "cmt",
    "grpo": "grpo",
    "iw": "iw",
    "iw-opd": "iw",
    "iw_opd": "iw",
    "rac": "rac",
    "pgt": "pgt",
}


def _metric_slug(metric_name: str) -> str:
    return metric_name.replace("@", "_at_").replace("-", "_")


def _pass8_metric(parameters: dict[str, Any], source: Path) -> str:
    samples = int(parameters.get("num_responses", 0))
    metric = str(parameters.get("metric", "")).strip().lower()
    if metric != "pass@8" or samples != 8:
        raise ValueError(
            f"{source}: expected metric='pass@8' and num_responses=8; "
            f"got metric={metric!r}, num_responses={samples}"
        )
    return metric


def _write_recovered_artifacts(
    run_output: Path,
    history_path: Path,
    metrics_path: Path,
    history: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    """Atomically replace both recovery outputs without importing CUDA code."""
    token = uuid.uuid4().hex
    history_temp = run_output / f".{history_path.name}.recovery-{token}.tmp"
    metrics_temp = run_output / f".{metrics_path.name}.recovery-{token}.tmp"
    canonical_fields = (
        "step",
        "method",
        "backend",
        "benchmark",
        "correct",
        "total",
        "accuracy",
        "avg_at_n",
        "avg_at_8",
        "pass_at_k",
        "pass_at_8",
        "problems",
        "samples_per_problem",
        "metric",
        "evaluation_time_sec",
    )
    history_backup = run_output / f".{history_path.name}.recovery-backup-{token}"
    metrics_backup = run_output / f".{metrics_path.name}.recovery-backup-{token}"
    history_existed = history_path.exists()
    metrics_existed = metrics_path.exists()
    try:
        with history_temp.open("w", encoding="utf-8") as handle:
            for row in history:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        with metrics_temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=canonical_fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(metric_rows)
        if history_existed:
            os.replace(history_path, history_backup)
        if metrics_existed:
            os.replace(metrics_path, metrics_backup)
        try:
            os.replace(history_temp, history_path)
            os.replace(metrics_temp, metrics_path)
        except BaseException:
            history_path.unlink(missing_ok=True)
            metrics_path.unlink(missing_ok=True)
            if history_existed and history_backup.exists():
                os.replace(history_backup, history_path)
            if metrics_existed and metrics_backup.exists():
                os.replace(metrics_backup, metrics_path)
            raise
    finally:
        history_temp.unlink(missing_ok=True)
        metrics_temp.unlink(missing_ok=True)
        history_backup.unlink(missing_ok=True)
        metrics_backup.unlink(missing_ok=True)


def _canonical_method(method: str) -> str:
    normalized = str(method).strip().lower()
    try:
        return _METHOD_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported method for history recovery: {method!r}") from error


def _max_steps(run_output: Path, recovered_steps: list[int]) -> int:
    candidates = list(recovered_steps)
    summary_path = run_output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("steps") is not None:
            candidates.append(int(summary["steps"]))
    latest_path = run_output / "latest.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if latest.get("step") is not None:
            candidates.append(int(latest["step"]))
    return max(candidates)


def _require_committed_artifact(step_dir: Path, value: Any, description: str) -> None:
    if not value:
        raise ValueError(f"{step_dir}: summary has no {description} path")
    path = Path(str(value)).expanduser()
    if path.is_file():
        return
    relocated = step_dir / path.name
    if not relocated.is_file():
        raise FileNotFoundError(
            f"{step_dir}: missing committed {description}: {path} "
            f"(also checked {relocated})"
        )


def _history_row_from_summary(
    suite: dict[str, Any],
    *,
    summary_path: Path,
    step: int,
    max_steps: int,
    method: str,
    expected_metric: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parameters = dict(suite.get("parameters", {}))
    samples = int(parameters.get("num_responses", 0))
    actual_metric = _pass8_metric(parameters, summary_path)
    if actual_metric != expected_metric:
        raise ValueError(
            f"{summary_path}: metric is {actual_metric!r}, expected {expected_metric!r}"
        )
    benchmarks = suite.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError(f"{summary_path}: evaluation summary has no benchmarks")

    step_dir = summary_path.parent
    _require_committed_artifact(
        step_dir, suite.get("detailed_outputs"), "detailed output"
    )
    recovered_benchmarks: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for benchmark, result in benchmarks.items():
        if not isinstance(result, dict):
            raise ValueError(f"{summary_path}: invalid result for {benchmark}")
        _require_committed_artifact(
            step_dir, result.get("predictions"), f"{benchmark} predictions"
        )
        if result.get("accuracy") is None:
            raise ValueError(f"{summary_path}: {benchmark} has no accuracy")
        recovered = {
            "correct": result.get("correct"),
            "total": result.get("total"),
            "accuracy": result["accuracy"],
            "avg_at_n": result.get("avg_at_n", result["accuracy"]),
            **(
                {"avg_at_8": result["avg_at_8"]}
                if "avg_at_8" in result
                else {}
            ),
            "problems": result.get("problems"),
            **(
                {"pass_at_k": result["pass_at_k"]}
                if "pass_at_k" in result
                else {}
            ),
            **(
                {"pass_at_8": result["pass_at_8"]}
                if "pass_at_8" in result
                else {}
            ),
            "samples_per_problem": result.get("samples_per_problem", samples),
            "metric": actual_metric,
        }
        recovered_benchmarks[str(benchmark)] = recovered
        metric_rows.append(
            {
                "step": step,
                "method": method,
                "backend": str(parameters.get("backend", "vllm")),
                "benchmark": str(benchmark),
                **recovered,
                "evaluation_time_sec": None,
            }
        )
    return (
        {
            "step": step,
            "max_steps": max_steps,
            "method": method,
            "model_role": "base_student" if step == 0 else method,
            "backend": str(parameters.get("backend", "vllm")),
            "evaluation_time": None,
            "base_cache_status": None,
            "benchmarks": recovered_benchmarks,
            "parameters": parameters,
            "details": str(summary_path.resolve()),
            "recovered_from_step_summary": True,
        },
        metric_rows,
    )


def rebuild_evaluation_history(
    run_output: str | Path,
    method: str,
    *,
    metric: str = "pass@8",
) -> dict[str, Any]:
    run_output = Path(run_output).expanduser().resolve()
    if not run_output.is_dir():
        raise FileNotFoundError(f"Run output does not exist: {run_output}")
    config_path = run_output / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved training config: {config_path}")
    config = load_config(config_path)
    method = _canonical_method(method)
    expected_metric = str(metric).strip().lower()
    if expected_metric != "pass@8":
        raise ValueError("This recovery command currently requires metric=pass@8")

    output_subdir = str(
        config.get("training_evaluation", {}).get("output_subdir", "training_eval")
    )
    slug = _metric_slug(expected_metric)
    eval_root = run_output / f"{output_subdir}_{slug}"
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Missing pass@8 evaluation directory: {eval_root}")

    committed: list[tuple[int, Path, dict[str, Any]]] = []
    skipped_incomplete: list[str] = []
    for step_dir in sorted(eval_root.iterdir()):
        if not step_dir.is_dir():
            continue
        match = _STEP_DIRECTORY_PATTERN.fullmatch(step_dir.name)
        if match is None:
            continue
        summary_path = step_dir / "summary.json"
        if not summary_path.is_file():
            skipped_incomplete.append(str(step_dir))
            continue
        suite = json.loads(summary_path.read_text(encoding="utf-8"))
        committed.append((int(match.group(1)), summary_path, suite))
    if not committed:
        raise RuntimeError(f"No committed step summaries were found under {eval_root}")

    steps = [step for step, _path, _suite in committed]
    if len(steps) != len(set(steps)):
        raise RuntimeError(f"Duplicate pass@8 step directories under {eval_root}")
    max_steps = _max_steps(run_output, steps)
    history: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for step, summary_path, suite in committed:
        history_row, rows = _history_row_from_summary(
            suite,
            summary_path=summary_path,
            step=step,
            max_steps=max_steps,
            method=method,
            expected_metric=expected_metric,
        )
        history.append(history_row)
        metric_rows.extend(rows)
    history.sort(key=lambda row: int(row["step"]))
    metric_rows.sort(
        key=lambda row: (int(row["step"]), str(row["benchmark"]))
    )

    history_path = run_output / f"eval_history_{slug}.jsonl"
    metrics_path = run_output / f"eval_metrics_{slug}.csv"
    _write_recovered_artifacts(
        run_output,
        history_path,
        metrics_path,
        history,
        metric_rows,
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "committed_step_summaries",
        "metric": expected_metric,
        "method": method,
        "run_output": str(run_output),
        "evaluation_root": str(eval_root),
        "history": str(history_path),
        "metrics": str(metrics_path),
        "recovered_steps": steps,
        "skipped_incomplete_directories": skipped_incomplete,
        "complete_sweep_claimed": False,
    }
    manifest_path = run_output / f"eval_history_recovery_{slug}.json"
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.recovery-{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**manifest, "manifest": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild eval_history_pass_at_8.jsonl from committed "
            "training_eval_pass_at_8/step-*/summary.json artifacts"
        )
    )
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--metric", default="pass@8")
    args = parser.parse_args(argv)
    result = rebuild_evaluation_history(
        args.run_output, args.method, metric=args.metric
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
