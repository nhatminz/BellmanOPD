from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_config, resolve_runtime_paths, save_config
from .evaluation import evaluation_metric_name, metric_problem_score
from .evaluation_cache import base_evaluation_cache_key
from .vllm_evaluation import evaluate_vllm_distributed, evaluate_vllm_suite


PASSK_SCHEMA_VERSION = 1
SUPPORTED_BENCHMARKS = ("AIME24", "AIME25", "AMC23")
DEFAULT_CONFIG_BY_METHOD = {
    "opd": "qwen3_b200_opd.yaml",
    "cmt": "qwen3_b200_cmt.yaml",
    "ta": "qwen3_b200_ta.yaml",
    "ta-opd": "qwen3_b200_ta.yaml",
    "grpo": "qwen3_b200_grpo.yaml",
    "iw": "qwen3_b200_iw.yaml",
    "pgt": "qwen3_b200_pgt.yaml",
    "rac": "qwen3_b200_rac.yaml",
}


@dataclass(frozen=True)
class CheckpointSpec:
    model_group: str
    method: str
    label: str
    checkpoint: Path
    config: Path


def _slug(value: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else "_" for character in value
    )
    return "_".join(part for part in cleaned.split("_") if part) or "unnamed"


def _parse_int_values(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.replace(",", " ").split())
    if not values or any(value <= 0 for value in values):
        raise ValueError("Pass@K values must be non-empty positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"Pass@K values must be unique, got {values}")
    return tuple(sorted(values))


def _parse_float_or_string(value: str) -> float | str:
    normalized = str(value).strip().lower()
    return normalized if normalized == "auto" else float(normalized)


def _checkpoint_identity(path: Path) -> str:
    """Keep run/method/checkpoint names visible while the hash prevents clashes."""
    parts = path.parts[-3:]
    return _slug("_".join(parts))


def parse_checkpoint_spec(raw: str, repo_root: Path) -> CheckpointSpec:
    parts = [item.strip() for item in raw.split("|")]
    if len(parts) not in {4, 5}:
        raise ValueError(
            "Checkpoint spec must be MODEL_GROUP|METHOD|LABEL|CHECKPOINT[|CONFIG]"
        )
    model_group, method, label, checkpoint_raw = parts[:4]
    if not all((model_group, method, label, checkpoint_raw)):
        raise ValueError(f"Checkpoint spec contains an empty required field: {raw!r}")
    method_key = method.lower()
    if len(parts) == 5 and parts[4]:
        config = Path(parts[4]).expanduser()
    else:
        config_name = DEFAULT_CONFIG_BY_METHOD.get(method_key, "qwen3_b200_cmt.yaml")
        config = repo_root / "configs" / config_name
    if not config.is_absolute():
        config = repo_root / config
    checkpoint = Path(checkpoint_raw).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    checkpoint = checkpoint.resolve()
    config = config.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint lacks config.json: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {config}")
    return CheckpointSpec(model_group, method, label, checkpoint, config)


def compute_passk_metrics(
    correctness_by_problem: Iterable[Iterable[bool]], k_values: Iterable[int]
) -> dict[int, float]:
    """Compute several unbiased Pass@k values from one set of n samples."""
    rows = [list(map(bool, row)) for row in correctness_by_problem]
    if not rows:
        raise ValueError("Pass@K aggregation requires at least one problem")
    sample_counts = {len(row) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("Every problem must have the same number of samples")
    n = sample_counts.pop()
    if n <= 0:
        raise ValueError("Every problem must have at least one sample")
    values = tuple(sorted({int(value) for value in k_values}))
    if not values:
        raise ValueError("At least one K value is required")
    for k in values:
        # Reuse the evaluator's validation and exact metric naming.
        evaluation_metric_name(n, f"pass@{k}")
    return {
        k: sum(metric_problem_score(row, f"pass@{k}") for row in rows) / len(rows)
        for k in values
    }


def summarize_checkpoint_correctness(
    *,
    model_group: str,
    method: str,
    label: str,
    checkpoint: str | Path,
    correctness_by_benchmark: dict[str, list[list[bool]]],
    k_by_benchmark: dict[str, tuple[int, ...]],
    generation_dirs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark in SUPPORTED_BENCHMARKS:
        if benchmark not in correctness_by_benchmark:
            continue
        scores = compute_passk_metrics(
            correctness_by_benchmark[benchmark], k_by_benchmark[benchmark]
        )
        n = len(correctness_by_benchmark[benchmark][0])
        for k, score in scores.items():
            rows.append(
                {
                    "model_group": model_group,
                    "method": method,
                    "label": label,
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "benchmark": benchmark,
                    "k": k,
                    "num_samples": n,
                    "pass_at_k": score,
                    "generation_dir": ((generation_dirs or {}).get(benchmark, "")),
                }
            )
    return rows


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prediction_correctness(
    directory: Path, summary: dict[str, Any], benchmark: str, expected_n: int
) -> list[list[bool]]:
    result = summary["benchmarks"][benchmark]
    prediction_path = directory / Path(result["predictions"]).name
    if not prediction_path.is_file():
        raise ValueError(f"Missing raw prediction file: {prediction_path}")
    rows = _read_prediction_rows(prediction_path)
    if len(rows) != int(result["problems"]):
        raise ValueError(
            f"{benchmark} prediction count mismatch: {len(rows)} vs {result['problems']}"
        )
    correctness = []
    for row in rows:
        flags = list(map(bool, row.get("correct", [])))
        responses = row.get("responses", [])
        if len(flags) != expected_n or len(responses) != expected_n:
            raise ValueError(
                f"{benchmark} problem {row.get('id')!r} has incomplete cached samples: "
                f"correct={len(flags)}, responses={len(responses)}, expected={expected_n}"
            )
        correctness.append(flags)
    return correctness


def _cache_is_valid(
    directory: Path,
    *,
    cache_key: str,
    benchmarks: tuple[str, ...],
    expected_n: int,
) -> tuple[bool, dict[str, Any] | None]:
    manifest_path = directory / "passk_generation_manifest.json"
    summary_path = directory / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return False, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            return False, None
        if tuple(manifest.get("benchmarks", ())) != benchmarks:
            return False, None
        if int(manifest.get("num_responses", -1)) != expected_n:
            return False, None
        if tuple(summary.get("benchmarks", {})) != benchmarks:
            return False, None
        for benchmark in benchmarks:
            _prediction_correctness(directory, summary, benchmark, expected_n)
        detailed = directory / Path(summary["detailed_outputs"]).name
        if not detailed.is_file():
            return False, None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, None
    return True, summary


def _rewrite_artifact_paths(
    staging: Path, destination: Path, suite: dict[str, Any]
) -> None:
    for result in suite["benchmarks"].values():
        result["predictions"] = str(
            (destination / Path(result["predictions"]).name).resolve()
        )
    suite["detailed_outputs"] = str(
        (destination / Path(suite["detailed_outputs"]).name).resolve()
    )
    (staging / "summary.json").write_text(
        json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        backup = destination.with_name(
            f"{destination.name}.invalid-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        suffix = 0
        while backup.exists():
            suffix += 1
            backup = destination.with_name(
                f"{destination.name}.invalid-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
            )
        os.replace(destination, backup)
    os.replace(staging, destination)


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        return len([item for item in visible.split(",") if item.strip()])
    try:
        import torch

        return max(int(torch.cuda.device_count()), 1)
    except Exception:
        return 1


def _generate_or_reuse(
    *,
    spec: CheckpointSpec,
    config: dict[str, Any],
    destination: Path,
    benchmarks: tuple[str, ...],
    num_responses: int,
    runtime_settings: dict[str, Any],
    world_size: int,
) -> tuple[dict[str, Any], str]:
    cache_key = base_evaluation_cache_key(config, runtime_settings, spec.checkpoint)
    valid, cached = _cache_is_valid(
        destination,
        cache_key=cache_key,
        benchmarks=benchmarks,
        expected_n=num_responses,
    )
    if valid and cached is not None:
        print(f"CACHE HIT: {destination}", flush=True)
        return cached, "hit"

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    print(
        f"GENERATE: {spec.label} benchmarks={','.join(benchmarks)} "
        f"n={num_responses} -> {destination}",
        flush=True,
    )
    try:
        resolved_config = staging / "resolved_eval_config.yaml"
        save_config(config, resolved_config)
        if world_size > 1:
            suite = evaluate_vllm_distributed(
                spec.label,
                spec.checkpoint,
                config,
                staging,
                runtime_settings,
                resolved_config,
                world_size=world_size,
            )
        else:
            suite = evaluate_vllm_suite(
                spec.label,
                spec.checkpoint,
                config,
                staging,
                runtime_settings,
            )
        _rewrite_artifact_paths(staging, destination, suite)
        manifest = {
            "schema_version": PASSK_SCHEMA_VERSION,
            "cache_key": cache_key,
            "model_group": spec.model_group,
            "method": spec.method,
            "label": spec.label,
            "checkpoint": str(spec.checkpoint),
            "benchmarks": list(benchmarks),
            "num_responses": num_responses,
            "runtime_settings": runtime_settings,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (staging / "passk_generation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        valid, staged_suite = _cache_is_valid(
            staging,
            cache_key=cache_key,
            benchmarks=benchmarks,
            expected_n=num_responses,
        )
        if not valid or staged_suite is None:
            raise RuntimeError("New Pass@K generation failed artifact validation")
        _publish_directory(staging, destination)
        return staged_suite, "generated"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "model_group",
        "method",
        "label",
        "checkpoint",
        "benchmark",
        "k",
        "num_samples",
        "pass_at_k",
        "generation_dir",
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_experiment(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    specs = [parse_checkpoint_spec(raw, repo_root) for raw in args.checkpoint_spec]
    benchmarks = tuple(args.benchmarks)
    unknown = [name for name in benchmarks if name not in SUPPORTED_BENCHMARKS]
    if unknown:
        raise ValueError(f"Unsupported Pass@K benchmarks: {unknown}")
    aime_k = _parse_int_values(args.aime_k_values)
    amc_k = _parse_int_values(args.amc_k_values)
    aime_n = int(args.aime_num_samples)
    amc_n = int(args.amc_num_samples)
    if any(name in {"AIME24", "AIME25"} for name in benchmarks) and any(
        k > aime_n for k in aime_k
    ):
        raise ValueError(f"AIME K values {aime_k} exceed num_samples={aime_n}")
    if "AMC23" in benchmarks and any(k > amc_n for k in amc_k):
        raise ValueError(f"AMC23 K values {amc_k} exceed num_samples={amc_n}")
    if args.tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    world_size = int(args.world_size)
    if world_size < 0:
        raise ValueError("world_size must be zero (auto) or a positive integer")
    if world_size == 0:
        world_size = 1 if args.tensor_parallel_size > 1 else _visible_gpu_count()
    if world_size > _visible_gpu_count():
        raise ValueError(
            f"world_size={world_size} exceeds {_visible_gpu_count()} visible GPU(s)"
        )
    if world_size > 1 and args.tensor_parallel_size != 1:
        raise ValueError(
            "Replica-parallel evaluation requires tensor_parallel_size=1; "
            "set world_size=1 to use tensor parallelism"
        )

    output_root = args.output_root.resolve()
    run_root = output_root / "runs" / _slug(args.tag)
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for spec in specs:
        config = load_config(spec.config)
        if args.storage_root is not None:
            config.setdefault("paths", {})["storage_root"] = str(
                args.storage_root.expanduser().resolve()
            )
        config = resolve_runtime_paths(config)
        checkpoint_hash = hashlib.sha256(str(spec.checkpoint).encode()).hexdigest()[:8]
        run_name = "__".join(
            (
                _slug(spec.method),
                _checkpoint_identity(spec.checkpoint),
                checkpoint_hash,
            )
        )
        run_dir = run_root / _slug(spec.model_group) / run_name
        correctness_by_benchmark: dict[str, list[list[bool]]] = {}
        generation_dirs: dict[str, str] = {}
        cache_status: dict[str, str] = {}
        groups = []
        aime_benchmarks = tuple(
            name for name in benchmarks if name in {"AIME24", "AIME25"}
        )
        if aime_benchmarks:
            groups.append(("aime", aime_benchmarks, aime_n))
        if "AMC23" in benchmarks:
            groups.append(("amc23", ("AMC23",), amc_n))
        for group_name, group_benchmarks, n in groups:
            generation_dir = run_dir / "generations" / f"{group_name}_n{n}"
            runtime_settings = {
                "backend": "vllm",
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "num_responses": n,
                "metric": f"pass@{n}",
                "max_new_tokens": int(args.max_new_tokens),
                "limit": args.limit,
                "benchmark_names": list(group_benchmarks),
                "sync_timeout_sec": float(args.sync_timeout_sec),
                "vllm": {
                    "tensor_parallel_size": int(args.tensor_parallel_size),
                    "gpu_memory_utilization": _parse_float_or_string(
                        args.gpu_memory_utilization
                    ),
                    "gpu_headroom_gib": float(args.gpu_headroom_gib),
                    "gpu_workspace_headroom_gib": float(
                        args.gpu_workspace_headroom_gib
                    ),
                    "max_num_seqs": int(args.max_num_seqs),
                    "max_model_len": int(args.max_model_len),
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                    "performance_mode": "throughput",
                    "async_scheduling": True,
                    "seed": int(args.seed),
                },
            }
            suite, status = _generate_or_reuse(
                spec=spec,
                config=config,
                destination=generation_dir,
                benchmarks=group_benchmarks,
                num_responses=n,
                runtime_settings=runtime_settings,
                world_size=world_size,
            )
            for benchmark in group_benchmarks:
                correctness_by_benchmark[benchmark] = _prediction_correctness(
                    generation_dir, suite, benchmark, n
                )
                generation_dirs[benchmark] = str(generation_dir.resolve())
                cache_status[benchmark] = status
        k_by_benchmark = {
            "AIME24": aime_k,
            "AIME25": aime_k,
            "AMC23": amc_k,
        }
        rows = summarize_checkpoint_correctness(
            model_group=spec.model_group,
            method=spec.method,
            label=spec.label,
            checkpoint=spec.checkpoint,
            correctness_by_benchmark=correctness_by_benchmark,
            k_by_benchmark=k_by_benchmark,
            generation_dirs=generation_dirs,
        )
        all_rows.extend(rows)
        metadata = {
            "model_group": spec.model_group,
            "method": spec.method,
            "label": spec.label,
            "checkpoint": str(spec.checkpoint),
            "config": str(spec.config),
            "run_dir": str(run_dir.resolve()),
            "cache_status": cache_status,
            "results": rows,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "passk_results.json", metadata)
        run_metadata.append(metadata)

    all_rows.sort(
        key=lambda row: (
            row["model_group"],
            row["label"],
            row["benchmark"],
            int(row["k"]),
        )
    )
    summary = {
        "schema_version": PASSK_SCHEMA_VERSION,
        "experiment_tag": args.tag,
        "protocol": {
            "benchmarks": list(benchmarks),
            "aime_k_values": list(aime_k),
            "amc_k_values": list(amc_k),
            "aime_num_samples": aime_n,
            "amc_num_samples": amc_n,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
            "max_model_len": int(args.max_model_len),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "world_size": world_size,
            "seed": int(args.seed),
            "estimator": "1 - C(n-c,k) / C(n,k)",
        },
        "runs": run_metadata,
        "results": all_rows,
    }
    stem = f"passk_summary_{_slug(args.tag)}"
    json_path = summary_root / f"{stem}.json"
    csv_path = summary_root / f"{stem}.csv"
    _atomic_json(json_path, summary)
    _atomic_csv(csv_path, all_rows)
    print(f"Pass@K summary JSON: {json_path}")
    print(f"Pass@K summary CSV:  {csv_path}")
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cached multi-checkpoint Pass@K evaluation using existing vLLM grading"
    )
    parser.add_argument("--checkpoint-spec", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="Override paths.storage_root before resolving benchmark paths",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(SUPPORTED_BENCHMARKS))
    parser.add_argument("--aime-k-values", default="8 16 32 64 128")
    parser.add_argument("--amc-k-values", default="4 8 16 32 64")
    parser.add_argument("--aime-num-samples", type=int, default=128)
    parser.add_argument("--amc-num-samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=7168)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", default="auto")
    parser.add_argument("--gpu-headroom-gib", type=float, default=4.0)
    parser.add_argument("--gpu-workspace-headroom-gib", type=float, default=2.0)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sync-timeout-sec", type=float, default=86400.0)
    return parser


def main() -> int:
    run_experiment(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
