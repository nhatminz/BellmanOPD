"""Fresh, standalone ALFWorld checkpoint evaluation; never reads math histories."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from .alfworld_env import SPLITS, load_task_splits
from .config import apply_overrides, load_config, resolve_runtime_paths, save_config
from .multiseed_evaluation import checkpoint_step, resolve_run_output
from .alfworld_protocol import resolve_protocol
from .alfworld_diagnostics import (
    RATE_METRICS,
    episode_metrics,
    merge_counts,
    metrics_from_counts,
)

METRICS = RATE_METRICS
SAMPLED_METRICS = ("avg_at_8", "pass_at_8")


def metric_names(episodes_per_task):
    return (*METRICS, *SAMPLED_METRICS) if episodes_per_task == 8 else METRICS


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def episode_seed(seed, task):
    # Stable per-task seeds: independent of GPU partition and scheduling order.
    offset = int(hashlib.sha256(task["id"].encode()).hexdigest()[:8], 16)
    return (int(seed) + offset) % (2**31 - 1)


def sample_seed(seed, task, sample_index):
    # Keep the old single-episode sampling seed for attempt zero. Other
    # attempts use independent policy streams, with the same simulator reset.
    return (episode_seed(seed, task) + int(sample_index) * 7919) % (2**31 - 1)


def metrics(traces, episodes_per_task=1):
    result = episode_metrics(traces)
    if episodes_per_task == 8:
        groups = defaultdict(list)
        for trace in traces:
            groups[(trace["split"], trace["task_id"])].append(trace)
        if any(
            sorted(t.get("sample_index", -1) for t in group) != list(range(8))
            for group in groups.values()
        ):
            raise ValueError(
                "avg@8/pass@8 require eight distinct episode samples per task and seed"
            )
        n = len(groups)
        result.update(
            tasks=n,
            avg_at_8=sum(bool(t["success"]) for t in traces) / (8 * n) if n else 0.0,
            pass_at_8=sum(any(t["success"] for t in group) for group in groups.values())
            / n
            if n
            else 0.0,
        )
    return result


def per_seed_results(traces, episodes_per_task=1):
    result = {}
    for split in SPLITS[1:]:
        rows = [t for t in traces if t["split"] == split]
        result[split] = {
            **metrics(rows, episodes_per_task),
            "task_types": {
                kind: metrics(
                    [t for t in rows if t["task_type"] == kind], episodes_per_task
                )
                for kind in sorted({t["task_type"] for t in rows})
            },
        }
    return result


def _mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n": len(values),
        "display": f"{mean:.6f} ± {std:.6f}",
    }


def aggregate(seed_results, episodes_per_task=1):
    """Sample standard deviation across seeds, not across individual tasks."""
    result = {}
    for split in SPLITS[1:]:
        rows = [s[split] for s in seed_results]
        kinds = sorted({kind for row in rows for kind in row["task_types"]})
        result[split] = _aggregate_entries(rows, metric_names(episodes_per_task))
        result[split]["task_types"] = {
            kind: _aggregate_entries(
                [r["task_types"][kind] for r in rows], metric_names(episodes_per_task)
            )
            for kind in kinds
        }
    return result


def _aggregate_entries(rows, names=METRICS):
    result = {key: _mean_std([r[key] for r in rows]) for key in names}
    for key in SAMPLED_METRICS:
        if key in result:
            entry = result[key]
            entry.update(
                percent_mean=100 * entry["mean"],
                percent_std=100 * entry["std"],
                display_percent=f"{100 * entry['mean']:.2f} ± {100 * entry['std']:.2f}",
            )
    reasons = sorted(
        {key for row in rows for key in row["termination_reason_distribution"]}
    )
    result["termination_reason_distribution"] = {
        key: _mean_std(
            [row["termination_reason_distribution"].get(key, 0) for row in rows]
        )
        for key in reasons
    }
    combined = metrics_from_counts(merge_counts([row["raw_counts"] for row in rows]))
    result["top_responses"] = combined["top_responses"]
    result["top_actions"] = combined["top_actions"]
    return result


def build_plan(args):
    run = resolve_run_output(args.run_output)
    config = load_config(run / "resolved_config.yaml")
    if not config.get("alfworld", {}).get("enabled", False):
        raise ValueError("Select an ALFWorld multi-turn training run, not a math run")
    original_protocol = resolve_protocol(config)["alfworld"]
    saved_fingerprint = config["alfworld"].get(
        "protocol_fingerprint", original_protocol["protocol_fingerprint"]
    )
    if os.environ.get("STORAGE_ROOT"):
        config["paths"]["storage_root"] = os.environ["STORAGE_ROOT"]
    config = apply_overrides(config, args.set)
    for field in ("temperature", "top_p"):
        config["rollout"][field] = config["evaluation"][field]
    config = resolve_runtime_paths(config)
    changed_protocol = saved_fingerprint != config["alfworld"]["protocol_fingerprint"]
    if changed_protocol and not getattr(args, "allow_protocol_change", False):
        raise ValueError(
            "Evaluation interaction protocol differs from the checkpoint's training protocol. For a separately labeled new-protocol diagnostic, pass --allow-protocol-change and a NEW --tag. Training resume across protocols is rejected."
        )
    config["alfworld"]["training_protocol_fingerprint"] = saved_fingerprint
    config["alfworld"]["evaluation_protocol_changed"] = changed_protocol
    if not config.get("alfworld", {}).get("enabled", False):
        raise ValueError("Select an ALFWorld multi-turn training run, not a math run")
    method = config["experiment"]["method"]
    if method not in {"cmt", "opd", "ta", "iw"} or (
        args.method and args.method != method
    ):
        raise ValueError(f"Method does not match the selected run: {method}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be distinct non-negative integers")
    episodes_per_task = int(getattr(args, "episodes_per_task", 1))
    if episodes_per_task not in {1, 8}:
        raise ValueError(
            "--episodes-per-task must be 1 (legacy SR) or 8 (avg@8/pass@8)"
        )
    if episodes_per_task == 8 and (
        len(args.seeds) < 2 or float(config["evaluation"]["temperature"]) <= 0
    ):
        raise ValueError(
            "Sampled avg@8/pass@8 needs temperature > 0 and at least two seeds for sample std"
        )
    config["evaluation"]["episodes_per_task"] = episodes_per_task
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        raise ValueError("--tag must be one filename component")
    chosen, seen = [], set()
    for value in args.checkpoints:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = run / (
                f"checkpoint-{int(value):06d}" if value.isdigit() else value
            )
        checkpoint = candidate.resolve()
        if checkpoint.parent != run or not (checkpoint / "config.json").is_file():
            raise ValueError(
                f"Expected a saved checkpoint directly inside {run}: {checkpoint}"
            )
        step = checkpoint_step(checkpoint, run)
        if step in seen:
            raise ValueError(f"Duplicate checkpoint step {step}")
        seen.add(step)
        chosen.append({"checkpoint": str(checkpoint), "step": step})
    if int(config["alfworld"]["max_episode_tokens"]) > int(
        config["evaluation"]["vllm"]["max_model_len"]
    ):
        raise ValueError(
            "evaluation.vllm.max_model_len must cover alfworld.max_episode_tokens"
        )
    if int(config["evaluation"].get("num_responses", 1)) != 1:
        raise ValueError(
            "ALFWorld generates one action per turn; set evaluation.num_responses=1 and use --episodes-per-task 8 for avg@8/pass@8"
        )
    return run, config, chosen


def _evaluate_episode(
    task, tokenizer, config, generate, seed, sample_index, *, env_factory=None
):
    from .alfworld_rollout import collect_episode

    kwargs = {} if env_factory is None else {"env_factory": env_factory}
    episode = collect_episode(
        task,
        tokenizer,
        config,
        generate,
        sample_seed(seed, task, sample_index),
        environment_seed=episode_seed(seed, task),
        **kwargs,
    )
    return {**episode["trace"], "evaluation_seed": seed, "sample_index": sample_index}


def evaluate_tasks(
    tasks, tokenizer, config, generate, seed, *, env_factory=None, episodes_per_task=1
):
    from .alfworld_rollout import collect_episodes

    if episodes_per_task == 8:
        return [
            _evaluate_episode(
                task,
                tokenizer,
                config,
                generate,
                seed,
                sample_index,
                env_factory=env_factory,
            )
            for task in tasks
            for sample_index in range(8)
        ]
    if episodes_per_task != 1:
        raise ValueError("episodes_per_task must be 1 or 8")
    kwargs = {} if env_factory is None else {"env_factory": env_factory}
    episodes = collect_episodes(
        tasks,
        tokenizer,
        config,
        generate,
        [episode_seed(seed, t) for t in tasks],
        **kwargs,
    )
    return [{**e["trace"], "evaluation_seed": seed} for e in episodes]


def worker(args):
    from .models import generation_stop_token_ids, load_student_tokenizer
    from .vllm_evaluation import _release_vllm_engine, _resolve_gpu_memory_utilization
    from vllm import LLM, SamplingParams

    config = load_config(args.worker_config)
    tasks = load_task_splits(config)
    probe = config.get("alfworld_probe")
    selected_splits = [probe["split"]] if probe else SPLITS[1:]
    selected = [
        task
        for split in selected_splits
        for i, task in enumerate(
            tasks[split][: probe["limit"]] if probe else tasks[split]
        )
        if i % args.world_size == args.rank
    ]
    config = copy.deepcopy(config)
    if not probe:
        config["models"]["student_path"] = args.checkpoint
    tokenizer = load_student_tokenizer(config)
    stop_ids = generation_stop_token_ids(tokenizer, config)
    settings = config["evaluation"]
    episodes_per_task = 1 if probe else int(settings.get("episodes_per_task", 1))
    vcfg = settings["vllm"]
    engine_kwargs = {
        "model": args.checkpoint,
        "tokenizer": config["models"]["student_path"],
        "dtype": config["models"].get("dtype", "bfloat16"),
        "trust_remote_code": True,
        "generation_config": "vllm",
        "tensor_parallel_size": 1,
        "seed": args.seed,
        "gpu_memory_utilization": _resolve_gpu_memory_utilization(vcfg),
        "max_model_len": int(vcfg["max_model_len"]),
        "max_num_seqs": int(vcfg.get("max_num_seqs", 16)),
        "enable_prefix_caching": bool(vcfg.get("enable_prefix_caching", True)),
    }
    for name in (
        "enable_chunked_prefill",
        "async_scheduling",
        "performance_mode",
        "max_num_batched_tokens",
    ):
        if vcfg.get(name) is not None:
            engine_kwargs[name] = vcfg[name]
    engine = None
    try:
        engine = LLM(**engine_kwargs)

        def generate(ids, seed, budget):
            sampling = SamplingParams(
                n=1,
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                max_tokens=budget,
                seed=seed,
                stop_token_ids=stop_ids,
            )
            outputs = engine.generate(
                [{"prompt_token_ids": ids}], sampling, use_tqdm=False
            )
            return list(outputs[0].outputs[0].token_ids)

        # Offline LLM.generate is not thread-safe. Each GPU runs an independent
        # replica and serial real episodes; no shared mutable engine in threads.
        trace_path = Path(args.worker_output)
        with trace_path.open("x", encoding="utf-8") as handle:
            for i, task in enumerate(selected):
                for sample_index in range(episodes_per_task):
                    trace = _evaluate_episode(
                        task, tokenizer, config, generate, args.seed, sample_index
                    )
                    handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"ALFWorld seed={args.seed} rank={args.rank}: "
                        f"{i * episodes_per_task + sample_index + 1}/{len(selected) * episodes_per_task} "
                        f"task={task['id']} sample={sample_index + 1}/{episodes_per_task} success={trace['success']}",
                        flush=True,
                    )
        trace_path.with_suffix(".done").write_text("complete\n", encoding="utf-8")
    finally:
        if engine is not None:
            _release_vllm_engine(engine)


def run_workers(config_path, selected, seed, directory, devices):
    from .distributed import isolate_distributed_subprocess_environment
    from .vllm_evaluation import _terminate_process_group

    cfg = load_config(config_path)
    splits = load_task_splits(cfg)
    probe = cfg.get("alfworld_probe")
    total = (
        len(splits[probe["split"]][: probe["limit"]])
        if probe
        else sum(len(splits[s]) for s in SPLITS[1:])
        * int(cfg["evaluation"].get("episodes_per_task", 1))
    )
    progress = tqdm(
        total=total,
        desc=f"ALFWorld {Path(selected['checkpoint']).name} seed {seed}",
        unit="episode",
        dynamic_ncols=True,
        position=1,
    )
    processes, logs, paths, readers, buffers = [], [], [], [], []

    def update_progress():
        for rank, path in enumerate(paths):
            if readers[rank] is None and path.is_file():
                readers[rank] = path.open("rb")
            if readers[rank] is not None:
                buffers[rank] += readers[rank].read()
                complete = buffers[rank].count(b"\n")
                if complete:
                    progress.update(complete)
                    buffers[rank] = buffers[rank].rsplit(b"\n", 1)[-1]

    try:
        for rank, device in enumerate(devices):
            path = directory / f"rank-{rank:05d}.jsonl"
            paths.append(path)
            readers.append(None)
            buffers.append(b"")
            log = (directory / f"rank-{rank:05d}.log").open("x", encoding="utf-8")
            logs.append(log)
            env = isolate_distributed_subprocess_environment()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                "-m",
                "b200_experiment.alfworld_evaluation",
                "--worker",
                "--worker-config",
                str(config_path),
                "--checkpoint",
                selected["checkpoint"],
                "--seed",
                str(seed),
                "--rank",
                str(rank),
                "--world-size",
                str(len(devices)),
                "--worker-output",
                str(path),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        timeout = float(cfg["evaluation"].get("sync_timeout_sec", 86400))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("evaluation.sync_timeout_sec must be positive and finite")
        deadline = time.monotonic() + timeout
        while any(p.poll() is None for p in processes):
            update_progress()
            for rank, process in enumerate(processes):
                if process.poll() not in (None, 0):
                    raise RuntimeError(
                        f"ALFWorld evaluation worker {rank} failed; see {directory / f'rank-{rank:05d}.log'}"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ALFWorld evaluation worker timeout; inspect logs in {directory}"
                )
            time.sleep(0.5)
        update_progress()
        for rank, process in enumerate(processes):
            if (
                process.returncode != 0
                or not paths[rank].with_suffix(".done").is_file()
            ):
                raise RuntimeError(
                    f"Incomplete ALFWorld worker {rank}; inspect logs in {directory}"
                )
        return [
            json.loads(line)
            for path in paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    finally:
        for process in processes:
            _terminate_process_group(process)
        for reader in readers:
            if reader is not None:
                reader.close()
        for log in logs:
            log.close()
        progress.close()


def run_evaluation(args, *, worker_runner=run_workers):
    run, config, checkpoints = build_plan(args)
    devices = [device.strip() for device in args.gpus.split(",")]
    if not devices or any(not d for d in devices) or len(devices) != len(set(devices)):
        raise ValueError("--gpus must list distinct visible device identifiers")
    splits = load_task_splits(config)
    task_lookup = {
        (split, task["id"]): task for split in SPLITS[1:] for task in splits[split]
    }
    episodes_per_task = config["evaluation"]["episodes_per_task"]
    names = metric_names(episodes_per_task)
    expected = {
        (split, t["id"], sample_index)
        for split in SPLITS[1:]
        for t in splits[split]
        for sample_index in range(episodes_per_task)
    }
    plan = {
        "method": config["experiment"]["method"],
        "seeds": args.seeds,
        "checkpoints": checkpoints,
        "tasks": {s: len(splits[s]) for s in SPLITS[1:]},
        "fresh_episodes": True,
        "episodes_per_task": episodes_per_task,
        "episodes_per_seed": len(expected),
        "total_episodes": len(expected) * len(args.seeds) * len(checkpoints),
        "gpus": devices,
        "metric_names": list(names),
        "sample_protocol": "Each task is reset to the same environment seed for all attempts within an evaluation seed; each attempt has a distinct policy sampling seed.",
        "protocol_version": config["alfworld"]["protocol_version"],
        "protocol_fingerprint": config["alfworld"]["protocol_fingerprint"],
        "protocol_manifest": config["alfworld"]["protocol_manifest"],
        "training_protocol_fingerprint": config["alfworld"][
            "training_protocol_fingerprint"
        ],
        "evaluation_protocol_changed": config["alfworld"][
            "evaluation_protocol_changed"
        ],
        "std_definition": "sample standard deviation across seeds (ddof=1)",
        "training_only_metrics": {
            "student_teacher_topk_overlap": None,
            "student_topk_mass": None,
            "teacher_topk_mass": None,
            "gradient_norm": None,
        },
        "training_only_metrics_note": "Standalone episode eval loads no teacher and performs no backward; these diagnostics are recorded in training metrics/TensorBoard, not fabricated at inference.",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    output = run / "alfworld_eval" / args.tag
    if getattr(args, "checkpoint_subdir", None):
        output = output / args.checkpoint_subdir
    output.mkdir(parents=True, exist_ok=False)  # An existing namespace is never reused.
    _write_json(output / "plan.json", plan)
    config_path = output / "resolved_config.yaml"
    save_config(config, config_path)
    all_results, csv_rows, task_rows = [], [], []
    from .vllm_evaluation import _exclusive_evaluation_gpu_locks

    with _exclusive_evaluation_gpu_locks(devices):
        for selected in checkpoints:
            checkpoint_dir = output / f"step-{selected['step']:06d}"
            checkpoint_dir.mkdir()
            seeds, traces_by_task = [], {}
            for seed in args.seeds:
                directory = checkpoint_dir / f"seed-{seed}"
                directory.mkdir()
                print(
                    f"Evaluating {selected['checkpoint']} seed={seed}; logs: {directory}",
                    flush=True,
                )
                traces = worker_runner(config_path, selected, seed, directory, devices)
                actual = [
                    (t["split"], t["task_id"], t.get("sample_index", 0)) for t in traces
                ]
                if len(actual) != len(set(actual)) or set(actual) != expected:
                    raise RuntimeError(
                        f"Missing/duplicate ID/OOD task/sample results in {directory}; refusing incomplete summary"
                    )
                if any(t["evaluation_seed"] != seed for t in traces):
                    raise RuntimeError(
                        "Worker returned results for a different evaluation seed"
                    )
                if episodes_per_task == 8 and any(
                    t.get("seed")
                    != sample_seed(
                        seed, task_lookup[(t["split"], t["task_id"])], t["sample_index"]
                    )
                    or t.get("environment_seed")
                    != episode_seed(seed, task_lookup[(t["split"], t["task_id"])])
                    for t in traces
                ):
                    raise RuntimeError(
                        "Worker returned incorrect policy sampling/reset seeds for avg@8/pass@8"
                    )
                if any(
                    t.get("protocol_fingerprint") != plan["protocol_fingerprint"]
                    for t in traces
                ):
                    raise RuntimeError(
                        "Worker episode protocol fingerprint differs from this evaluation; refusing mixed-protocol summary"
                    )
                traces.sort(
                    key=lambda t: (t["split"], t["task_id"], t.get("sample_index", 0))
                )
                with (directory / "episodes.jsonl").open(
                    "x", encoding="utf-8"
                ) as handle:
                    for trace in traces:
                        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        traces_by_task.setdefault(
                            (trace["split"], trace["task_id"]), []
                        ).append(trace)
                results = per_seed_results(traces, episodes_per_task)
                seeds.append(results)
                _write_json(
                    directory / "summary.json",
                    {"seed": seed, **selected, "splits": results},
                )
                for split, entry in results.items():
                    for kind, values in [("ALL", entry), *entry["task_types"].items()]:
                        csv_rows.append(
                            {
                                "step": selected["step"],
                                "seed": seed,
                                "split": split,
                                "task_type": kind,
                                **{k: values[k] for k in names},
                                "episodes": values["episodes"],
                            }
                        )
            result = {
                **selected,
                "seeds": dict(zip(map(str, args.seeds), seeds)),
                "aggregate": aggregate(seeds, episodes_per_task),
            }
            all_results.append(result)
            _write_json(checkpoint_dir / "summary.json", result)
            for (split, task_id), traces in sorted(traces_by_task.items()):
                task_seed_metrics = [
                    metrics(
                        [t for t in traces if t["evaluation_seed"] == seed],
                        episodes_per_task,
                    )
                    for seed in args.seeds
                ]
                task_rows.append(
                    {
                        "step": selected["step"],
                        "split": split,
                        "task_id": task_id,
                        "task_type": traces[0]["task_type"],
                        "seeds": args.seeds,
                        **{
                            key: _mean_std(
                                [values[key] for values in task_seed_metrics]
                            )
                            for key in names
                        },
                    }
                )
    summary = {**plan, "results": all_results, "output": str(output)}
    _write_json(output / "summary.json", summary)
    with (output / "per_seed.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "seed", "split", "task_type", *names, "episodes"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    with (output / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "split",
                "task_type",
                "metric",
                "mean",
                "std",
                "n",
                "display",
                "percent_mean",
                "percent_std",
                "display_percent",
            ],
        )
        writer.writeheader()
        for result in all_results:
            for split, entry in result["aggregate"].items():
                for kind, values in [("ALL", entry), *entry["task_types"].items()]:
                    for key in names:
                        writer.writerow(
                            {
                                "step": result["step"],
                                "split": split,
                                "task_type": kind,
                                "metric": key,
                                **values[key],
                            }
                        )
    with (output / "task_results.jsonl").open("x", encoding="utf-8") as handle:
        for row in task_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(output / "complete.json", {"status": "complete"})
    print(f"ALFWorld summary: {output / 'summary.csv'}", flush=True)
    return summary


def _checkpoint_path(value):
    root = Path(
        os.environ.get("STORAGE_ROOT", "/workspace/storage-shared")
    ).expanduser()
    path = Path(value).expanduser()
    if not path.is_absolute():
        if str(path).startswith("workspace/storage-shared/"):
            path = Path("/") / path
        elif path.parts and path.parts[0] == root.name:
            path = root.parent / path
        else:
            path = root / path
    return path.resolve()


def _write_batch_csv(path, results):
    fields = [
        "checkpoint",
        "method",
        "step",
        "split",
        "task_type",
        "metric",
        "mean",
        "std",
        "n",
        "display",
        "percent_mean",
        "percent_std",
        "display_percent",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for completed in results:
            for result in completed["results"]:
                for split, entry in result["aggregate"].items():
                    for kind, values in [("ALL", entry), *entry["task_types"].items()]:
                        for name in SAMPLED_METRICS:
                            if name in values:
                                writer.writerow(
                                    {
                                        "checkpoint": result["checkpoint"],
                                        "method": completed["method"],
                                        "step": result["step"],
                                        "split": split,
                                        "task_type": kind,
                                        "metric": name,
                                        **values[name],
                                    }
                                )
    temporary.replace(path)


def run_checkpoint_list(args, *, worker_runner=run_workers):
    """Evaluate arbitrary runs in user order, keeping errors local to a checkpoint."""
    paths = [_checkpoint_path(value) for value in args.checkpoint_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Checkpoint list contains duplicate paths")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        raise ValueError("--tag must be one filename component")
    output = Path(args.batch_output_root).expanduser().resolve() / args.tag
    plan = {
        "checkpoints": list(map(str, paths)),
        "seeds": args.seeds,
        "episodes_per_task": args.episodes_per_task,
        "gpus": args.gpus,
        "output": str(output),
        "tag": args.tag,
    }
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "plan.json", plan)
    results, failures, dry_plans = [], [], []
    for path in tqdm(
        paths, desc="ALFWorld checkpoints", unit="checkpoint", dynamic_ncols=True
    ):
        try:
            selected = copy.deepcopy(args)
            selected.run_output = str(path.parent)
            selected.checkpoints = [str(path)]
            selected.checkpoint_subdir = path.name
            print(f"ALFWorld checkpoint: {path}", flush=True)
            summary = run_evaluation(selected, worker_runner=worker_runner)
            if args.dry_run:
                dry_plans.append({"checkpoint": str(path), "plan": summary})
            else:
                results.append(
                    {
                        "checkpoint": str(path),
                        "method": summary["method"],
                        "summary": str(Path(summary["output"]) / "summary.json"),
                        "results": summary["results"],
                    }
                )
        except Exception as error:
            traceback.print_exc()
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED, continuing: {path}", flush=True)
        if not args.dry_run:
            _write_json(
                output / "summary.json",
                {**plan, "status": "running", "results": results, "failures": failures},
            )
            _write_batch_csv(output / "summary.csv", results)
    combined = {
        **plan,
        "status": "complete_with_failures" if failures else "complete",
        "results": results,
        "failures": failures,
    }
    if args.dry_run:
        combined["plans"] = dry_plans
    else:
        _write_json(output / "summary.json", combined)
        print(f"ALFWorld combined summary: {output / 'summary.csv'}", flush=True)
    return combined


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-output")
    result.add_argument("--method", choices=["cmt", "opd", "ta", "iw"])
    result.add_argument("--checkpoints", nargs="+")
    result.add_argument(
        "--checkpoint-paths",
        nargs="+",
        help="Checkpoint directories from any ALFWorld runs, evaluated sequentially",
    )
    result.add_argument(
        "--episodes-per-task",
        type=int,
        choices=[1, 8],
        default=1,
        help="8 fresh interactive episodes per task/seed for avg@8 and pass@8; legacy default 1",
    )
    result.add_argument(
        "--batch-output-root",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "alfworld_eval"),
    )
    result.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    result.add_argument(
        "--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    result.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    result.add_argument("--set", action="append", default=[])
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--allow-protocol-change",
        action="store_true",
        help="Explicitly re-evaluate a legacy checkpoint under a changed interaction protocol, in a new output namespace; never changes the checkpoint",
    )
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--worker-config", help=argparse.SUPPRESS)
    result.add_argument("--checkpoint", help=argparse.SUPPRESS)
    result.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    result.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    result.add_argument("--world-size", type=int, default=1, help=argparse.SUPPRESS)
    result.add_argument("--worker-output", help=argparse.SUPPRESS)
    return result


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.worker:
        worker(args)
    elif args.checkpoint_paths:
        if args.run_output or args.checkpoints:
            arg_parser.error("Use --checkpoint-paths OR --run-output/--checkpoints")
        return int(bool(run_checkpoint_list(args)["failures"]))
    else:
        if not args.run_output or not args.checkpoints:
            arg_parser.error("--run-output and --checkpoints are required")
        run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
