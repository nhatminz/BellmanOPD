"""Fresh, standalone ALFWorld checkpoint evaluation; never reads math histories."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .alfworld_env import SPLITS, load_task_splits
from .config import apply_overrides, load_config, resolve_runtime_paths, save_config
from .multiseed_evaluation import checkpoint_step, resolve_run_output

METRICS = ("success_rate", "average_turns", "invalid_action_rate")


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


def metrics(traces):
    n = len(traces)
    turns = sum(len(t["turns"]) for t in traces)
    invalid = sum(t["invalid_actions"] for t in traces)
    return {
        "episodes": n,
        "successes": sum(bool(t["success"]) for t in traces),
        "turns": turns,
        "invalid_actions": invalid,
        "success_rate": sum(bool(t["success"]) for t in traces) / max(n, 1),
        "average_turns": turns / max(n, 1),
        "invalid_action_rate": invalid / max(turns, 1),
    }


def per_seed_results(traces):
    result = {}
    for split in SPLITS[1:]:
        rows = [t for t in traces if t["split"] == split]
        result[split] = {
            **metrics(rows),
            "task_types": {
                kind: metrics([t for t in rows if t["task_type"] == kind])
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


def aggregate(seed_results):
    """Sample standard deviation across seeds, not across individual tasks."""
    result = {}
    for split in SPLITS[1:]:
        rows = [s[split] for s in seed_results]
        kinds = sorted({kind for row in rows for kind in row["task_types"]})
        result[split] = {key: _mean_std([r[key] for r in rows]) for key in METRICS}
        result[split]["task_types"] = {
            kind: {
                key: _mean_std([r["task_types"][kind][key] for r in rows])
                for key in METRICS
            }
            for kind in kinds
        }
    return result


def build_plan(args):
    run = resolve_run_output(args.run_output)
    config = load_config(run / "resolved_config.yaml")
    if os.environ.get("STORAGE_ROOT"):
        config["paths"]["storage_root"] = os.environ["STORAGE_ROOT"]
    config = resolve_runtime_paths(apply_overrides(config, args.set))
    if not config.get("alfworld", {}).get("enabled", False):
        raise ValueError("Select an ALFWorld multi-turn training run, not a math run")
    method = config["experiment"]["method"]
    if method not in {"cmt", "opd", "ta", "iw"} or (
        args.method and args.method != method
    ):
        raise ValueError(f"Method does not match the selected run: {method}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be distinct non-negative integers")
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
            "ALFWorld evaluation executes one fresh episode per task/seed; set evaluation.num_responses=1"
        )
    return run, config, chosen


def evaluate_tasks(tasks, tokenizer, config, generate, seed, *, env_factory=None):
    from .alfworld_rollout import collect_episodes

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
    from .alfworld_rollout import collect_episode
    from .models import generation_stop_token_ids, load_student_tokenizer
    from .vllm_evaluation import _release_vllm_engine, _resolve_gpu_memory_utilization
    from vllm import LLM, SamplingParams

    config = load_config(args.worker_config)
    tasks = load_task_splits(config)
    selected = [
        task
        for split in SPLITS[1:]
        for i, task in enumerate(tasks[split])
        if i % args.world_size == args.rank
    ]
    config = copy.deepcopy(config)
    config["models"]["student_path"] = args.checkpoint
    tokenizer = load_student_tokenizer(config)
    stop_ids = generation_stop_token_ids(tokenizer, config)
    settings = config["evaluation"]
    vcfg = settings["vllm"]
    engine_kwargs = {
        "model": args.checkpoint,
        "tokenizer": args.checkpoint,
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
                episode = collect_episode(
                    task, tokenizer, config, generate, episode_seed(args.seed, task)
                )
                trace = {**episode["trace"], "evaluation_seed": args.seed}
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"ALFWorld seed={args.seed} rank={args.rank}: {i + 1}/{len(selected)} task={task['id']} success={trace['success']}",
                    flush=True,
                )
        trace_path.with_suffix(".done").write_text("complete\n", encoding="utf-8")
    finally:
        if engine is not None:
            _release_vllm_engine(engine)


def run_workers(config_path, selected, seed, directory, devices):
    from .distributed import isolate_distributed_subprocess_environment
    from .vllm_evaluation import _terminate_process_group

    processes, logs, paths = [], [], []
    try:
        for rank, device in enumerate(devices):
            path = directory / f"rank-{rank:05d}.jsonl"
            paths.append(path)
            log = (directory / f"rank-{rank:05d}.log").open("x", encoding="utf-8")
            logs.append(log)
            env = isolate_distributed_subprocess_environment()
            env["CUDA_VISIBLE_DEVICES"] = device
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
        while any(p.poll() is None for p in processes):
            for rank, process in enumerate(processes):
                if process.poll() not in (None, 0):
                    raise RuntimeError(
                        f"ALFWorld evaluation worker {rank} failed; see {directory / f'rank-{rank:05d}.log'}"
                    )
            # Subprocess management runs in the CLI, not in an agent tool wait.
            import time

            time.sleep(0.5)
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
            if process.poll() is None:
                _terminate_process_group(process)
        for log in logs:
            log.close()


def run_evaluation(args, *, worker_runner=run_workers):
    run, config, checkpoints = build_plan(args)
    splits = load_task_splits(config)
    expected = {(split, t["id"]) for split in SPLITS[1:] for t in splits[split]}
    plan = {
        "method": config["experiment"]["method"],
        "seeds": args.seeds,
        "checkpoints": checkpoints,
        "tasks": {s: len(splits[s]) for s in SPLITS[1:]},
        "fresh_episodes": True,
        "std_definition": "sample standard deviation across seeds (ddof=1)",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    devices = [device.strip() for device in args.gpus.split(",")]
    if (
        not devices
        or any(not d.strip() for d in devices)
        or len(devices) != len(set(devices))
    ):
        raise ValueError("--gpus must list distinct visible device identifiers")
    output = run / "alfworld_eval" / args.tag
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
                actual = [(t["split"], t["task_id"]) for t in traces]
                if len(actual) != len(set(actual)) or set(actual) != expected:
                    raise RuntimeError(
                        f"Missing/duplicate ID/OOD task results in {directory}; refusing incomplete summary"
                    )
                if any(t["evaluation_seed"] != seed for t in traces):
                    raise RuntimeError(
                        "Worker returned results for a different evaluation seed"
                    )
                traces.sort(key=lambda t: (t["split"], t["task_id"]))
                with (directory / "episodes.jsonl").open(
                    "x", encoding="utf-8"
                ) as handle:
                    for trace in traces:
                        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        traces_by_task.setdefault(
                            (trace["split"], trace["task_id"]), []
                        ).append(trace)
                results = per_seed_results(traces)
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
                                **{k: values[k] for k in METRICS},
                                "episodes": values["episodes"],
                            }
                        )
            result = {
                **selected,
                "seeds": dict(zip(map(str, args.seeds), seeds)),
                "aggregate": aggregate(seeds),
            }
            all_results.append(result)
            _write_json(checkpoint_dir / "summary.json", result)
            for (split, task_id), traces in sorted(traces_by_task.items()):
                task_rows.append(
                    {
                        "step": selected["step"],
                        "split": split,
                        "task_id": task_id,
                        "task_type": traces[0]["task_type"],
                        "seeds": args.seeds,
                        **{
                            key: _mean_std([metrics([trace])[key] for trace in traces])
                            for key in METRICS
                        },
                    }
                )
    summary = {**plan, "results": all_results, "output": str(output)}
    _write_json(output / "summary.json", summary)
    with (output / "per_seed.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "seed", "split", "task_type", *METRICS, "episodes"],
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
            ],
        )
        writer.writeheader()
        for result in all_results:
            for split, entry in result["aggregate"].items():
                for kind, values in [("ALL", entry), *entry["task_types"].items()]:
                    for key in METRICS:
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-output")
    result.add_argument("--method", choices=["cmt", "opd", "ta", "iw"])
    result.add_argument("--checkpoints", nargs="+")
    result.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    result.add_argument(
        "--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    result.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    result.add_argument("--set", action="append", default=[])
    result.add_argument("--dry-run", action="store_true")
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
    else:
        if not args.run_output or not args.checkpoints:
            arg_parser.error("--run-output and --checkpoints are required")
        run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
