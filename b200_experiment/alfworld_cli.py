"""Readiness checks and an explicitly oracle-assisted simulator smoke test."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .alfworld_env import (
    SPLITS,
    EnvironmentClient,
    NativeEnvironment,
    load_task_splits,
    materialize_game,
)
from .config import apply_overrides, load_config, resolve_runtime_paths


class DiagnosticTokenizer:
    """Small reversible tokenizer for SCRIPTED simulator tests, never training."""

    eos_token_id, pad_token_id = 1, 0

    def __init__(self):
        self.vocabulary, self.reverse = {}, {}

    def encode(self, text, **kwargs):
        import re

        result = []
        for token in re.findall(r"\S+|\s+", text):
            if token not in self.vocabulary:
                key = len(self.vocabulary) + 2
                self.vocabulary[token], self.reverse[key] = key, token
            result.append(self.vocabulary[token])
        return result

    def decode(self, ids, **kwargs):
        return "".join(self.reverse[i] for i in ids if i > 1)

    def get_vocab(self):
        return self.vocabulary

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(f"<{m['role']}>\n{m['content']}<end>\n" for m in messages)
            + "<assistant>\n"
        )


def configured(path, overrides):
    config = load_config(path)
    root = os.environ.get("STORAGE_ROOT")
    if root:
        config["paths"]["storage_root"] = root
    return resolve_runtime_paths(apply_overrides(config, overrides))


def readiness(config, split="eval_in_distribution", task_index=0, *, smoke=False):
    tasks = load_task_splits(config)
    report = {
        "splits": {
            name: {
                "tasks": len(rows),
                "task_types": dict(Counter(r["task_type"] for r in rows)),
            }
            for name, rows in tasks.items()
        },
        "split_isolation": "passed: IDs, game paths and game content disjoint",
    }
    task = tasks[split][task_index]
    env = EnvironmentClient(task, config["alfworld"])
    try:
        initial = env.reset(42)
        result = env.step("look")
        report["simulator"] = {
            "task_id": task["id"],
            "initial_observation": initial["observation"],
            "look_observation": result["observation"],
            "admissible_actions": len(result["admissible_actions"]),
            "backend": "TextWorld PDDL, real subprocess reset/step",
            "native_site": initial.get("native_site"),
        }
    finally:
        env.close()
    if smoke:
        # Deliberate diagnostic ONLY: the training/evaluation loader strips
        # walkthroughs before any policy can see a task. No expert in rollouts.
        from .data import read_records

        source = (
            config["data"]["path"]
            if split == "train"
            else config["alfworld"]["eval_paths"][split]
        )
        rows, _ = read_records(source)
        row = rows[task_index]
        raw = row.get("game_content", {})
        raw = json.loads(raw) if isinstance(raw, str) else raw
        walkthrough = raw.get("walkthrough")
        if not walkthrough:
            raise ValueError(
                "Smoke requires a raw embedded walkthrough; readiness reset/step already succeeded. Use the published awawa-agi raw parquet."
            )
        native = NativeEnvironment()
        actions = []
        try:
            state = native.reset(
                materialize_game(task, config["alfworld"]["game_cache_dir"]), 42
            )
            aliases = native.env.aliases
            for raw_action in walkthrough:
                action = str(raw_action)
                for name in sorted(aliases, key=len, reverse=True):
                    action = action.replace(name, aliases[name])
                state = native.step(action)
                actions.append({"action": action, **state})
                if state["terminated"]:
                    break
            report["oracle_simulator_smoke"] = {
                "success": state["success"],
                "steps": actions,
                "purpose": "simulator installation verification only; not a policy evaluation",
            }
            if not state["success"]:
                raise RuntimeError(
                    "Real simulator smoke failed to complete the stored diagnostic walkthrough"
                )
        finally:
            native.close()
        from .alfworld_rollout import collect_episode, collate_episodes
        from .alfworld_diagnostics import episode_metrics

        tokenizer = DiagnosticTokenizer()
        script = iter(item["action"] for item in actions)

        def generate(ids, seed, budget):
            # Deliberately exercise the repaired trailing-semicolon parser.
            return tokenizer.encode(
                next(script)
                + (";" if config["alfworld"].get("protocol_version", 1) == 2 else "")
            ) + [1]

        episode = collect_episode(task, tokenizer, config, generate, 42)
        batch = collate_episodes([episode], 0, "cpu")
        if not episode["trace"]["success"] or int(batch.actor_token_mask.sum()) != int(
            batch.valid_mask.sum()
        ):
            raise RuntimeError("Parser/environment/actor-mask scripted smoke failed")
        report["parser_environment_smoke"] = {
            "metrics": episode_metrics([episode["trace"]]),
            "trace": episode["trace"],
            "purpose": "scripted diagnostic; not model task success",
        }
    return report


def probe_models(config, args):
    from .alfworld_evaluation import run_workers, _write_json
    from .alfworld_diagnostics import episode_metrics
    from .config import save_config
    from .models import inspect_model_assets
    from .vllm_evaluation import _exclusive_evaluation_gpu_locks

    if not args.output or args.limit <= 0:
        raise ValueError("probe requires --output NEW_DIRECTORY and --limit > 0")
    assets = inspect_model_assets(config)  # same student tokenizer for BOTH roles
    for field in ("temperature", "top_p"):
        config["rollout"][field] = config["evaluation"][field]
    from .alfworld_protocol import resolve_protocol

    config = resolve_protocol(config)
    config["alfworld_probe"] = {
        "role": args.role,
        "split": args.split,
        "limit": args.limit,
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = output / "resolved_config.yaml"
    save_config(config, cfg)
    model_path = config["models"][f"{args.role}_path"]
    devices = [d.strip() for d in args.gpus.split(",")]
    if not devices or any(not d for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("--gpus must contain distinct device identifiers")
    with _exclusive_evaluation_gpu_locks(devices):
        traces = run_workers(
            cfg, {"checkpoint": model_path, "step": 0}, args.seed, output, devices
        )
    expected = {
        task["id"] for task in load_task_splits(config)[args.split][: args.limit]
    }
    if len(traces) != len(expected) or {t["task_id"] for t in traces} != expected:
        raise RuntimeError("Probe returned incomplete/duplicate task results")
    result = {
        "role": args.role,
        "model": model_path,
        "seed": args.seed,
        "split": args.split,
        "protocol_version": config["alfworld"]["protocol_version"],
        "protocol_fingerprint": config["alfworld"]["protocol_fingerprint"],
        "metrics": episode_metrics(traces),
        "assets": assets,
    }
    _write_json(output / "summary.json", result)
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ready", "smoke", "probe"])
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs/alfworld_multiturn.yaml"
        ),
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--split", choices=SPLITS, default="eval_in_distribution")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--role", choices=["student", "teacher"], default="student")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Also check local HF assets and exact tokenizer compatibility; no GPU model load",
    )
    args = parser.parse_args()
    config = configured(args.config, args.set)
    if args.command == "probe":
        print(json.dumps(probe_models(config, args), ensure_ascii=False, indent=2))
        return 0
    result = readiness(
        config,
        args.split,
        args.task_index,
        smoke=args.command == "smoke",
    )
    if args.check_models:
        from .models import inspect_model_assets

        result["models"] = inspect_model_assets(config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
