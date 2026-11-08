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
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ready", "smoke"])
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
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Also check local HF assets and exact tokenizer compatibility; no GPU model load",
    )
    args = parser.parse_args()
    config = configured(args.config, args.set)
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
