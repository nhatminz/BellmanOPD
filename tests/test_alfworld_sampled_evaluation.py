import copy
import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from b200_experiment import alfworld_evaluation as evaluation
from b200_experiment.alfworld_env import SPLITS, load_task_splits
from b200_experiment.config import load_config, save_config
import test_alfworld_evaluation as legacy_tests
from test_alfworld_evaluation import args_for, trace
from test_alfworld_multiturn import Tokenizer, InteractiveEnv

alfconfig = legacy_tests.alfconfig
saved_run = legacy_tests.saved_run


def sample_trace(task, seed, index, *, won=True, fingerprint=None):
    return {
        **trace(task, seed, won),
        "sample_index": index,
        "seed": evaluation.sample_seed(seed, task, index),
        "environment_seed": evaluation.episode_seed(seed, task),
        "protocol_fingerprint": fingerprint,
    }


def runner_for(config, counts=None):
    tasks = load_task_splits(config)
    counts = {42: 1, 43: 0, 44: 8} if counts is None else counts

    def runner(path, selected, seed, directory, devices):
        return [
            sample_trace(
                task,
                seed,
                index,
                won=index < counts[seed],
                fingerprint=config["alfworld"]["protocol_fingerprint"],
            )
            for split in SPLITS[1:]
            for task in tasks[split]
            for index in range(8)
        ]

    return runner


def test_metrics_average_eight_successes_and_pass_any_with_question_denominator():
    first = {"id": "a", "split": SPLITS[1], "task_type": "pick"}
    second = {"id": "b", "split": SPLITS[1], "task_type": "clean"}
    traces = [
        sample_trace(task, 42, i, won=task == first and i == 0)
        for task in [first, second]
        for i in range(8)
    ]
    result = evaluation.per_seed_results(traces, 8)[SPLITS[1]]
    assert result["episodes"] == 16 and result["tasks"] == 2
    assert result["avg_at_8"] == result["success_rate"] == 1 / 16
    assert result["pass_at_8"] == 1 / 2
    assert result["task_types"]["pick"]["avg_at_8"] == 1 / 8
    assert result["task_types"]["clean"]["pass_at_8"] == 0


def test_eight_attempts_have_fresh_environments_same_reset_seed_and_unique_policy_seeds(
    alfconfig,
):
    task = load_task_splits(alfconfig)[SPLITS[1]][0]
    tokenizer = Tokenizer()
    before = len(InteractiveEnv.instances)
    generate_seeds = []

    def generate(ids, seed, budget):
        text = tokenizer.decode(ids)
        action = (
            "take cup 1 from cabinet 1"
            if "Cabinet is open" in text
            else "open cabinet 1"
        )
        generate_seeds.append(seed)
        return tokenizer.encode(action) + [1]

    traces = evaluation.evaluate_tasks(
        [task],
        tokenizer,
        alfconfig,
        generate,
        42,
        env_factory=InteractiveEnv,
        episodes_per_task=8,
    )
    instances = InteractiveEnv.instances[before:]
    assert len(instances) == 8 and all(env.closed for env in instances)
    assert len({env.seed for env in instances}) == 1
    assert {trace["sample_index"] for trace in traces} == set(range(8))
    assert len({trace["seed"] for trace in traces}) == 8
    assert len(set(generate_seeds[::2])) == 8
    assert all(trace["success"] for trace in traces)
    assert traces[0]["seed"] == evaluation.episode_seed(42, task)


def test_avg_and_pass_sample_std_are_across_three_seeds(saved_run, alfconfig):
    args = args_for(saved_run, checkpoints=["50"], episodes_per_task=8)
    result = evaluation.run_evaluation(args, worker_runner=runner_for(alfconfig))
    aggregate = result["results"][0]["aggregate"][SPLITS[1]]
    assert aggregate["avg_at_8"]["mean"] == 0.375
    assert aggregate["avg_at_8"]["std"] == statistics.stdev([1 / 8, 0, 1])
    assert aggregate["pass_at_8"]["mean"] == pytest.approx(2 / 3)
    assert aggregate["pass_at_8"]["std"] == statistics.stdev([1, 0, 1])
    assert aggregate["pass_at_8"]["n"] == 3
    output = Path(result["output"])
    rows = list(csv.DictReader((output / "summary.csv").open()))
    assert any(
        row["metric"] == "avg_at_8" and row["display_percent"].startswith("37.50 ±")
        for row in rows
    )
    task_rows = [
        json.loads(line)
        for line in (output / "task_results.jsonl").read_text().splitlines()
    ]
    assert all(row["avg_at_8"]["n"] == 3 for row in task_rows)  # not 24 pooled episodes
    assert (
        len((output / "step-000050/seed-42/episodes.jsonl").read_text().splitlines())
        == 16
    )


@pytest.mark.parametrize(
    "error", ["missing", "duplicate", "wrong_seed", "wrong_reset", "wrong_protocol"]
)
def test_incomplete_or_mixed_samples_never_publish_summary(saved_run, alfconfig, error):
    generate = runner_for(alfconfig)

    def broken(*args):
        traces = generate(*args)
        if error == "missing":
            traces.pop()
        elif error == "duplicate":
            traces[-1] = traces[-2]
        elif error == "wrong_seed":
            traces[-1]["seed"] += 1
        elif error == "wrong_reset":
            traces[-1]["environment_seed"] += 1
        else:
            traces[-1]["protocol_fingerprint"] = "different"
        return traces

    with pytest.raises(RuntimeError):
        evaluation.run_evaluation(
            args_for(saved_run, checkpoints=["50"], episodes_per_task=8),
            worker_runner=broken,
        )
    assert not (saved_run / "alfworld_eval/test-fresh/complete.json").exists()


def batch_args(tmp_path, paths, **kwargs):
    args = evaluation.parser().parse_args(
        [
            "--checkpoint-paths",
            *map(str, paths),
            "--episodes-per-task",
            "8",
            "--gpus",
            "0,1",
            "--batch-output-root",
            str(tmp_path / "combined"),
            "--tag",
            "batch",
        ]
    )
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def test_batch_checkpoints_same_and_different_methods_in_order_continue_after_failure(
    saved_run, alfconfig, tmp_path
):
    other = tmp_path / "other_run" / "ta_opd"
    other.mkdir(parents=True)
    config = copy.deepcopy(alfconfig)
    config["experiment"]["method"] = "ta"
    save_config(config, other / "resolved_config.yaml")
    checkpoint = other / "checkpoint-000050"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    paths = [
        saved_run / "checkpoint-000050",
        saved_run / "missing",
        checkpoint,
        saved_run / "final",
    ]
    args = batch_args(tmp_path, paths)
    calls = []
    generate = runner_for(alfconfig)

    def runner(config, selected, seed, directory, devices):
        calls.append((selected["checkpoint"], seed))
        return generate(config, selected, seed, directory, devices)

    result = evaluation.run_checkpoint_list(args, worker_runner=runner)
    assert result["status"] == "complete_with_failures" and len(result["failures"]) == 1
    assert [row["checkpoint"] for row in result["results"]] == list(
        map(str, [paths[0], paths[2], paths[3]])
    )
    assert calls == [
        (str(path), seed)
        for path in [paths[0], paths[2], paths[3]]
        for seed in [42, 43, 44]
    ]
    assert [row["method"] for row in result["results"]] == ["cmt", "ta", "cmt"]
    assert (saved_run / "alfworld_eval/batch/checkpoint-000050/summary.json").is_file()
    assert (saved_run / "alfworld_eval/batch/final/summary.json").is_file()
    assert (other / "alfworld_eval/batch/checkpoint-000050/summary.json").is_file()
    csv_rows = list(csv.DictReader((Path(result["output"]) / "summary.csv").open()))
    assert {row["metric"] for row in csv_rows} == {"avg_at_8", "pass_at_8"}
    assert all(int(row["n"]) == 3 for row in csv_rows)
    with pytest.raises(FileExistsError):
        evaluation.run_checkpoint_list(args, worker_runner=runner)


def test_batch_dry_run_checks_eight_samples_without_loading_models_or_writing(
    saved_run, tmp_path
):
    args = batch_args(tmp_path, [saved_run / "checkpoint-000050"], dry_run=True)
    result = evaluation.run_checkpoint_list(
        args, worker_runner=lambda *a: pytest.fail("No model load in dry run")
    )
    assert not result["failures"]
    assert result["plans"][0]["plan"]["episodes_per_task"] == 8
    assert result["plans"][0]["plan"]["episodes_per_seed"] == 16
    assert result["plans"][0]["plan"]["total_episodes"] == 48
    assert result["plans"][0]["plan"]["seeds"] == [42, 43, 44]
    assert (
        not Path(result["output"]).exists()
        and not (saved_run / "alfworld_eval").exists()
    )


@pytest.mark.parametrize(
    "value",
    ["storage-shared/nlp/file", "workspace/storage-shared/nlp/file", "nlp/file"],
)
def test_user_checkpoint_path_without_leading_slash(monkeypatch, value):
    monkeypatch.setenv("STORAGE_ROOT", "/workspace/storage-shared")
    assert evaluation._checkpoint_path(value) == Path(
        "/workspace/storage-shared/nlp/file"
    )


def test_shell_actual_list_mode_and_legacy_flags(saved_run, tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHON_BIN": sys.executable, "ALFWORLD_EVAL_GPUS": "0,1"}
    script = str(root / "scripts/eval_alfworld_multiseed.sh")
    result = subprocess.run(
        [
            "bash",
            script,
            "--checkpoint-paths",
            str(saved_run / "checkpoint-000050"),
            "--batch-output-root",
            str(tmp_path / "combined"),
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"episodes_per_task": 8' in result.stdout
    assert '"avg_at_8"' in result.stdout and '"pass_at_8"' in result.stdout
    assert not (tmp_path / "combined").exists()
    legacy = subprocess.run(
        [
            "bash",
            script,
            "--run-output",
            str(saved_run),
            "--checkpoints",
            "50",
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert legacy.returncode == 0, legacy.stderr
    assert json.loads(legacy.stdout)["episodes_per_task"] == 1


def test_parent_tqdm_counts_complete_worker_episodes_and_gpu_masks(
    alfconfig, monkeypatch, tmp_path
):
    config = copy.deepcopy(alfconfig)
    config["evaluation"]["episodes_per_task"] = 8
    path = tmp_path / "worker.yaml"
    save_config(config, path)
    directory = tmp_path / "seed42"
    directory.mkdir()
    selected = {"checkpoint": "/saved/checkpoint-000400", "step": 400}
    tasks = load_task_splits(config)
    progress = SimpleNamespace(total=0, n=0, close=lambda: None)

    def bar(*args, **kwargs):
        progress.total = kwargs["total"]
        progress.update = lambda count: setattr(progress, "n", progress.n + count)
        return progress

    launched, terminated = [], []

    def popen(command, **kwargs):
        rank = int(command[command.index("--rank") + 1])
        destination = Path(command[command.index("--worker-output") + 1])
        traces = [
            sample_trace(task, 42, i)
            for split in SPLITS[1:]
            for index, task in enumerate(tasks[split])
            if index % 2 == rank
            for i in range(8)
        ]
        destination.write_text("".join(json.dumps(t) + "\n" for t in traces))
        destination.with_suffix(".done").write_text("done")
        process = SimpleNamespace(poll=lambda: 0, returncode=0)
        launched.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        return process

    monkeypatch.setattr(evaluation, "tqdm", bar)
    monkeypatch.setattr(evaluation.subprocess, "Popen", popen)
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._terminate_process_group", terminated.append
    )
    traces = evaluation.run_workers(path, selected, 42, directory, ["2", "3"])
    assert len(traces) == progress.n == progress.total == 16
    assert launched == ["2", "3"] and len(terminated) == 2


@pytest.mark.skipif(
    not os.environ.get("ALFWORLD_REAL_DATA"), reason="Optional real simulator/parquet"
)
def test_real_scripted_id_ood_avg8_pass8_three_seed_summary(tmp_path):
    from b200_experiment.alfworld_cli import readiness, DiagnosticTokenizer

    root = Path(os.environ["ALFWORLD_REAL_DATA"])
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/alfworld_multiturn.yaml"
    )
    config["data"]["path"] = str(root / "train-00000-of-00001.parquet")
    config["alfworld"]["eval_paths"] = {
        s: str(root / f"{s}-00000-of-00001.parquet") for s in SPLITS[1:]
    }
    config["alfworld"]["game_cache_dir"] = str(tmp_path / "games")
    tasks = load_task_splits(config)
    # Explicit oracle-assisted scripted mechanics check; never an LLM score.
    scripts = {
        split: [
            step["action"]
            for step in readiness(config, split=split, smoke=True)[
                "oracle_simulator_smoke"
            ]["steps"]
        ]
        for split in SPLITS[1:]
    }
    results = []
    for seed in [42, 43, 44]:
        traces = []
        for split in SPLITS[1:]:
            tokenizer = DiagnosticTokenizer()
            actions = itertools.cycle(scripts[split])

            def generate(ids, sample_seed, budget):
                return tokenizer.encode(next(actions)) + [tokenizer.eos_token_id]

            traces.extend(
                evaluation.evaluate_tasks(
                    [tasks[split][0]],
                    tokenizer,
                    config,
                    generate,
                    seed,
                    episodes_per_task=8,
                )
            )
        assert len(traces) == 16 and all(t["success"] for t in traces)
        results.append(evaluation.per_seed_results(traces, 8))
    summary = evaluation.aggregate(results, 8)
    for split in SPLITS[1:]:
        assert (
            summary[split]["avg_at_8"]["mean"]
            == summary[split]["pass_at_8"]["mean"]
            == 1.0
        )
        assert (
            summary[split]["avg_at_8"]["std"] == summary[split]["pass_at_8"]["std"] == 0
        )
        assert summary[split]["avg_at_8"]["n"] == 3
