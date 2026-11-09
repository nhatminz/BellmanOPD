import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from b200_experiment.alfworld_evaluation import (
    aggregate,
    build_plan,
    episode_seed,
    evaluate_tasks,
    parser,
    per_seed_results,
    run_evaluation,
)
from b200_experiment.alfworld_env import load_task_splits
from b200_experiment.config import save_config
from test_alfworld_multiturn import make_alfconfig, Tokenizer, InteractiveEnv


@pytest.fixture
def alfconfig(tmp_path):
    return make_alfconfig(tmp_path)


def args_for(run, checkpoints=("50", "final"), **kwargs):
    args = parser().parse_args(
        [
            "--run-output",
            str(run),
            "--checkpoints",
            *checkpoints,
            "--tag",
            "test-fresh",
            "--gpus",
            "0",
        ]
    )
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def saved_run(alfconfig, tmp_path):
    run = tmp_path / "run" / "cmt_opd"
    run.mkdir(parents=True)
    save_config(alfconfig, run / "resolved_config.yaml")
    for name in ("checkpoint-000050", "final"):
        directory = run / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
    (run / "latest.json").write_text(json.dumps({"checkpoint": "final", "step": 100}))
    return run


def trace(task, seed, success=True):
    return {
        "task_id": task["id"],
        "task_type": task["task_type"],
        "split": task["split"],
        "success": success,
        "turns": [{}, {}],
        "invalid_actions": int(not success),
        "evaluation_seed": seed,
    }


def test_checkpoint_selection_without_baseline_and_validation(saved_run):
    args = args_for(saved_run)
    run, config, chosen = build_plan(args)
    assert run == saved_run
    assert [p["step"] for p in chosen] == [50, 100]
    assert config["alfworld"]["enabled"]
    args.checkpoints = [str(saved_run / "checkpoint-000050")]
    assert build_plan(args)[2][0]["step"] == 50
    args.method = "ta"
    with pytest.raises(ValueError, match="Method"):
        build_plan(args)
    args.method = None
    args.seeds = [42, 42]
    with pytest.raises(ValueError, match="distinct"):
        build_plan(args)


def test_success_turns_invalid_rates_mean_and_sample_std():
    task = {"id": "x", "task_type": "pick", "split": "eval_in_distribution"}
    values = [
        per_seed_results([trace(task, seed, success=seed != 44)])
        for seed in [42, 43, 44]
    ]
    summary = aggregate(values)["eval_in_distribution"]
    assert summary["success_rate"]["mean"] == pytest.approx(2 / 3)
    assert summary["success_rate"]["std"] == pytest.approx(3**-0.5)
    assert summary["average_turns"]["mean"] == 2
    assert summary["invalid_action_rate"]["mean"] == pytest.approx(1 / 6)
    assert "pick" in summary["task_types"]


def test_all_seeds_execute_fresh_environment_interactions(alfconfig):
    task = load_task_splits(alfconfig)["eval_in_distribution"][0]
    instances_before = len(InteractiveEnv.instances)
    token = Tokenizer()
    results = []
    for seed in [42, 43, 44]:
        actions = iter(["open cabinet 1", "take cup 1 from cabinet 1"])

        def generate(ids, sample_seed, budget):
            return token.encode(next(actions)) + [1]

        results += evaluate_tasks(
            [task], token, alfconfig, generate, seed, env_factory=InteractiveEnv
        )
    assert len(InteractiveEnv.instances) - instances_before == 3
    assert [t["evaluation_seed"] for t in results] == [42, 43, 44]
    assert len({t["seed"] for t in results}) == 3
    assert all(t["success"] for t in results)
    assert episode_seed(42, task) == episode_seed(42, copy.deepcopy(task))


def test_evaluation_artifacts_independent_of_training_and_never_overwritten(
    saved_run, alfconfig
):
    old = saved_run / "eval_history.jsonl"
    old.write_text("old math artifact must stay byte-identical\n")
    tasks = load_task_splits(alfconfig)
    calls = []

    def worker(config, selected, seed, directory, devices):
        calls.append((selected["step"], seed))
        return [
            {
                **trace(task, seed, seed != 44),
                "protocol_fingerprint": alfconfig["alfworld"]["protocol_fingerprint"],
            }
            for split in ("eval_in_distribution", "eval_out_of_distribution")
            for task in tasks[split]
        ]

    result = run_evaluation(args_for(saved_run), worker_runner=worker)
    output = Path(result["output"])
    assert calls == [(50, 42), (50, 43), (50, 44), (100, 42), (100, 43), (100, 44)]
    assert (output / "summary.csv").is_file()
    assert (output / "task_results.jsonl").is_file()
    assert (output / "complete.json").is_file()
    assert len(list(output.glob("step-*/seed-*/episodes.jsonl"))) == 6
    assert old.read_text() == "old math artifact must stay byte-identical\n"
    with pytest.raises(FileExistsError):
        run_evaluation(args_for(saved_run), worker_runner=worker)
    assert len(calls) == 6


def test_missing_or_duplicate_tasks_cannot_produce_complete_summary(saved_run):
    with pytest.raises(RuntimeError, match="Missing/duplicate"):
        run_evaluation(args_for(saved_run), worker_runner=lambda *args: [])
    assert not (saved_run / "alfworld_eval/test-fresh/complete.json").exists()


def test_dry_run_validates_full_splits_without_running_a_model(saved_run):
    plan = run_evaluation(
        args_for(saved_run, dry_run=True),
        worker_runner=lambda *args: pytest.fail("should not start a model"),
    )
    assert plan["seeds"] == [42, 43, 44]
    assert not (saved_run / "alfworld_eval").exists()


def test_legacy_checkpoint_requires_explicit_changed_protocol_eval(
    saved_run, alfconfig
):
    legacy = copy.deepcopy(alfconfig)
    legacy["alfworld"].update(protocol_version=1, action_prompt_mode="legacy")
    from b200_experiment.alfworld_protocol import resolve_protocol

    legacy = resolve_protocol(legacy)
    save_config(legacy, saved_run / "resolved_config.yaml")
    before = (saved_run / "resolved_config.yaml").read_bytes()
    args = args_for(
        saved_run,
        set=["alfworld.protocol_version=2", "alfworld.action_prompt_mode=admissible"],
        dry_run=True,
    )
    with pytest.raises(ValueError, match="--allow-protocol-change"):
        build_plan(args)
    args.allow_protocol_change = True
    plan = run_evaluation(args)
    assert plan["evaluation_protocol_changed"] and plan["protocol_version"] == 2
    assert plan["protocol_fingerprint"] != plan["training_protocol_fingerprint"]
    assert (saved_run / "resolved_config.yaml").read_bytes() == before


def test_changed_eval_sampling_is_recorded_as_a_protocol_change(saved_run):
    args = args_for(saved_run, set=["evaluation.temperature=0.7"])
    with pytest.raises(ValueError, match="protocol differs"):
        build_plan(args)
    args.allow_protocol_change = True
    _, config, _ = build_plan(args)
    assert config["alfworld"]["protocol_manifest"]["generation"]["temperature"] == 0.7


def test_vllm_multi_turn_syncs_once_for_variable_length_episodes(
    monkeypatch, alfconfig
):
    from b200_experiment.vllm_rollout import VLLMRolloutEngine
    import b200_experiment.models as models

    engine = object.__new__(VLLMRolloutEngine)
    engine.process = object()
    engine.config = alfconfig
    calls = []
    monkeypatch.setattr(models, "load_student_tokenizer", lambda config: Tokenizer())
    monkeypatch.setattr(engine, "_release_torch_cache_if_needed", lambda: (False, 0))
    monkeypatch.setattr(engine, "_wake_weights", lambda: calls.append("wake_weights"))
    monkeypatch.setattr(engine, "_wake_kv_cache", lambda: calls.append("wake_kv"))
    monkeypatch.setattr(
        engine, "_sync_weights", lambda model: (calls.append("sync") or {})
    )
    monkeypatch.setattr(engine, "_sleep", lambda: calls.append("sleep"))
    monkeypatch.setattr(
        engine, "_generate_one", lambda *args, **kwargs: (0, [10, 1], None)
    )
    with engine.multiturn_session(object()) as generate:
        for i in range(7):
            assert generate([2, 3], i, 10) == [10, 1]
    assert calls == ["wake_weights", "sync", "wake_kv", "sleep"]


@pytest.mark.parametrize("large_pair", [False, True])
def test_four_launchers_share_workload_and_resolve_b200_paths(tmp_path, large_pair):
    from b200_experiment.cli import _configured, build_parser
    from b200_experiment.training_cost import comparison_signature

    repo = Path(__file__).resolve().parents[1]
    capture = tmp_path / "commands.jsonl"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        f"real={sys.executable!r}\n"
        "if sys.argv[1] == '-': os.execv(real,[real,*sys.argv[1:]])\n"
        "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
    )
    executable.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": str(executable),
        "CAPTURE": str(capture),
        "RUN_NAME": "fair_alfworld",
        "CUDA_VISIBLE_DEVICES": "0,1" if large_pair else "0,1,2,3",
        "STORAGE_ROOT": "/workspace/storage-shared",
    }
    launcher = (
        "train_alfworld_multiturn_14b.sh"
        if large_pair
        else "train_alfworld_multiturn.sh"
    )
    for method in ["cmt", "opd", "ta", "iw"]:
        result = subprocess.run(
            ["bash", str(repo / "scripts" / launcher), method],
            env=env,
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    configs = [
        _configured(build_parser().parse_args(c[c.index("train") :])) for c in commands
    ]
    world_size = 2 if large_pair else 4
    assert len({comparison_signature(c, world_size) for c in configs}) == 1
    for method, config in zip(["cmt", "opd", "ta", "iw"], configs):
        assert config["experiment"]["method"] == method
        assert config["experiment"]["output_dir"].endswith(
            f"/fair_alfworld/{method}_opd"
        )
        assert (
            config["data"]["path"]
            == "/workspace/storage-shared/nlp/minhpn19/data/ALFworld/train-00000-of-00001.parquet"
        )
        assert config["models"]["teacher_path"] == (
            "/workspace/storage-shared/models/Qwen3-14B"
            if large_pair
            else "/workspace/storage-shared/models/Qwen3-4B"
        )
        assert config["rollout"]["batch_size"] == (16 if large_pair else 8)
        assert config["training"]["micro_batch_size_per_gpu"] == (
            4 if large_pair else 1
        )
        assert config["training"]["save_interval"] == (100 if large_pair else 50)
        if large_pair:
            assert (
                config["models"]["student_path"]
                == "/workspace/storage-shared/models/Qwen3-4B"
            )
        assert config["rollout"]["num_responses"] == 4
        assert config["alfworld"]["max_turns"] == 30
        assert config["training"]["max_steps"] == 1000
        assert not config["training_evaluation"]["enabled"]
        assert config["logging"]["cost"]["enabled"]


def test_offline_worker_connects_generation_to_fresh_episodes(
    monkeypatch, alfconfig, tmp_path
):
    import b200_experiment.alfworld_rollout as rollout
    import b200_experiment.models as models
    import b200_experiment.vllm_evaluation as vllm_eval
    from b200_experiment.alfworld_evaluation import worker

    tok = Tokenizer()
    called = []

    class LLM:
        def __init__(self, **kwargs):
            called.append(kwargs)

        def generate(self, prompts, sampling, use_tqdm):
            prefix = tok.decode(prompts[0]["prompt_token_ids"])
            action = (
                "take cup 1 from cabinet 1"
                if "Cabinet is open" in prefix
                else "open cabinet 1"
            )
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=tok.encode(action) + [1])]
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(models, "load_student_tokenizer", lambda cfg: tok)
    monkeypatch.setattr(vllm_eval, "_resolve_gpu_memory_utilization", lambda cfg: 0.5)
    monkeypatch.setattr(vllm_eval, "_release_vllm_engine", lambda engine: None)
    original = rollout.collect_episode
    monkeypatch.setattr(
        rollout,
        "collect_episode",
        lambda *a, **kw: original(*a, **kw, env_factory=InteractiveEnv),
    )
    cfg = tmp_path / "worker.yaml"
    save_config(alfconfig, cfg)
    path = tmp_path / "shard.jsonl"
    worker(
        SimpleNamespace(
            worker_config=str(cfg),
            checkpoint="/saved/student",
            seed=43,
            rank=0,
            world_size=1,
            worker_output=str(path),
        )
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and all(row["success"] for row in rows)
    assert {row["split"] for row in rows} == {
        "eval_in_distribution",
        "eval_out_of_distribution",
    }
    assert called[0]["model"] == "/saved/student"
    assert called[0]["tensor_parallel_size"] == 1
    assert called[0]["generation_config"] == "vllm"
    assert path.with_suffix(".done").exists()
