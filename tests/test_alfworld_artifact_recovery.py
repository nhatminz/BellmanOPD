import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from b200_experiment import alfworld_evaluation as evaluation
from b200_experiment.alfworld_eval_artifacts import (
    read_episode_jsonl,
    read_completed_seed,
)
from b200_experiment.alfworld_env import SPLITS, load_task_splits
from b200_experiment.config import save_config
import test_alfworld_evaluation as legacy_tests
from test_alfworld_evaluation import args_for
from test_alfworld_sampled_evaluation import runner_for

alfconfig = legacy_tests.alfconfig
saved_run = legacy_tests.saved_run


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_line_separators_are_preserved_inside_long_json_strings(
    tmp_path, separator
):
    record = {
        "response": "a" * 46500 + separator + "漢字\nnext",
        "token_ids": [10, 12, 13],
    }
    wire = json.dumps(record, ensure_ascii=False) + "\n"
    with pytest.raises(json.JSONDecodeError, match="Unterminated string"):
        [json.loads(part) for part in wire.splitlines()]
    path = tmp_path / "rank-00000.jsonl"
    path.write_bytes(
        wire.encode("utf-8")
        + b"\r\n"
        + json.dumps(record, ensure_ascii=False).encode("utf-8")
        + b"\r\n"
    )
    assert list(read_episode_jsonl(path)) == [record, record]


def test_real_corruption_reports_physical_file_and_line_and_is_not_skipped(tmp_path):
    path = tmp_path / "rank-00000.jsonl"
    path.write_bytes(b'{"success":false}\n{"response":"cut off')
    with pytest.raises(ValueError, match=r"rank-00000.jsonl, physical line 2"):
        list(read_episode_jsonl(path))


def real_game_runner(config):
    base = runner_for(config)
    lookup = {
        (s, t["id"]): t for s, tasks in load_task_splits(config).items() for t in tasks
    }

    def run(*args):
        records = base(*args)
        for record in records:
            record["game_fingerprint"] = lookup[(record["split"], record["task_id"])][
                "game_fingerprint"
            ]
            record["turns"] = [
                {"response": "look", "action": "look", "success": record["success"]}
            ]
            if record["evaluation_seed"] == 43 and record["sample_index"] == 0:
                record["turns"][0].update(
                    response="invalid\u2028response", action=None, parser_failure=True
                )
        return records

    return run


@pytest.fixture
def prior_evaluation(saved_run, alfconfig):
    subdir = "checkpoint-000050"
    args = args_for(
        saved_run,
        checkpoints=["50"],
        episodes_per_task=8,
        seeds=[42, 43],
        gpus="0,1",
        checkpoint_subdir=subdir,
    )
    result = evaluation.run_evaluation(args, worker_runner=real_game_runner(alfconfig))
    root = Path(result["output"])
    directory = root / "step-000050" / "seed-43"
    rows = list(read_episode_jsonl(directory / "episodes.jsonl"))
    # Simulate the historical crash after both workers finished but before
    # Python's splitlines reader could publish seed-43 episodes/summary.
    for rank, split in enumerate(SPLITS[1:]):
        path = directory / f"rank-{rank:05d}.jsonl"
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in rows
                if r["split"] == split
            )
        )
        path.with_suffix(".done").write_text("complete\n")
    (directory / "episodes.jsonl").unlink()
    (directory / "summary.json").unlink()
    (root / "complete.json").unlink()
    return root


def recovered_args(saved_run, tag="recovered", **kwargs):
    return args_for(
        saved_run,
        checkpoints=["50"],
        episodes_per_task=8,
        checkpoint_subdir="checkpoint-000050",
        tag=tag,
        reuse_tag="test-fresh",
        **kwargs,
    )


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_reuses_finished_seeds_42_and_43_only_generates_44_and_preserves_old_files(
    prior_evaluation, saved_run, alfconfig
):
    before = snapshot(prior_evaluation)
    calls = []
    generate = real_game_runner(alfconfig)

    def run(config, selected, seed, directory, devices):
        calls.append(seed)
        return generate(config, selected, seed, directory, devices)

    result = evaluation.run_evaluation(recovered_args(saved_run), worker_runner=run)
    assert calls == [44]
    assert result["fresh_episodes"] is False and result["reuse_tag"] == "test-fresh"
    assert snapshot(prior_evaluation) == before
    aggregate = result["results"][0]["aggregate"][SPLITS[1]]
    assert aggregate["avg_at_8"]["mean"] == 0.375 and aggregate["avg_at_8"]["n"] == 3
    assert aggregate["pass_at_8"]["mean"] == pytest.approx(2 / 3)
    root = Path(result["output"]) / "step-000050"
    for seed in [42, 43, 44]:
        summary = json.loads((root / f"seed-{seed}" / "summary.json").read_text())
        assert bool(summary["reused_from"]) == (seed != 44)
    diagnostics = json.loads((root / "seed-43" / "diagnostics.json").read_text())
    assert diagnostics["zero_success"] is True and diagnostics["successes"] == 0
    assert diagnostics["response_unicode_line_separator_occurrences"]["U+2028"] == 2
    assert diagnostics["top_parser_failures"][0]["response"] == "invalid\u2028response"


def test_audit_seed_is_read_only_and_works_without_vllm(prior_evaluation, monkeypatch):
    before = snapshot(prior_evaluation)
    monkeypatch.setitem(sys.modules, "vllm", None)
    report = evaluation.audit_seed_directory(prior_evaluation / "step-000050/seed-43")
    assert report["seed"] == 43 and report["diagnostics"]["episodes"] == 16
    assert report["diagnostics"]["zero_success"] is True
    assert (
        report["diagnostics"]["response_unicode_line_separator_occurrences"]["U+2028"]
        == 2
    )
    assert snapshot(prior_evaluation) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "iw"),
        ("episodes_per_task", 1),
        ("protocol_fingerprint", "different"),
        ("checkpoints", []),
    ],
)
def test_reuse_rejects_different_checkpoint_protocol_or_sampling(
    prior_evaluation, saved_run, field, value
):
    path = prior_evaluation / "plan.json"
    plan = json.loads(path.read_text())
    plan[field] = value
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Cannot reuse"):
        evaluation.run_evaluation(
            recovered_args(saved_run),
            worker_runner=lambda *a: pytest.fail("No generation"),
        )
    assert not (saved_run / "alfworld_eval/recovered").exists()


def test_reuse_rejects_changed_game_and_damaged_completed_data(
    prior_evaluation, saved_run
):
    shard = prior_evaluation / "step-000050/seed-43/rank-00000.jsonl"
    rows = list(read_episode_jsonl(shard))
    rows[0]["game_fingerprint"] = "changed"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="game fingerprint"):
        evaluation.run_evaluation(
            recovered_args(saved_run),
            worker_runner=lambda *a: pytest.fail("No generation"),
        )
    shard.write_text('{"turns":[{"response":"truncated')
    with pytest.raises(ValueError, match="physical line 1"):
        evaluation.run_evaluation(
            recovered_args(saved_run, tag="corrupt"),
            worker_runner=lambda *a: pytest.fail("No generation"),
        )
    assert not (
        saved_run / "alfworld_eval/corrupt/checkpoint-000050/complete.json"
    ).exists()


def test_incomplete_seed_is_regenerated_in_new_namespace(
    prior_evaluation, saved_run, alfconfig
):
    (prior_evaluation / "step-000050/seed-43/rank-00001.done").unlink()
    assert read_completed_seed(prior_evaluation / "step-000050/seed-43", 2) is None
    calls = []
    base = real_game_runner(alfconfig)

    def run(cfg, selected, seed, directory, devices):
        calls.append(seed)
        return base(cfg, selected, seed, directory, devices)

    evaluation.run_evaluation(recovered_args(saved_run), worker_runner=run)
    assert calls == [43, 44]


def test_parent_worker_result_reader_preserves_unicode(
    alfconfig, tmp_path, monkeypatch
):
    cfg = tmp_path / "worker.yaml"
    config = dict(alfconfig)
    config["evaluation"]["episodes_per_task"] = 8
    save_config(config, cfg)
    all_rows = real_game_runner(alfconfig)(cfg, {}, 43, tmp_path, ["0", "1"])

    def popen(command, **kwargs):
        rank = int(command[command.index("--rank") + 1])
        path = Path(command[command.index("--worker-output") + 1])
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in all_rows
                if r["split"] == SPLITS[rank + 1]
            )
        )
        path.with_suffix(".done").write_text("complete\n")
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(evaluation.subprocess, "Popen", popen)
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._terminate_process_group", lambda *a: None
    )
    traces = evaluation.run_workers(
        cfg, {"checkpoint": "/saved/checkpoint"}, 43, tmp_path, ["0", "1"]
    )
    assert len(traces) == 16
    assert sum("\u2028" in t["turns"][0]["response"] for t in traces) == 2


def test_shell_accepts_reuse_tag_and_read_only_audit(prior_evaluation, saved_run):
    script = str(
        Path(__file__).resolve().parents[1] / "scripts/eval_alfworld_multiseed.sh"
    )
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "CUDA_VISIBLE_DEVICES": "",
        "REUSE_TAG": "test-fresh",
        "EVAL_TAG": "dry-recovered",
    }
    result = subprocess.run(
        [
            "bash",
            script,
            "--checkpoint-paths",
            str(saved_run / "checkpoint-000050"),
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"reuse_tag": "test-fresh"' in result.stdout
    assert not (saved_run / "alfworld_eval/dry-recovered").exists()
    audit = subprocess.run(
        ["bash", script, "--audit-seed", str(prior_evaluation / "step-000050/seed-43")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert audit.returncode == 0, audit.stderr
    assert json.loads(audit.stdout)["diagnostics"]["successes"] == 0
