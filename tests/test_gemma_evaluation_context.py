import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from b200_experiment.evaluation_context import resolve_evaluation_max_model_len
from b200_experiment.multiseed_evaluation import build_plan, execute
from b200_experiment.vllm_evaluation import (
    evaluate_vllm_suite,
    merge_vllm_evaluation_shards,
)
import test_multiseed_evaluation as multiseed_tests
from test_vllm_evaluation import _LLM, _SamplingParams, _Tokenizer


@pytest.mark.parametrize(
    "requested, expected", [(9216, 8192), (8192, 8192), (4096, 4096), (None, 8192)]
)
def test_gemma2_native_limit_from_checkpoint(tmp_path, requested, expected):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma2", "max_position_embeddings": 8192})
    )
    original = (tmp_path / "config.json").read_bytes()
    if requested == 9216:
        with pytest.warns(RuntimeWarning, match="using 8192"):
            actual = resolve_evaluation_max_model_len(tmp_path, requested)
    else:
        actual = resolve_evaluation_max_model_len(tmp_path, requested)
    assert actual == expected
    assert (tmp_path / "config.json").read_bytes() == original


def test_other_model_context_extensions_are_preserved(tmp_path):
    # Do not mistake an original RoPE limit for an extended model's context.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "max_position_embeddings": 8192,
                "rope_scaling": {"factor": 4},
            }
        )
    )
    assert resolve_evaluation_max_model_len(tmp_path, 16384) == 16384


def test_legacy_run_is_resolved_before_request_fingerprint_and_dry_run(tmp_path):
    fixture = multiseed_tests.MultiseedEvaluationTests()
    run = fixture.make_run(tmp_path)
    checkpoint = run / "checkpoint-000300"
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "gemma2", "max_position_embeddings": 8192})
    )
    source = run / "resolved_config.yaml"
    config = yaml.safe_load(source.read_text())
    config["training_evaluation"].update(
        max_new_tokens=7168, vllm={"seed": 1234, "max_model_len": 9216}
    )
    source.write_text(yaml.safe_dump(config))
    before = source.read_bytes()
    args = fixture.args(run, checkpoints=["300"], seeds=[42, 43, 44], fresh_seeds=True)
    with pytest.warns(RuntimeWarning, match="using 8192"):
        _, _, plan = build_plan(args)
    settings = plan[0]["settings"]
    assert settings["vllm"]["max_model_len"] == 8192
    assert settings["max_new_tokens"] == 7168
    assert settings["temperature"] == 0.7 and settings["top_p"] == 0.95
    assert settings["num_responses"] == 8
    calls = []
    with pytest.warns(RuntimeWarning, match="using 8192"):
        summary = execute(args, fixture.evaluator(calls))
    assert len(calls) == 3
    assert all(call[2]["vllm"]["max_model_len"] == 8192 for call in calls)
    assert summary["results"][0]["seeds"] == [42, 43, 44]
    request = json.loads(
        (
            run / "eval_multiseed/extra_seeds/checkpoint-000300/seed_42/request.json"
        ).read_text()
    )
    assert request["settings"]["vllm"]["max_model_len"] == 8192
    assert source.read_bytes() == before
    # Explicit 8192 and automatic resolution produce the same resume request.
    args.max_model_len = 8192
    execute(args, lambda *a, **kw: pytest.fail("Must reuse the committed result"))


def test_gemma_engine_on_two_shards_honors_context_and_preserves_avg_pass(
    monkeypatch, tmp_path
):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma2", "max_position_embeddings": 8192})
    )
    dataset = tmp_path / "math.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"problem": f"{i}?", "answer": "1"}) + "\n" for i in range(3)
        )
    )
    config = {
        "models": {"dtype": "bfloat16", "stop_token_ids": [1, 107]},
        "data": {"chat_template_kwargs": {}},
        "evaluation": {"benchmarks": {"MATH-500": {"path": str(dataset)}}},
    }
    settings = {
        "max_new_tokens": 7168,
        "temperature": 0.7,
        "top_p": 0.95,
        "num_responses": 8,
        "metric": "avg@8",
        "benchmark_names": ["MATH-500"],
        "vllm": {"max_model_len": 9216, "seed": 42, "gpu_memory_utilization": 0.4},
    }
    original = copy.deepcopy(settings)

    class CheckedLLM(_LLM):
        instances = []

        def __init__(self, **kwargs):
            assert kwargs["max_model_len"] <= 8192, (
                "same validation that rejected the user's old Gemma config"
            )
            super().__init__(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=CheckedLLM, SamplingParams=_SamplingParams),
    )
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
        lambda *a, **kw: _Tokenizer(),
    )
    for rank in range(2):
        with pytest.warns(RuntimeWarning, match="using 8192"):
            evaluate_vllm_suite(
                "gemma",
                tmp_path,
                config,
                tmp_path / f"rank-{rank}",
                {**settings, "_shard_rank": rank, "_shard_world_size": 2},
            )
    merged = merge_vllm_evaluation_shards(
        "gemma",
        tmp_path,
        config,
        tmp_path / "merged",
        [tmp_path / "rank-0", tmp_path / "rank-1"],
        settings,
    )
    result = merged["benchmarks"]["MATH-500"]
    assert result["problems"] == 3 and result["samples_per_problem"] == 8
    assert result["avg_at_8"] == result["pass_at_8"] == 1.0
    assert merged["parameters"]["max_model_len"] == 8192
    assert merged["parameters"]["max_new_tokens"] == 7168
    for engine in CheckedLLM.instances:
        assert engine.kwargs["seed"] == 42
        sampling = engine.generate_calls[0][1].kwargs
        assert sampling["n"] == 8 and sampling["max_tokens"] == 7168
        assert sampling["stop_token_ids"] == [1, 107]
    with gzip.open(merged["detailed_outputs"], "rt") as handle:
        assert len([json.loads(line) for line in handle]) == 3
    assert settings == original


@pytest.mark.parametrize("selected", [None, "300 1050 1500"])
def test_gemma_batch_launcher_passes_native_context_and_selected_checkpoints(
    tmp_path, selected
):
    repo = Path(__file__).resolve().parents[1]
    capture = tmp_path / "requests.jsonl"
    mock = tmp_path / "python"
    mock.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps({'args':sys.argv[1:],'gpu':os.environ['CUDA_VISIBLE_DEVICES']})+'\\n')\n"
    )
    mock.chmod(0o755)
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": str(mock),
        "CAPTURE": str(capture),
    }
    if selected:
        environment["EVAL_CHECKPOINTS"] = selected
    result = subprocess.run(
        ["bash", str(repo / "eval_gemma_selected_multiseed.sh"), "--dry-run"],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    requests = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(requests) == 4
    steps = (selected or "200 1050 1500").split()
    for request in requests:
        args = request["args"]
        assert request["gpu"] == "0,1"
        assert args[args.index("--world-size") + 1] == "2"
        assert args[args.index("--max-model-len") + 1] == "8192"
        assert (
            args[args.index("--checkpoints") + 1 : args.index("--checkpoints") + 4]
            == steps
        )
        assert args[args.index("--seeds") + 1 : args.index("--seeds") + 4] == [
            "42",
            "43",
            "44",
        ]
        assert (
            args[args.index("--tag") + 1]
            == f"gemma_s42_43_44_steps{'_'.join(steps)}_ctx8192"
        )
        assert "--fresh-seeds" in args and "--dry-run" in args
