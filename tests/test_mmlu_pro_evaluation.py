import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from b200_experiment import mmlu_pro_evaluation as evaluation
from b200_experiment.mmlu_pro import (
    aggregate_seeds,
    extract_answer,
    load_questions,
    question_messages,
    score_predictions,
)


def make_args(tmp_path, count=5):
    data = tmp_path / "test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "question_id": i + 70,
                    "question": f"question {i}",
                    "options": ["first", "second", "N/A"],
                    "answer": "B",
                    "answer_index": 1,
                    "cot_content": "SECRET REFERENCE COT",
                    "category": "math" if i % 2 == 0 else "business",
                }
                for i in range(count)
            ]
        ),
        data,
    )
    checkpoint = tmp_path / "run" / "cmt_opd" / "checkpoint-001500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"qwen3"}')
    (checkpoint / "model.safetensors").write_bytes(b"fixture; model loading is mocked")
    return evaluation.build_parser().parse_args(
        [
            "--storage-root",
            str(tmp_path),
            "--data",
            str(data),
            "--checkpoints",
            str(checkpoint),
            "--output-root",
            str(tmp_path / "combined"),
            "--tag",
            "unit",
            "--batch-size",
            "2",
        ]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is (B).", "B"),
        ("The answer is (J).", None),  # only three choices in this fixture
        ("Answer: C", "C"),
        ("The final answer is: (b).", "B"),
        ("**The answer is (B).**", "B"),
        (r"Reasoning: \boxed{B}", "B"),
        ("B", "B"),
        ("(B).", "B"),
        ("A is possible. B could also work.", None),
        ("The answer is (A). Correction: the answer is (B).", "B"),
        ("<think>The answer is (A).</think>The answer is (B).", "B"),
        ("<think>The answer is (B).", None),
        ("<think>unresolved</think>", None),
        ("The answer is Apple.", None),
        ("", None),
    ],
)
def test_answer_extraction(text, expected):
    assert extract_answer(text, 3) == expected


def test_real_schema_has_no_reference_cot_in_prompt(tmp_path):
    args = make_args(tmp_path)
    rows = load_questions(args.data)
    assert len(rows) == 5 and rows[0]["question_id"] == "70"
    assert rows[0]["options"] == ["first", "second", "N/A"]
    assert "cot_content" not in rows[0]
    text = question_messages(rows[0])[0]["content"]
    assert "C. N/A" in text and "SECRET REFERENCE COT" not in text
    changed = {**rows[0], "answer": "A"}
    assert question_messages(changed) == question_messages(rows[0])
    assert load_questions(args.data, 2) == rows[:2]


@pytest.mark.parametrize(
    "change",
    [
        {"answer": "A"},
        {"answer": ""},
        {"answer": None, "answer_index": None},
        {"answer_index": 10},
        {"options": []},
        {"question": ""},
    ],
)
def test_invalid_dataset_references_fail_explicitly(tmp_path, change):
    args = make_args(tmp_path)
    rows = pq.read_table(args.data).to_pylist()
    rows[0].update(change)
    pq.write_table(pa.Table.from_pylist(rows), args.data)
    with pytest.raises(ValueError):
        load_questions(args.data)


class Tokenizer:
    eos_token_id = 1

    def get_vocab(self):
        return {"<end_of_turn>": 107, "<|im_end|>": 151645}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False and kwargs["tokenize"] is True
        assert "SECRET REFERENCE COT" not in messages[0]["content"]
        return list(range(30))

    def decode(self, ids, **kwargs):
        return "rendered native prompt"


class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class LLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.instances.append(self)

    def generate(self, prompts, params, use_tqdm):
        self.calls.append((prompts, params))
        assert use_tqdm is False
        outputs = []
        for param in params:
            assert param.kwargs["n"] == 8
            assert param.kwargs["temperature"] == 0.7 and param.kwargs["top_p"] == 0.95
            assert param.kwargs["stop_token_ids"] == [1, 107, 151645]
            outputs.append(
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="The answer is (B)."
                            if i == 0
                            else "The answer is (A).",
                            token_ids=[20, 21],
                            finish_reason="stop",
                        )
                        for i in range(8)
                    ]
                )
            )
        return outputs


def install_runtime(monkeypatch):
    LLM.instances = []
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams)
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._release_vllm_engine", lambda engine: None
    )


def local_runner(request):
    for rank in range(len(request["gpus"])):
        evaluation.run_worker(request, rank)


def test_two_shards_three_seeds_dual_metrics_and_cache(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    install_runtime(monkeypatch)
    summary = evaluation.execute(args, local_runner)
    assert not summary["failures"] and len(summary["results"]) == 1
    result = summary["results"][0]
    assert result["seeds"] == [42, 43, 44]
    assert result["overall"]["questions"] == 5
    assert result["overall"]["avg_at_8"]["mean"] == 0.125
    assert result["overall"]["pass_at_8"]["mean"] == 1.0
    assert result["overall"]["avg_at_8"]["std"] == 0.0
    assert result["by_category"]["math"]["questions"] == 3
    assert result["by_category"]["business"]["questions"] == 2
    # One model per GPU, reused for all three seeds (not six model loads).
    assert len(LLM.instances) == 2
    assert [len(model.calls) for model in LLM.instances] == [6, 3]
    assert all(model.kwargs["tensor_parallel_size"] == 1 for model in LLM.instances)
    request = evaluation.checkpoint_request(
        evaluation.build_plan(args), args.checkpoints[0]
    )
    output = Path(request["output"])
    before = {path: path.read_bytes() for path in output.glob("shards/*/*.jsonl.gz")}
    evaluation.execute(args, local_runner)
    assert len(LLM.instances) == 2  # committed seeds were validated and reused
    assert all(path.read_bytes() == contents for path, contents in before.items())
    for path in [
        output / "summary.csv",
        Path(args.output_root) / args.tag / "summary.csv",
    ]:
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["category"] == "ALL"
        assert rows[0]["avg_at_8_percent"] == "12.50 ± 0.00"


def test_question_sampling_seed_independent_of_rank_and_batch_size():
    seeds = {
        evaluation._seed_for_question(seed, index)
        for seed in [42, 43, 44]
        for index in range(5)
    }
    assert len(seeds) == 15
    assert evaluation._seed_for_question(42, 2) == evaluation._seed_for_question(42, 2)


def test_sample_sd_is_across_seeds_not_questions():
    seed_results = []
    for seed, value in zip([42, 43, 44], [0.25, 0.5, 0.75]):
        scores = {
            "questions": 2,
            "avg_at_8": value,
            "pass_at_8": value,
            "invalid_answer_rate": 0,
            "truncated_response_rate": 0,
        }
        seed_results.append(
            {"seed": seed, "overall": scores, "by_category": {"math": scores}}
        )
    result = aggregate_seeds(seed_results)
    assert result["overall"]["avg_at_8"]["mean"] == 0.5
    assert result["overall"]["avg_at_8"]["std"] == 0.25
    assert result["overall"]["avg_at_8"]["formatted_percent"] == "50.00 ± 25.00"


def test_worker_failure_continues_to_next_checkpoint_and_uses_completed_seed_cache(
    tmp_path, monkeypatch
):
    args = make_args(tmp_path)
    second = Path(args.checkpoints[0]).parent / "checkpoint-001050"
    second.mkdir()
    (second / "config.json").write_text('{"model_type":"qwen3"}')
    (second / "model.safetensors").write_bytes(b"fixture")
    args.checkpoints.append(str(second))
    install_runtime(monkeypatch)

    def runner(request):
        if request["checkpoint"] == args.checkpoints[0]:
            raise RuntimeError("simulated GPU failure")
        local_runner(request)

    result = evaluation.execute(args, runner)
    assert len(result["failures"]) == 1 and len(result["results"]) == 1
    assert result["results"][0]["checkpoint"] == str(second)
    result = evaluation.execute(args, local_runner)
    assert not result["failures"] and len(result["results"]) == 2


def test_same_tag_different_settings_rejected_without_overwrite(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    install_runtime(monkeypatch)
    evaluation.execute(args, local_runner)
    plan_path = Path(args.output_root) / args.tag / "plan.json"
    original = plan_path.read_bytes()
    args.temperature = 0.8
    with pytest.raises(ValueError, match="new --tag"):
        evaluation.execute(args, local_runner)
    assert plan_path.read_bytes() == original


def test_merge_rejects_wrong_question_shard(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    install_runtime(monkeypatch)
    evaluation.execute(args, local_runner)
    request = evaluation.checkpoint_request(
        evaluation.build_plan(args), args.checkpoints[0]
    )
    path = Path(request["output"]) / "shards/rank_1/seed_42.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    rows[0]["question_id"] = "wrong-id"
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="changed MMLU-Pro"):
        evaluation.merge_checkpoint(request)


def test_invalid_answers_and_length_diagnostics_are_not_guessed(tmp_path, monkeypatch):
    args = make_args(tmp_path, 1)
    install_runtime(monkeypatch)
    result = evaluation.execute(args, local_runner)["results"][0]
    request = json.loads(Path(result["request"]).read_text())
    rows = evaluation._read_predictions(
        Path(request["output"]) / "shards/rank_0/seed_42.jsonl.gz"
    )
    for item in rows[0]["responses"]:
        item.update(
            text="I cannot decide between A and B",
            parsed_answer=None,
            correct=False,
            finish_reason="length",
        )
    metrics = score_predictions(rows)
    assert metrics["avg_at_8"] == metrics["pass_at_8"] == 0
    assert metrics["invalid_answer_rate"] == metrics["truncated_response_rate"] == 1


def test_context_overflow_fails_before_loading_engine(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    args.max_model_len, args.max_new_tokens = 40, 20
    install_runtime(monkeypatch)
    request = evaluation.checkpoint_request(
        evaluation.build_plan(args), args.checkpoints[0]
    )
    with pytest.raises(ValueError, match="never silently truncated"):
        evaluation.run_worker(request, 0)
    assert not LLM.instances


def test_completed_seed_survives_failure_and_resume_only_generates_pending_seeds(
    tmp_path, monkeypatch
):
    args = make_args(tmp_path, 2)
    install_runtime(monkeypatch)
    request = evaluation.checkpoint_request(
        evaluation.build_plan(args), args.checkpoints[0]
    )
    original_generate = LLM.generate

    def fail_second_seed(self, prompts, params, use_tqdm):
        if len(self.calls) == 1:
            raise RuntimeError("simulated interruption during seed 43")
        return original_generate(self, prompts, params, use_tqdm)

    monkeypatch.setattr(LLM, "generate", fail_second_seed)
    with pytest.raises(RuntimeError, match="seed 43"):
        evaluation.run_worker(request, 0)
    output = Path(request["output"]) / "shards/rank_0"
    first_seed = output / "seed_42.jsonl.gz"
    before = first_seed.read_bytes()
    assert not (output / "seed_43.jsonl.gz").exists()
    assert not list(output.glob(".*.tmp"))
    monkeypatch.setattr(LLM, "generate", original_generate)
    evaluation.run_worker(request, 0)
    assert first_seed.read_bytes() == before
    assert len(LLM.instances[-1].calls) == 2  # seeds 43 and 44 only


@pytest.mark.parametrize("failed", [False, True])
def test_worker_launch_masks_gpu_strips_torchrun_and_cleans_all_process_groups(
    tmp_path, monkeypatch, failed
):
    args = make_args(tmp_path)
    request = evaluation.checkpoint_request(
        evaluation.build_plan(args), args.checkpoints[0]
    )
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    spawned, terminated = [], []

    def popen(command, **kwargs):
        code = 1 if failed and len(spawned) == 0 else 0
        process = SimpleNamespace(poll=lambda: code)
        spawned.append((command, kwargs, process))
        return process

    monkeypatch.setattr(evaluation.subprocess, "Popen", popen)
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._terminate_process_group", terminated.append
    )
    if failed:
        with pytest.raises(RuntimeError, match="worker failed"):
            evaluation.launch_workers(request)
    else:
        evaluation.launch_workers(request)
    assert len(spawned) == len(terminated) == 2
    for rank, (command, kwargs, process) in enumerate(spawned):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(rank)
        assert (
            "RANK" not in kwargs["env"]
            and "TORCHELASTIC_USE_AGENT_STORE" not in kwargs["env"]
        )
        assert command[-2:] == ["--rank", str(rank)]
        assert kwargs["start_new_session"] is True


def test_gemma_context_is_resolved_and_plan_is_read_only(tmp_path):
    args = make_args(tmp_path)
    args.max_model_len = 9216
    config = Path(args.checkpoints[0]) / "config.json"
    config.write_text('{"model_type":"gemma2","max_position_embeddings":8192}')
    before = config.read_bytes()
    args.dry_run = True
    with pytest.warns(RuntimeWarning, match="using 8192"):
        result = evaluation.execute(args)
    assert result["checkpoint_requests"][0]["settings"]["max_model_len"] == 8192
    assert not (Path(args.output_root) / args.tag).exists()
    assert config.read_bytes() == before


def test_adapter_only_weights_are_rejected_before_gpu_launch(tmp_path):
    args = make_args(tmp_path)
    checkpoint = Path(args.checkpoints[0])
    (checkpoint / "model.safetensors").rename(checkpoint / "adapter_model.safetensors")
    with pytest.raises(FileNotFoundError, match="adapter-only"):
        evaluation.checkpoint_request(evaluation.build_plan(args), str(checkpoint))


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("seeds", [42, 42, 44]),
        ("temperature", 0),
        ("temperature", float("nan")),
        ("top_p", 2),
        ("gpus", "0,0"),
        ("tag", "../invalid"),
        ("limit", 0),
        ("gpu_memory_utilization", 0),
        ("batch_size", 0),
    ],
)
def test_invalid_plan_arguments(tmp_path, attribute, value):
    args = make_args(tmp_path)
    setattr(args, attribute, value)
    with pytest.raises(ValueError):
        evaluation.build_plan(args)


def test_storage_paths_accept_user_missing_leading_slash(tmp_path):
    root = Path("/workspace/storage-shared")
    for value in [
        "/workspace/storage-shared/nlp/file",
        "workspace/storage-shared/nlp/file",
        "storage-shared/nlp/file",
        "nlp/file",
    ]:
        assert evaluation.resolve_asset_path(value, root) == root / "nlp/file"


def test_shell_launcher_cli_overrides_support_a_real_parquet_dry_run(tmp_path):
    args = make_args(tmp_path)
    command = [
        "bash",
        str(evaluation.REPO_DIR / "eval_mmlu_pro_multiseed.sh"),
        "--checkpoints",
        args.checkpoints[0],
        "--data",
        args.data,
        "--output-root",
        args.output_root,
        "--tag",
        "dry",
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout.split("\n", 1)[1])
    assert plan["seeds"] == [42, 43, 44]
    assert plan["checkpoint_requests"][0]["checkpoint"] == args.checkpoints[0]
    assert not (Path(args.output_root) / "dry").exists()


@pytest.mark.skipif(
    not os.environ.get("MMLU_PRO_REAL_DATA"),
    reason="Set MMLU_PRO_REAL_DATA to an optional official parquet",
)
def test_official_test_parquet_schema_without_model():
    questions = load_questions(os.environ["MMLU_PRO_REAL_DATA"])
    assert len(questions) == 12032
    assert len({row["category"] for row in questions}) == 14
    assert {len(row["options"]) for row in questions} == set(range(3, 11))
