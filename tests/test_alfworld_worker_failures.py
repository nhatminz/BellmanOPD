import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, Qwen2TokenizerFast

from b200_experiment import alfworld_evaluation as evaluation
from b200_experiment.config import save_config
from b200_experiment.tokenizer_compat import (
    prepare_tokenizer_view,
    verify_tokenizer_view,
)
from test_alfworld_multiturn import make_alfconfig


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint-000400"
    path.mkdir()
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "open": 1,
                "cabinet": 2,
                "1": 3,
                "<|im_end|>": 4,
                "<|endoftext|>": 5,
                "<extra>": 6,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<extra>"],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}"
    )
    tokenizer.save_pretrained(path)
    (path / "config.json").write_text('{"model_type":"qwen3"}')
    (path / "model.safetensors").write_bytes(b"dummy weights are never read/copied")
    return path


def set_extra(path, extra, name="tokenizer_config.json"):
    target = path / name
    config = json.loads(target.read_text())
    config["extra_special_tokens"] = extra
    target.write_text(json.dumps(config))


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["<extra>"],
        [
            {
                "content": "<extra>",
                "special": True,
                "normalized": False,
                "lstrip": False,
                "rstrip": False,
                "single_word": False,
            }
        ],
    ],
)
def test_real_qwen_tokenizer_list_metadata_compat_preserves_ids_template_and_checkpoint(
    checkpoint, tmp_path, extra
):
    reference = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    set_extra(checkpoint, extra)
    before = snapshot(checkpoint)
    if int(__import__("transformers").__version__.split(".")[0]) < 5:
        with pytest.raises(AttributeError, match="keys"):
            AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    runtime = prepare_tokenizer_view(checkpoint, tmp_path / "runtime")
    restored = AutoTokenizer.from_pretrained(runtime, local_files_only=True)
    verify_tokenizer_view(restored, checkpoint)
    assert snapshot(checkpoint) == before
    assert restored.get_vocab() == reference.get_vocab()
    assert restored.get_added_vocab() == reference.get_added_vocab()
    assert restored.all_special_ids == reference.all_special_ids
    assert restored.chat_template == reference.chat_template
    messages = [{"role": "user", "content": "open cabinet 1"}]
    assert restored.apply_chat_template(messages) == reference.apply_chat_template(
        messages
    )
    assert restored.encode("open cabinet 1 <extra>") == reference.encode(
        "open cabinet 1 <extra>"
    )
    assert restored.decode([1, 6, 4], skip_special_tokens=True) == reference.decode(
        [1, 6, 4], skip_special_tokens=True
    )
    assert (runtime / "model.safetensors").is_symlink()
    assert (runtime / "tokenizer.json").is_symlink()
    metadata = json.loads((runtime / "tokenizer_config.json").read_text())
    assert "extra_special_tokens" not in metadata


def test_mapping_metadata_needs_no_view_and_unknown_tokens_cannot_be_invented(
    checkpoint, tmp_path
):
    set_extra(checkpoint, {"image_token": "<extra>"})
    assert prepare_tokenizer_view(checkpoint, tmp_path / "unused") == checkpoint
    assert not (tmp_path / "unused").exists()
    set_extra(checkpoint, ["<missing>"])
    before = snapshot(checkpoint)
    with pytest.raises(ValueError, match="Refusing to invent/remap"):
        prepare_tokenizer_view(checkpoint, tmp_path / "invalid")
    assert snapshot(checkpoint) == before and not (tmp_path / "invalid").exists()


def test_special_tokens_map_list_is_also_normalized(checkpoint, tmp_path):
    set_extra(checkpoint, [])
    set_extra(checkpoint, ["<extra>"], "special_tokens_map.json")
    runtime = prepare_tokenizer_view(checkpoint, tmp_path / "runtime")
    assert "extra_special_tokens" not in json.loads(
        (runtime / "special_tokens_map.json").read_text()
    )
    verify_tokenizer_view(AutoTokenizer.from_pretrained(runtime), checkpoint)


def test_list_only_tokens_become_additional_tokens_without_new_ids(
    checkpoint, tmp_path
):
    reference = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = checkpoint / name
        metadata = json.loads(path.read_text())
        metadata.pop("additional_special_tokens", None)
        metadata["extra_special_tokens"] = ["<extra>"]
        path.write_text(json.dumps(metadata))
    runtime = prepare_tokenizer_view(checkpoint, tmp_path / "runtime")
    restored = AutoTokenizer.from_pretrained(runtime, local_files_only=True)
    assert restored.additional_special_tokens == ["<extra>"]
    assert restored.get_vocab() == reference.get_vocab()
    assert restored.all_special_ids == reference.all_special_ids
    verify_tokenizer_view(restored, checkpoint)


def test_vocabulary_identity_check_rejects_new_tokens(checkpoint):
    changed = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    changed.add_tokens(["new unintended token"])
    with pytest.raises(ValueError, match="changed vocabulary/IDs"):
        verify_tokenizer_view(changed, checkpoint)


@pytest.mark.parametrize("method", ["cmt", "opd", "ta", "iw"])
def test_worker_hf_and_vllm_both_use_compatible_checkpoint_tokenizer(
    checkpoint, tmp_path, monkeypatch, method
):
    config = make_alfconfig(tmp_path)
    config["experiment"]["method"] = method
    config["evaluation"]["episodes_per_task"] = 8
    cfg = tmp_path / "worker.yaml"
    save_config(config, cfg)
    set_extra(checkpoint, ["<extra>"])
    before = snapshot(checkpoint)
    engines, sampling_calls = [], []

    class LLM:
        def __init__(self, **kwargs):
            # Reproduce vLLM's second, independent tokenizer load.
            tok = AutoTokenizer.from_pretrained(
                kwargs["tokenizer"], local_files_only=True
            )
            verify_tokenizer_view(tok, checkpoint)
            engines.append(kwargs)

        def generate(self, prompts, sampling, **kwargs):
            sampling_calls.append(sampling)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4])])]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw)
    )
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._resolve_gpu_memory_utilization",
        lambda cfg: 0.5,
    )
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._release_vllm_engine", lambda engine: None
    )

    def episode(task, tokenizer, config, generate, seed, sample_index):
        assert generate([1, 2, 3], 123, 7) == [1, 2, 3, 4]
        return {"success": True, "task_id": task["id"], "sample_index": sample_index}

    monkeypatch.setattr(evaluation, "_evaluate_episode", episode)
    path = tmp_path / "rank-00000.jsonl"
    evaluation.worker(
        SimpleNamespace(
            worker_config=str(cfg),
            checkpoint=str(checkpoint),
            seed=42,
            rank=0,
            world_size=1,
            worker_output=str(path),
        )
    )
    assert engines[0]["model"] == str(checkpoint)
    assert engines[0]["tokenizer"] == str(path.with_suffix(".tokenizer"))
    assert len(sampling_calls) == 16 and all(x["n"] == 1 for x in sampling_calls)
    assert all(
        x["temperature"] == config["evaluation"]["temperature"] for x in sampling_calls
    )
    assert path.with_suffix(".done").is_file()
    status = json.loads(path.with_suffix(".status.json").read_text())
    assert status["status"] == "complete" and status["completed_episodes"] == 16
    assert status["tokenizer_metadata_compatibility_applied"] is True
    assert status["method"] == method and snapshot(checkpoint) == before


def test_import_failure_is_recorded_before_any_episode(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    path = tmp_path / "rank-00000.jsonl"
    with pytest.raises(ModuleNotFoundError):
        evaluation.worker(
            SimpleNamespace(
                checkpoint="/checkpoint",
                seed=42,
                rank=0,
                worker_output=str(path),
                worker_config="unused",
            )
        )
    error = json.loads(path.with_suffix(".error.json").read_text())
    assert error["phase"] == "imports" and error["status"] == "failed"
    assert "ModuleNotFoundError" in error["traceback"]
    assert not path.with_suffix(".done").exists()


@pytest.mark.parametrize("phase", ["vllm_startup", "evaluate_episode", "vllm_cleanup"])
def test_original_failure_phase_and_no_false_completion(
    checkpoint, tmp_path, monkeypatch, phase
):
    cfg = tmp_path / "worker.yaml"
    save_config(make_alfconfig(tmp_path), cfg)

    def fail():
        raise ValueError("original engine failure")

    class LLM:
        def __init__(self, **kwargs):
            if phase == "vllm_startup":
                fail()

    def episode(*args, **kwargs):
        if phase == "evaluate_episode":
            fail()
        return {"success": False}

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=object)
    )
    monkeypatch.setattr(evaluation, "_evaluate_episode", episode)
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._resolve_gpu_memory_utilization",
        lambda cfg: 0.5,
    )
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._release_vllm_engine",
        lambda engine: fail() if phase == "vllm_cleanup" else None,
    )
    path = tmp_path / "rank-00000.jsonl"
    with pytest.raises(ValueError, match="original engine failure"):
        evaluation.worker(
            SimpleNamespace(
                checkpoint=str(checkpoint),
                seed=42,
                rank=0,
                world_size=1,
                worker_output=str(path),
                worker_config=str(cfg),
            )
        )
    error = json.loads(path.with_suffix(".error.json").read_text())
    assert error["phase"] == phase and error["exception_type"] == "ValueError"
    assert error["message"] == "original engine failure"
    assert not path.with_suffix(".done").exists()


@pytest.mark.parametrize("codes", [(1, 2), (1, None), (0, 0), (-9, 0)])
def test_fast_and_live_worker_failures_surface_logs_and_exit_codes(
    tmp_path, monkeypatch, codes
):
    config = make_alfconfig(tmp_path)
    cfg = tmp_path / "worker.yaml"
    save_config(config, cfg)
    directory = tmp_path / "seed-42"
    directory.mkdir()
    terminated = []

    def popen(command, **kwargs):
        rank = int(command[command.index("--rank") + 1])
        path = Path(command[command.index("--worker-output") + 1])
        kwargs["stdout"].write(
            "Traceback (most recent call last):\nAttributeError: 'list' object has no attribute 'keys'\n"
        )
        kwargs["stdout"].flush()
        evaluation._write_json(
            path.with_suffix(".status.json"), {"phase": "load_tokenizer"}
        )
        return SimpleNamespace(poll=lambda: codes[rank], returncode=codes[rank])

    monkeypatch.setattr(evaluation.subprocess, "Popen", popen)
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._terminate_process_group", terminated.append
    )
    with pytest.raises(RuntimeError) as failure:
        evaluation.run_workers(
            cfg,
            {"checkpoint": "/saved/ta/checkpoint-000400"},
            42,
            directory,
            ["0", "1"],
        )
    message = str(failure.value)
    assert "AttributeError: 'list' object has no attribute 'keys'" in message
    assert "phase=load_tokenizer" in message and "exit_code=" in message
    assert "Incomplete ALFWorld worker" not in message
    payload = json.loads((directory / "worker_failures.json").read_text())
    assert [row["exit_code"] for row in payload["workers"]] == list(codes)
    assert len(terminated) == 2
    if codes[0] == -9:
        assert "signal=SIGKILL" in message
    if codes == (1, 2):
        assert "Worker 0" in message and "Worker 1" in message


def test_log_tail_is_bounded_and_handles_non_utf8(tmp_path):
    path = tmp_path / "worker.log"
    path.write_bytes(b"startup\n" * 10000 + b"last exception\xff\n")
    tail = evaluation._log_tail(path)
    assert len(tail.splitlines()) == 80 and "last exception" in tail
    assert len(tail.encode()) < 32780


def test_timeout_reports_active_workers_and_cleans_both(tmp_path, monkeypatch):
    config = make_alfconfig(tmp_path)
    config["evaluation"]["sync_timeout_sec"] = 1
    cfg = tmp_path / "worker.yaml"
    save_config(config, cfg)
    directory = tmp_path / "seed-42"
    directory.mkdir()
    terminated = []
    monkeypatch.setattr(
        evaluation.subprocess,
        "Popen",
        lambda *args, **kw: SimpleNamespace(poll=lambda: None),
    )
    clock = iter([0, 2])
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: next(clock, 2))
    monkeypatch.setattr(
        "b200_experiment.vllm_evaluation._terminate_process_group", terminated.append
    )
    with pytest.raises(TimeoutError, match="timeout after 1 seconds"):
        evaluation.run_workers(
            cfg, {"checkpoint": "/saved/student"}, 42, directory, ["0", "1"]
        )
    report = json.loads((directory / "worker_failures.json").read_text())
    assert report["failed_ranks"] == [0, 1] and len(terminated) == 2
    assert all(row["exit_code"] is None for row in report["workers"])
