"""Standalone sampled MMLU-Pro evaluation: GPU replicas, fresh seeds, mean ± SD.

Run with ``python -m b200_experiment.mmlu_pro_evaluation --help``. Workers own
one vLLM engine each and reuse it for all seeds of a checkpoint. No teacher,
training dataset, historical eval files or optimizer state is needed.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .evaluation_context import resolve_evaluation_max_model_len
from .mmlu_pro import (
    PROTOCOL,
    aggregate_seeds,
    extract_answer,
    load_questions,
    metrics_by_category,
    question_messages,
    score_predictions,
)

REPO_DIR = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_asset_path(value: str, storage_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Also accept the two missing-leading-slash forms used in IDE prompts.
        if str(path).startswith("workspace/storage-shared/"):
            path = Path("/") / path
        elif path.parts and path.parts[0] == storage_root.name:
            path = storage_root.parent / path
        else:
            path = storage_root / path
    return path.resolve()


def _model_fingerprint(path: Path) -> str:
    items = [
        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(path.iterdir())
        if p.is_file()
        and (
            p.suffix in {".json", ".safetensors", ".bin", ".model"}
            or p.name.startswith("tokenizer")
        )
    ]
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if (
        len(args.seeds) < 2
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
    ):
        raise ValueError("--seeds requires at least two distinct non-negative seeds")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError(
            "Three-seed sampled evaluation requires finite temperature > 0"
        )
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.max_new_tokens <= 0 or args.max_model_len <= args.max_new_tokens:
        raise ValueError("Require 0 < --max-new-tokens < --max-model-len")
    if min(args.max_num_seqs, args.batch_size, args.max_num_batched_tokens) <= 0:
        raise ValueError("Batch and vLLM scheduling limits must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be positive and finite")
    devices = [device.strip() for device in args.gpus.split(",") if device.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--gpus must contain distinct GPU IDs, e.g. 0,1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        raise ValueError("--tag must be a simple directory name")
    storage_root = Path(args.storage_root).expanduser().resolve()
    data = resolve_asset_path(args.data, storage_root)
    questions = load_questions(data, args.limit)
    checkpoints = [
        str(resolve_asset_path(value, storage_root)) for value in args.checkpoints
    ]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("Checkpoint list contains duplicates")
    return {
        "protocol": PROTOCOL,
        "dataset": str(data),
        "dataset_sha256": _sha256_file(data),
        "questions": len(questions),
        "categories": sorted({row["category"] for row in questions}),
        "checkpoints": checkpoints,
        "seeds": args.seeds,
        "gpus": devices,
        "tag": args.tag,
        "output_root": str(Path(args.output_root).expanduser().resolve() / args.tag),
        "settings": {
            "num_responses": 8,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "enable_thinking": args.enable_thinking == "true",
            "limit": args.limit,
            "enforce_eager": args.enforce_eager,
            "timeout_sec": args.timeout_sec,
        },
    }


def checkpoint_request(plan: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    path = Path(checkpoint)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Saved checkpoint is missing config.json: {path}")
    if not any(path.glob("model*.safetensors")) and not any(
        path.glob("pytorch_model*.bin")
    ):
        raise FileNotFoundError(
            f"No full Hugging Face model weights found in {path}; adapter-only checkpoints are not supported"
        )
    settings = dict(plan["settings"])
    settings["max_model_len"] = resolve_evaluation_max_model_len(
        path, settings["max_model_len"]
    )
    if settings["max_model_len"] <= settings["max_new_tokens"]:
        raise ValueError("Native model context must exceed --max-new-tokens")
    return {
        **plan,
        "checkpoint": checkpoint,
        "checkpoint_fingerprint": _model_fingerprint(path),
        "settings": settings,
        "output": str(path.parent / "mmlu_pro_eval" / plan["tag"] / path.name),
    }


def _seed_for_question(seed: int, index: int) -> int:
    # Independent deterministic streams; sharding/batch sizes do not change
    # a question's sampling seed. vLLM still has usual floating-point variance.
    digest = hashlib.sha256(f"mmlu-pro:{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _validate_rows(rows: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    keys = ("index", "question_id", "question", "options", "category", "answer")
    if len(rows) != len(expected) or any(
        any(row.get(key) != original[key] for key in keys)
        for row, original in zip(rows, expected)
    ):
        raise ValueError(
            "Prediction rows dropped, duplicated, reordered or changed MMLU-Pro questions"
        )
    if rows:
        score_predictions(rows)


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt_token_ids(
    tokenizer, row: dict[str, Any], *, enable_thinking: bool
) -> list[int]:
    # Transformers versions differ in apply_chat_template's return_dict default.
    # Iterating a BatchEncoding yields strings such as "input_ids", which
    # vLLM must never receive as token IDs. Request the form explicitly, then
    # extract and validate before allocating the engine or measuring length.
    encoded = tokenizer.apply_chat_template(
        question_messages(row),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True,
        return_tensors=None,
        padding=False,
        truncation=False,
    )
    ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if (
        isinstance(ids, (list, tuple))
        and len(ids) == 1
        and isinstance(ids[0], (list, tuple))
    ):
        ids = ids[0]  # tolerate a singleton batch, never flatten multiple prompts
    if (
        not isinstance(ids, (list, tuple))
        or not ids
        or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in ids
        )
    ):
        raise TypeError(
            f"Question {row['question_id']}: tokenizer must return a non-empty "
            "integer token-ID sequence, or a mapping with an input_ids field "
            f"containing that sequence; got {type(encoded).__name__} "
            f"with input_ids type {type(ids).__name__}."
        )
    return [int(value) for value in ids]


def run_worker(request: dict[str, Any], rank: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from .models import generation_stop_token_ids
    from .vllm_evaluation import _release_vllm_engine

    checkpoint = Path(request["checkpoint"])
    settings = request["settings"]
    questions = load_questions(request["dataset"], settings["limit"])
    questions = [
        row for row in questions if row["index"] % len(request["gpus"]) == rank
    ]
    output = Path(request["output"]) / "shards" / f"rank_{rank}"
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in request["seeds"]:
        predictions = output / f"seed_{seed}.jsonl.gz"
        if predictions.is_file():
            _validate_rows(_read_predictions(predictions), questions)
        else:
            pending.append(seed)
    if not pending:
        return
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    prompts = []
    device = request["gpus"][rank]
    for row in tqdm(
        questions,
        desc=f"MMLU-Pro GPU {device}: prompts",
        position=rank + 1,
        dynamic_ncols=True,
    ):
        ids = _prompt_token_ids(
            tokenizer, row, enable_thinking=settings["enable_thinking"]
        )
        if len(ids) + settings["max_new_tokens"] > settings["max_model_len"]:
            raise ValueError(
                f"Question {row['question_id']}: {len(ids)} prompt tokens + "
                f"{settings['max_new_tokens']} response tokens exceed context "
                f"{settings['max_model_len']}. Increase --max-model-len or reduce "
                "--max-new-tokens. Questions are never silently truncated or dropped."
            )
        prompts.append({"prompt_token_ids": ids})
    if prompts:
        rendered = tokenizer.decode(
            prompts[0]["prompt_token_ids"], skip_special_tokens=False
        )
        (output / "first_prompt.txt").write_text(rendered + "\n", encoding="utf-8")
    # Use each checkpoint's own EOS/turn terminators, never a teacher's IDs.
    stop_tokens = generation_stop_token_ids(tokenizer, {})
    vocab = tokenizer.get_vocab()
    for token in ("<|im_end|>", "<|eot_id|>"):
        if token in vocab and vocab[token] not in stop_tokens:
            stop_tokens.append(vocab[token])
    engine_kwargs = {
        "model": str(checkpoint),
        "tokenizer": str(checkpoint),
        "dtype": settings["dtype"],
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": settings["gpu_memory_utilization"],
        "max_model_len": settings["max_model_len"],
        "max_num_seqs": settings["max_num_seqs"],
        "max_num_batched_tokens": settings["max_num_batched_tokens"],
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "enforce_eager": settings["enforce_eager"],
        "disable_log_stats": True,
        "seed": request["seeds"][0],
        "trust_remote_code": False,
        "generation_config": "vllm",
    }
    model_config = json.loads((checkpoint / "config.json").read_text())
    if model_config.get("model_type") == "qwen3_5":
        engine_kwargs["language_model_only"] = True
    engine = None
    try:
        if prompts:
            engine = LLM(**engine_kwargs)
        for seed in pending:
            destination = output / f"seed_{seed}.jsonl.gz"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                with (
                    gzip.open(temporary, "wt", encoding="utf-8") as handle,
                    tqdm(
                        total=len(questions),
                        desc=f"MMLU-Pro seed {seed} GPU {device}",
                        unit="question",
                        position=rank + 1,
                        dynamic_ncols=True,
                    ) as progress,
                ):
                    for start in range(0, len(questions), settings["batch_size"]):
                        batch = questions[start : start + settings["batch_size"]]
                        params = [
                            SamplingParams(
                                n=8,
                                temperature=settings["temperature"],
                                top_p=settings["top_p"],
                                max_tokens=settings["max_new_tokens"],
                                stop_token_ids=stop_tokens,
                                seed=_seed_for_question(seed, row["index"]),
                            )
                            for row in batch
                        ]
                        generated = engine.generate(
                            prompts[start : start + len(batch)], params, use_tqdm=False
                        )
                        if len(generated) != len(batch):
                            raise RuntimeError("vLLM returned an incomplete batch")
                        for row, generated_row in zip(batch, generated):
                            if len(generated_row.outputs) != 8:
                                raise RuntimeError(
                                    "vLLM must return exactly 8 completions per question"
                                )
                            responses = []
                            for item in generated_row.outputs:
                                parsed = extract_answer(item.text, len(row["options"]))
                                responses.append(
                                    {
                                        "text": item.text,
                                        "parsed_answer": parsed,
                                        "correct": parsed == row["answer"],
                                        "finish_reason": item.finish_reason,
                                        "generated_tokens": len(item.token_ids),
                                    }
                                )
                            handle.write(
                                json.dumps(
                                    {
                                        **row,
                                        "seed": seed,
                                        "sampling_seed": _seed_for_question(
                                            seed, row["index"]
                                        ),
                                        "responses": responses,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        progress.update(len(batch))
                temporary.replace(destination)  # only complete seeds are resumable
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        _release_vllm_engine(engine)


def launch_workers(request: dict[str, Any]) -> None:
    from .distributed import isolate_distributed_subprocess_environment
    from .vllm_evaluation import (
        _exclusive_evaluation_gpu_locks,
        _terminate_process_group,
    )

    request_path = Path(request["output"]) / "request.json"
    processes = []
    timeout = request["settings"]["timeout_sec"]
    with _exclusive_evaluation_gpu_locks(request["gpus"], timeout):
        try:
            for rank, device in enumerate(request["gpus"]):
                environment = isolate_distributed_subprocess_environment()
                environment.update(
                    CUDA_VISIBLE_DEVICES=device,
                    PYTHONUNBUFFERED="1",
                    TOKENIZERS_PARALLELISM="false",
                    VLLM_WORKER_MULTIPROC_METHOD="spawn",
                )
                environment["PYTHONPATH"] = os.pathsep.join(
                    filter(None, (str(REPO_DIR), environment.get("PYTHONPATH")))
                )
                command = [
                    sys.executable,
                    "-m",
                    "b200_experiment.mmlu_pro_evaluation",
                    "--worker-plan",
                    str(request_path),
                    "--rank",
                    str(rank),
                ]
                processes.append(
                    subprocess.Popen(
                        command, cwd=REPO_DIR, env=environment, start_new_session=True
                    )
                )
            deadline = time.monotonic() + timeout
            while True:
                codes = [process.poll() for process in processes]
                if any(code is not None and code != 0 for code in codes):
                    raise RuntimeError(
                        f"MMLU-Pro GPU worker failed: exit codes={codes}; see traceback above"
                    )
                if all(code == 0 for code in codes):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"MMLU-Pro checkpoint exceeded {timeout:g} seconds"
                    )
                time.sleep(0.25)
        finally:
            # Terminate descendants too, including engines whose launcher failed.
            for process in processes:
                _terminate_process_group(process)


def merge_checkpoint(request: dict[str, Any]) -> dict[str, Any]:
    output = Path(request["output"])
    questions = load_questions(request["dataset"], request["settings"]["limit"])
    results = []
    for seed in request["seeds"]:
        rows = []
        for rank in range(len(request["gpus"])):
            shard = _read_predictions(
                output / "shards" / f"rank_{rank}" / f"seed_{seed}.jsonl.gz"
            )
            expected = [
                row for row in questions if row["index"] % len(request["gpus"]) == rank
            ]
            _validate_rows(shard, expected)
            if any(
                row["seed"] != seed
                or row["sampling_seed"] != _seed_for_question(seed, row["index"])
                for row in shard
            ):
                raise ValueError("Prediction shard has incorrect seed metadata")
            rows.extend(shard)
        rows.sort(key=lambda row: row["index"])
        _validate_rows(rows, questions)
        result = {
            "seed": seed,
            "overall": score_predictions(rows),
            "by_category": metrics_by_category(rows),
            "prediction_shards": [
                str(output / "shards" / f"rank_{rank}" / f"seed_{seed}.jsonl.gz")
                for rank in range(len(request["gpus"]))
            ],
        }
        _write_json(output / f"seed_{seed}_summary.json", result)
        results.append(result)
    summary = {
        "benchmark": "MMLU-Pro",
        "protocol": PROTOCOL,
        "shots": 0,
        "checkpoint": request["checkpoint"],
        "dataset": request["dataset"],
        "settings": request["settings"],
        "gpus": request["gpus"],
        "request": str(output / "request.json"),
        **aggregate_seeds(results),
    }
    _write_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", [summary])
    return summary


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "checkpoint",
        "category",
        "questions",
        "seeds",
        "avg_at_8_mean",
        "avg_at_8_std",
        "avg_at_8_percent",
        "pass_at_8_mean",
        "pass_at_8_std",
        "pass_at_8_percent",
        "invalid_answer_rate",
        "truncated_response_rate",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for summary in summaries:
                for category, scores in [
                    ("ALL", summary["overall"]),
                    *summary["by_category"].items(),
                ]:
                    writer.writerow(
                        {
                            "checkpoint": summary["checkpoint"],
                            "category": category,
                            "questions": scores["questions"],
                            "seeds": " ".join(map(str, summary["seeds"])),
                            **{
                                f"{metric}_{stat}": scores[metric][stat]
                                for metric in ("avg_at_8", "pass_at_8")
                                for stat in ("mean", "std")
                            },
                            "avg_at_8_percent": scores["avg_at_8"]["formatted_percent"],
                            "pass_at_8_percent": scores["pass_at_8"][
                                "formatted_percent"
                            ],
                            "invalid_answer_rate": scores["invalid_answer_rate"][
                                "mean"
                            ],
                            "truncated_response_rate": scores[
                                "truncated_response_rate"
                            ]["mean"],
                        }
                    )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _commit_request(path: Path, request: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError(
                f"Existing request differs at {path}. Use a new --tag; previous results are preserved."
            )
    else:
        _write_json(path, request)


def execute(args: argparse.Namespace, runner=None) -> dict[str, Any]:
    plan = build_plan(args)
    if args.dry_run:
        return {
            **plan,
            "checkpoint_requests": [
                checkpoint_request(plan, path) for path in plan["checkpoints"]
            ],
        }
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with (root / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _commit_request(root / "plan.json", plan)
        for checkpoint in tqdm(
            plan["checkpoints"],
            desc="MMLU-Pro checkpoints",
            unit="checkpoint",
            dynamic_ncols=True,
        ):
            try:
                request = checkpoint_request(plan, checkpoint)
                output = Path(request["output"])
                output.mkdir(parents=True, exist_ok=True)
                with (output / ".lock").open("a+") as checkpoint_lock:
                    fcntl.flock(checkpoint_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _commit_request(output / "request.json", request)
                    tqdm.write(
                        f"MMLU-Pro: {checkpoint}; seeds={plan['seeds']}; GPUs={','.join(plan['gpus'])}; output={output}"
                    )
                    (runner or launch_workers)(request)
                    result = merge_checkpoint(request)
                results.append(result)
                scores = result["overall"]
                tqdm.write(
                    f"Completed: avg@8={scores['avg_at_8']['formatted_percent']}%; pass@8={scores['pass_at_8']['formatted_percent']}%"
                )
            except Exception as error:
                failures.append(
                    {
                        "checkpoint": checkpoint,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                traceback.print_exc()
                tqdm.write(f"FAILED, continuing to next checkpoint: {checkpoint}")
            combined = {
                "benchmark": "MMLU-Pro",
                "protocol": PROTOCOL,
                "plan": plan,
                "results": results,
                "failures": failures,
            }
            _write_json(root / "summary.json", combined)
            write_csv(root / "summary.csv", results)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--storage-root",
        default=os.environ.get("STORAGE_ROOT", "/workspace/storage-shared"),
    )
    parser.add_argument(
        "--data", default="nlp/minhpn19/data/mmlu-pro/test-00000-of-00001.parquet"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Full saved Hugging Face checkpoint directories",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument(
        "--tag",
        default=f"mmlu_pro_{datetime.now():%Y%m%d_%H%M%S}_{time.time_ns() % 10**9:09d}",
    )
    parser.add_argument(
        "--output-root", default=str(REPO_DIR / "outputs" / "mmlu_pro_eval")
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Questions per generate call on each GPU",
    )
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--enable-thinking", choices=["true", "false"], default="false")
    parser.add_argument(
        "--limit",
        type=int,
        help="First N questions for a smoke run; omitted for full test",
    )
    parser.add_argument("--timeout-sec", type=float, default=86400)
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Disable vLLM CUDA graphs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset/checkpoints and print plan without loading model weights",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_plan:
        run_worker(json.loads(args.worker_plan.read_text()), args.rank)
        return 0
    if not args.checkpoints:
        parser.error("--checkpoints requires at least one checkpoint directory")
    summary = execute(args)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(
            f"Combined summary: {Path(summary['plan']['output_root']) / 'summary.json'}",
            flush=True,
        )
    return int(bool(summary.get("failures")))


if __name__ == "__main__":
    raise SystemExit(main())
