"""Measured training cost, kept alongside a run's checkpoints and evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import threading
from pathlib import Path
from typing import Any, Callable


COST_TAGS = {
    "cost/train_seconds_per_optimizer_step": "cost_train_seconds_per_optimizer_step",
    "cost/optimizer_update_seconds": "cost_optimizer_update_seconds",
    "cost/rollout_seconds_per_step": "cost_rollout_seconds_per_step",
    "cost/scoring_seconds_per_step": "cost_scoring_seconds_per_step",
    "cost/selector_seconds_per_step": "cost_selector_seconds_per_step",
    "cost/allocation_seconds": "cost_allocation_seconds",
    "cost/checkpoint_seconds": "cost_checkpoint_seconds",
    "cost/evaluation_snapshot_seconds": "cost_evaluation_snapshot_seconds",
    "cost/evaluation_seconds": "cost_evaluation_seconds",
    "cost/peak_torch_allocated_gib": "cost_peak_torch_allocated_gib",
    "cost/peak_torch_reserved_gib": "cost_peak_torch_reserved_gib",
    "cost/peak_device_used_gib_sampled": "cost_peak_device_used_gib_sampled",
    "cost/selector_fraction_percent": "cost_selector_fraction_percent",
    "cost/overhead_vs_opd_percent": "cost_overhead_vs_opd_percent",
    "cost/is_warmup": "cost_is_warmup",
}


def comparison_signature(config: dict[str, Any], world_size: int) -> str:
    """Compare identical workloads, while allowing the method/selector to vary."""
    training = config.get("training", {})
    payload = {
        "world_size": world_size,
        "models": config.get("models", {}),
        "data": config.get("data", {}),
        "rollout": config.get("rollout", {}),
        "distributed": config.get("distributed", {}),
        "training": {
            k: training.get(k)
            for k in (
                "micro_batch_size_per_gpu",
                "ppo_mini_batch_size",
                "gradient_checkpointing",
                "use_lora",
                "lora",
                "fused_optimizer",
                "length_bucketed_micro_batches",
            )
        },
        "scoring": {
            k: config.get("selector", {}).get(k)
            for k in (
                "top_k",
                "score_micro_batch_size",
                "trim_padding",
                "length_bucketed_scoring",
                "joint_cross_scoring",
                "score_chunk_steps",
            )
        },
    }
    if config.get("alfworld", {}).get("enabled", False):
        payload["alfworld"] = config["alfworld"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DeviceMemoryMonitor:
    """Sample the whole device, including the separate co-located vLLM process.

    This is a sampled device-wide high-water mark, not a PyTorch allocator
    measurement and not a guaranteed capture of every short allocation spike.
    """

    def __init__(self, read_used_bytes: Callable[[], int], interval: float = 0.25):
        self.read = read_used_bytes
        self.interval = max(float(interval), 0.05)
        self.peak = 0
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            self.peak = max(self.peak, int(self.read()))
            self.samples += 1
        except Exception as error:
            self.error = str(error)

    def start(self) -> None:
        self._sample()

        def sample_loop():
            while not self._stop.wait(self.interval):
                self._sample()

        self._thread = threading.Thread(target=sample_loop, daemon=True)
        self._thread.start()

    def finish(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()


def cost_fields(
    event: dict,
    *,
    rollout_train_seconds: float,
    optimizer_steps: int,
    allocation_seconds: float,
    warmup: bool,
    device_peak_bytes: int,
) -> dict[str, float]:
    n = max(int(optimizer_steps), 1)
    selector = float(event.get("selector_time", 0.0))
    # CMT allocation is performed inside each optimizer step, outside selector_time.
    fields = {
        "cost_train_seconds_per_optimizer_step": rollout_train_seconds / n,
        "cost_optimizer_update_seconds": float(event["optimizer_step_wall_time_sec"]),
        "cost_rollout_seconds_per_step": float(event.get("rollout_time", 0.0)) / n,
        "cost_scoring_seconds_per_step": float(event.get("total_scoring_time_sec", 0.0))
        / n,
        "cost_selector_seconds_per_step": selector / n,
        "cost_allocation_seconds": allocation_seconds,
        "cost_checkpoint_seconds": float(event.get("checkpoint_time_sec", 0.0)),
        "cost_evaluation_snapshot_seconds": float(
            event.get("evaluation_snapshot_time_sec", 0.0)
        ),
        "cost_evaluation_seconds": float(
            event.get("periodic_evaluation", {}).get("evaluation_time", 0.0)
        ),
        "cost_peak_torch_allocated_gib": float(event.get("peak_gpu_allocated_gb", 0.0)),
        "cost_peak_torch_reserved_gib": float(event.get("peak_gpu_reserved_gb", 0.0)),
        "cost_peak_device_used_gib_sampled": device_peak_bytes / 2**30,
        "cost_selector_fraction_percent": 100
        * (selector / n + allocation_seconds)
        / max(rollout_train_seconds / n, 1e-12),
        "cost_is_warmup": float(warmup),
    }
    return fields


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class TrainingCostLogger:
    def __init__(
        self,
        output: Path,
        config: dict,
        world_size: int,
        resume_step: int | None = None,
    ):
        self.output = Path(output)
        self.config = config
        self.signature = comparison_signature(config, world_size)
        self.world_size = world_size
        settings = config.get("logging", {}).get("cost", {})
        self.enabled = bool(settings.get("enabled", True))
        self.warmup_rollouts = max(int(settings.get("warmup_rollouts", 1)), 0)
        self.resume_step = resume_step
        self.excluded_other_workload_rows = 0
        self._session_first_rollout: int | None = None
        self.runtime_settings = {
            "micro_batch_size_per_gpu": config.get("training", {}).get(
                "micro_batch_size_per_gpu"
            ),
            "score_micro_batch_size": config.get("selector", {}).get(
                "score_micro_batch_size"
            ),
            "ppo_mini_batch_size": config.get("training", {}).get(
                "ppo_mini_batch_size"
            ),
            "rollout_batch_size": config.get("rollout", {}).get("batch_size"),
            "num_responses": config.get("rollout", {}).get("num_responses"),
        }
        self.baseline = None
        baseline_path = settings.get("opd_baseline")
        if self.enabled and baseline_path:
            path = Path(baseline_path).expanduser()
            if path.is_dir():
                path = path / "training_cost_summary.json"
            self.baseline = json.loads(path.read_text(encoding="utf-8"))
            if (
                self.baseline.get("method") != "opd"
                or self.baseline.get("comparison_signature") != self.signature
            ):
                raise ValueError(
                    f"Cost baseline is not an OPD run with the same model/data/batch/GPU workload: {path}"
                )
            if not self.baseline.get("measured_optimizer_steps"):
                raise ValueError(
                    f"OPD cost baseline contains no steps after warmup: {path}"
                )
            self.baseline["source"] = str(path.resolve())
        self.rows: dict[int, dict] = {}
        path = self.output / "training_cost.jsonl"
        if self.enabled and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if resume_step is not None and int(row["step"]) > resume_step:
                        continue
                    if row["comparison_signature"] != self.signature:
                        if resume_step is None:
                            raise ValueError(
                                "Existing training_cost.jsonl belongs to a different workload"
                            )
                        # Micro-batch/runtime changes are supported by model
                        # resume. Do not reject them solely because older cost
                        # measurements used another workload, or average the
                        # two configurations together.
                        self.excluded_other_workload_rows += 1
                        continue
                    self.rows[int(row["step"])] = row

    def is_warmup(self, rollout_index: int) -> bool:
        # A resumed process also warms up kernels/allocators. Its first
        # rollout can be a partially consumed PPO group, so exclude it too.
        if self._session_first_rollout is None:
            self._session_first_rollout = int(rollout_index)
        return int(rollout_index) - self._session_first_rollout < self.warmup_rollouts

    def write(self, event: dict) -> None:
        if not self.enabled:
            return
        if self.baseline:
            baseline_seconds = self.baseline["train_seconds_per_optimizer_step_mean"]
            event["cost_overhead_vs_opd_percent"] = 100 * (
                event["cost_train_seconds_per_optimizer_step"] / baseline_seconds - 1
            )
        row = {
            "step": int(event["step"]),
            "method": event["method"],
            "rollout_batch_index": int(event["rollout_batch_index"]),
            "comparison_signature": self.signature,
            "world_size": self.world_size,
            "resume_step": self.resume_step,
            "excluded_other_workload_rows": self.excluded_other_workload_rows,
            "runtime_settings": self.runtime_settings,
            "num_valid_tokens": event.get("num_valid_tokens"),
            **{k: event[k] for k in COST_TAGS.values() if k in event},
        }
        # Replays after checkpoint resume replace the same logical optimizer
        # step in the summary; the append-only log retains the measurement audit.
        self.rows[row["step"]] = row
        with (self.output / "training_cost.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        csv_path = self.output / "training_cost.csv"
        fields = (
            "step",
            "method",
            "rollout_batch_index",
            "world_size",
            *COST_TAGS.values(),
        )
        exists = csv_path.is_file() and csv_path.stat().st_size > 0
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.save_summary()

    def save_summary(self) -> None:
        measured = [r for r in self.rows.values() if not r["cost_is_warmup"]]
        times = [r["cost_train_seconds_per_optimizer_step"] for r in measured]
        summary = {
            "schema_version": 1,
            "method": self.config["experiment"]["method"],
            "comparison_signature": self.signature,
            "world_size": self.world_size,
            "warmup_rollouts": self.warmup_rollouts,
            "warmup_scope": "first N rollouts in each fresh/resumed process session",
            "resume_step": self.resume_step,
            "runtime_settings": self.runtime_settings,
            "excluded_other_workload_rows": self.excluded_other_workload_rows,
            "measured_optimizer_steps": len(measured),
            "logged_step_max": max(self.rows, default=None),
            "measured_rollouts": len({r["rollout_batch_index"] for r in measured}),
            "train_seconds_per_optimizer_step_mean": statistics.mean(times)
            if times
            else None,
            "train_seconds_per_optimizer_step_std": statistics.stdev(times)
            if len(times) > 1
            else None,
            "train_seconds_total": sum(times),
            "time_scope": "rollout + scoring + selection + optimization + training diagnostics, amortized over optimizer updates; excludes checkpoint writes, evaluation snapshot export and periodic evaluation",
            "memory_scope": "torch peaks are max per rank including checkpoint export; device peak is sampled whole-device usage including vLLM, not a guaranteed instantaneous peak",
            "overhead_definition": "100 * (method_mean_step_seconds / matched_opd_mean_step_seconds - 1)",
            "peak_torch_allocated_gib": max(
                (r["cost_peak_torch_allocated_gib"] for r in measured), default=None
            ),
            "peak_torch_reserved_gib": max(
                (r["cost_peak_torch_reserved_gib"] for r in measured), default=None
            ),
            "peak_device_used_gib_sampled": max(
                (r["cost_peak_device_used_gib_sampled"] for r in measured), default=None
            ),
            "opd_baseline": self.baseline.get("source") if self.baseline else None,
            "overhead_vs_opd_percent": (
                100
                * (
                    statistics.mean(times)
                    / self.baseline["train_seconds_per_optimizer_step_mean"]
                    - 1
                )
                if times and self.baseline
                else None
            ),
        }
        tensorboard_dir = Path(
            self.config.get("logging", {})
            .get("tensorboard", {})
            .get("log_dir", "tensorboard")
        )
        summary["tensorboard_log_dir"] = str(
            (
                tensorboard_dir
                if tensorboard_dir.is_absolute()
                else self.output / tensorboard_dir
            ).resolve()
        )
        summary_path = self.output / "training_cost_summary.json"
        if summary_path.is_file():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_signature = previous.get("comparison_signature")
            if previous_signature and previous_signature != self.signature:
                # Preserve the previous measurement segment before updating the
                # canonical summary following an OOM/runtime configuration change.
                archive = (
                    self.output / f"training_cost_summary_{previous_signature}.json"
                )
                if not archive.exists():
                    _atomic_json(archive, previous)
        _atomic_json(summary_path, summary)


def write_comparison_tensorboard(output: Path, summary: dict, baseline: dict) -> dict:
    """Backfill measured per-step overhead when the OPD run finishes later."""
    rows = {}
    path = output / "training_cost.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"TensorBoard backfill requires per-step cost logs: {path}"
        )
    maximum_step = summary.get("logged_step_max")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("comparison_signature") != summary["comparison_signature"]:
            continue
        if maximum_step is not None and int(row["step"]) > int(maximum_step):
            continue
        rows[int(row["step"])] = row
    measured = [row for row in rows.values() if not row["cost_is_warmup"]]
    if not measured:
        raise ValueError(
            f"No matching measured per-step cost logs to backfill: {output}"
        )
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(summary.get("tensorboard_log_dir") or output / "tensorboard")
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for row in measured:
            overhead = 100 * (
                row["cost_train_seconds_per_optimizer_step"]
                / baseline["train_seconds_per_optimizer_step_mean"]
                - 1
            )
            writer.add_scalar(
                "cost/overhead_vs_opd_percent", overhead, global_step=int(row["step"])
            )
    finally:
        writer.close()
    return {"log_dir": str(log_dir), "written_steps": len(measured)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare measured run costs against a matching OPD baseline"
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--outputs", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also backfill per-step overhead into each run's TensorBoard after OPD is measured",
    )
    args = parser.parse_args(argv)

    def read(path):
        path = path / "training_cost_summary.json" if path.is_dir() else path
        return json.loads(path.read_text(encoding="utf-8"))

    baseline = read(args.baseline)
    if baseline.get("method") != "opd" or not baseline.get(
        "train_seconds_per_optimizer_step_mean"
    ):
        raise ValueError("Baseline must be an OPD cost summary with measured steps")
    for output in args.outputs:
        summary = read(output)
        if summary["comparison_signature"] != baseline["comparison_signature"]:
            raise ValueError(
                f"Cannot compare different model/data/batch/GPU workloads: {output}"
            )
        mean = summary["train_seconds_per_optimizer_step_mean"]
        if mean is None:
            raise ValueError(f"No measured steps after warmup: {output}")
        report = {
            "baseline": str(args.baseline.resolve()),
            "method": summary["method"],
            "overhead_vs_opd_percent": 100
            * (mean / baseline["train_seconds_per_optimizer_step_mean"] - 1),
            "method_cost": summary,
            "opd_cost": baseline,
        }
        if args.tensorboard:
            report["tensorboard_backfill"] = write_comparison_tensorboard(
                output, summary, baseline
            )
        _atomic_json(output / "training_cost_comparison.json", report)
        print(
            json.dumps(
                {k: v for k, v in report.items() if not k.endswith("_cost")}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
