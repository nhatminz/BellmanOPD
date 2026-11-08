from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.tensorboard_logging import production_tensorboard_metrics
from b200_experiment.training_cost import (
    TrainingCostLogger,
    comparison_signature,
    cost_fields,
    main,
)


class TrainingCostTests(unittest.TestCase):
    def config(self, method="opd"):
        return {
            "experiment": {"method": method},
            "models": {"student_path": "student", "teacher_path": "teacher"},
            "data": {},
            "rollout": {"batch_size": 64},
            "training": {"micro_batch_size_per_gpu": 1},
            "logging": {"cost": {"warmup_rollouts": 1}},
        }

    def event(self, step, rollout, seconds):
        event = {
            "step": step,
            "method": "opd",
            "rollout_batch_index": rollout,
            "optimizer_step_wall_time_sec": 2,
            "rollout_time": 8,
            "total_scoring_time_sec": 4,
            "selector_time": 2,
            "checkpoint_time_sec": 11,
            "periodic_evaluation": {"evaluation_time": 100},
            "peak_gpu_allocated_gb": 20,
            "peak_gpu_reserved_gb": 22,
        }
        event.update(
            cost_fields(
                event,
                rollout_train_seconds=seconds,
                optimizer_steps=4,
                allocation_seconds=0.25,
                warmup=rollout == 0,
                device_peak_bytes=30 * 2**30,
            )
        )
        return event

    def test_amortized_step_cost_separates_io_and_exports_tensorboard(self):
        event = self.event(1, 0, 40)
        self.assertEqual(event["cost_train_seconds_per_optimizer_step"], 10)
        self.assertEqual(event["cost_checkpoint_seconds"], 11)
        self.assertEqual(event["cost_evaluation_seconds"], 100)
        self.assertEqual(event["cost_selector_fraction_percent"], 7.5)
        self.assertEqual(event["cost_peak_device_used_gib_sampled"], 30)
        tags = production_tensorboard_metrics(event, "opd")
        self.assertEqual(tags["cost/train_seconds_per_optimizer_step"], 10)
        self.assertEqual(tags["cost/peak_device_used_gib_sampled"], 30)
        self.assertNotIn("cost/overhead_vs_opd_percent", tags)

    def test_warmup_resume_and_measured_opd_overhead(self):
        with tempfile.TemporaryDirectory() as temporary:
            opd, cmt = Path(temporary) / "opd", Path(temporary) / "cmt"
            opd.mkdir()
            cmt.mkdir()
            logger = TrainingCostLogger(opd, self.config(), 4)
            logger.write(self.event(1, 0, 4000))
            logger.write(self.event(2, 1, 40))
            logger.write(self.event(3, 1, 40))
            baseline = json.loads((opd / "training_cost_summary.json").read_text())
            self.assertEqual(baseline["train_seconds_per_optimizer_step_mean"], 10)
            self.assertEqual(baseline["measured_optimizer_steps"], 2)
            config = self.config("cmt")
            config["logging"]["cost"]["opd_baseline"] = str(opd)
            other = TrainingCostLogger(cmt, config, 4)
            event = self.event(2, 1, 50)
            event["method"] = "cmt"
            other.write(event)
            self.assertEqual(event["cost_overhead_vs_opd_percent"], 25)
            self.assertEqual(
                production_tensorboard_metrics(event, "opd")[
                    "cost/overhead_vs_opd_percent"
                ],
                25,
            )
            resumed = TrainingCostLogger(opd, self.config(), 4, resume_step=2)
            resumed.save_summary()
            self.assertEqual(
                json.loads((opd / "training_cost_summary.json").read_text())[
                    "measured_optimizer_steps"
                ],
                1,
            )

    def test_different_workloads_cannot_claim_overhead(self):
        config = self.config()
        different = copy.deepcopy(config)
        different["models"]["student_path"] = "other_student"
        self.assertNotEqual(
            comparison_signature(config, 4), comparison_signature(different, 4)
        )
        self.assertNotEqual(
            comparison_signature(config, 4), comparison_signature(config, 2)
        )
        different = copy.deepcopy(config)
        different["experiment"]["method"] = "ta"
        self.assertEqual(
            comparison_signature(config, 4), comparison_signature(different, 4)
        )

    def test_oom_resume_with_smaller_micro_keeps_old_logs_but_separates_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old_config = self.config("cmt")
            old_config["training"]["micro_batch_size_per_gpu"] = 16
            old_logger = TrainingCostLogger(output, old_config, 4)
            for step in (149, 150, 188):
                event = self.event(step, step // 4, 40)
                event["method"] = "cmt"
                old_logger.write(event)
            old_summary = json.loads(
                (output / "training_cost_summary.json").read_text()
            )
            old_log = (output / "training_cost.jsonl").read_bytes()
            current = copy.deepcopy(old_config)
            current["training"]["micro_batch_size_per_gpu"] = 4
            with self.assertRaisesRegex(ValueError, "different workload"):
                TrainingCostLogger(output, current, 4)
            resumed = TrainingCostLogger(output, current, 4, resume_step=150)
            self.assertEqual(resumed.rows, {})
            self.assertEqual(resumed.excluded_other_workload_rows, 2)
            self.assertTrue(resumed.is_warmup(37))
            self.assertTrue(resumed.is_warmup(37))
            self.assertFalse(resumed.is_warmup(38))
            event = self.event(153, 38, 50)
            event["method"] = "cmt"
            resumed.write(event)
            summary = json.loads((output / "training_cost_summary.json").read_text())
            self.assertEqual(summary["measured_optimizer_steps"], 1)
            self.assertEqual(summary["train_seconds_per_optimizer_step_mean"], 12.5)
            self.assertEqual(summary["runtime_settings"]["micro_batch_size_per_gpu"], 4)
            self.assertEqual(summary["resume_step"], 150)
            self.assertTrue(
                (output / "training_cost.jsonl").read_bytes().startswith(old_log)
            )
            archive = output / f"training_cost_summary_{old_logger.signature}.json"
            self.assertEqual(json.loads(archive.read_text()), old_summary)

    def test_smaller_micro_resume_still_rejects_mismatched_opd_cost_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            opd = Path(temporary) / "opd"
            cmt = Path(temporary) / "cmt"
            opd.mkdir()
            cmt.mkdir()
            old = self.config()
            old["training"]["micro_batch_size_per_gpu"] = 16
            TrainingCostLogger(opd, old, 4).write(self.event(2, 1, 40))
            new = self.config("cmt")
            new["training"]["micro_batch_size_per_gpu"] = 4
            new["logging"]["cost"]["opd_baseline"] = str(opd)
            with self.assertRaisesRegex(
                ValueError, "same model/data/batch/GPU workload"
            ):
                TrainingCostLogger(cmt, new, 4, resume_step=150)

    def test_later_opd_baseline_backfills_overhead_into_original_tensorboard(self):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd, cmt = root / "opd", root / "cmt"
            opd.mkdir()
            cmt.mkdir()
            TrainingCostLogger(opd, self.config(), 4).write(self.event(2, 1, 40))
            config = self.config("cmt")
            config["logging"]["tensorboard"] = {"log_dir": "custom_tensorboard"}
            logger = TrainingCostLogger(cmt, config, 4)
            event = self.event(1, 0, 400)
            event["method"] = "cmt"
            logger.write(event)
            event = self.event(2, 1, 50)
            event["method"] = "cmt"
            logger.write(event)
            # Raw audit contains a stale future branch; the canonical summary
            # determines the committed range to backfill.
            future = self.event(188, 30, 999)
            future["comparison_signature"] = logger.signature
            with (cmt / "training_cost.jsonl").open("a") as handle:
                handle.write(json.dumps(future) + "\n")
            self.assertEqual(
                main(["--baseline", str(opd), "--outputs", str(cmt), "--tensorboard"]),
                0,
            )
            report = json.loads((cmt / "training_cost_comparison.json").read_text())
            self.assertEqual(report["overhead_vs_opd_percent"], 25)
            self.assertEqual(report["tensorboard_backfill"]["written_steps"], 1)
            events = EventAccumulator(str(cmt / "custom_tensorboard")).Reload()
            values = events.Scalars("cost/overhead_vs_opd_percent")
            self.assertEqual([(value.step, value.value) for value in values], [(2, 25)])


if __name__ == "__main__":
    unittest.main()
