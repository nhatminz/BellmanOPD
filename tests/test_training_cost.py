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


if __name__ == "__main__":
    unittest.main()
