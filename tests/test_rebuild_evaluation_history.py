from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from b200_experiment.rebuild_evaluation_history import rebuild_evaluation_history


class RebuildEvaluationHistoryTests(unittest.TestCase):
    @staticmethod
    def _write_step(root: Path, step: int, value: float) -> None:
        step_dir = root / f"step-{step:06d}"
        step_dir.mkdir(parents=True)
        predictions = step_dir / "math_500_predictions.jsonl.gz"
        detailed = step_dir / "model_outputs_detailed.jsonl.gz"
        predictions.write_bytes(b"predictions")
        detailed.write_bytes(b"details")
        suite = {
            "parameters": {
                "backend": "vllm",
                "metric": "pass@8",
                "num_responses": 8,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            "benchmarks": {
                "MATH-500": {
                    "correct": 1,
                    "total": 8,
                    "accuracy": value,
                    "avg_at_n": value,
                    "pass_at_k": value,
                    "pass_at_8": value,
                    "problems": 1,
                    "samples_per_problem": 8,
                    "predictions": str(predictions),
                }
            },
            "detailed_outputs": str(detailed),
        }
        (step_dir / "summary.json").write_text(
            json.dumps(suite), encoding="utf-8"
        )

    def test_rebuilds_only_committed_steps_after_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_output = Path(temporary) / "cmt_opd"
            run_output.mkdir()
            (run_output / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {"training_evaluation": {"output_subdir": "training_eval"}}
                ),
                encoding="utf-8",
            )
            (run_output / "summary.json").write_text(
                json.dumps({"steps": 300}), encoding="utf-8"
            )
            eval_root = run_output / "training_eval_pass_at_8"
            self._write_step(eval_root, 0, 0.25)
            self._write_step(eval_root, 150, 0.5)
            (eval_root / "step-000300").mkdir(parents=True)
            (eval_root / ".reeval-step-000300-interrupted").mkdir()

            result = rebuild_evaluation_history(run_output, "cmt")

            self.assertEqual(result["recovered_steps"], [0, 150])
            self.assertEqual(len(result["skipped_incomplete_directories"]), 1)
            history_path = run_output / "eval_history_pass_at_8.jsonl"
            rows = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["step"] for row in rows], [0, 150])
            self.assertTrue(all(row["max_steps"] == 300 for row in rows))
            self.assertTrue(all(row["parameters"]["metric"] == "pass@8" for row in rows))
            with (run_output / "eval_metrics_pass_at_8.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual([int(row["step"]) for row in metrics], [0, 150])


if __name__ == "__main__":
    unittest.main()
