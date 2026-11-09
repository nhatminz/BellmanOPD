from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from b200_experiment.multiseed_evaluation import BENCHMARKS, execute


class MultiseedEvaluationTests(unittest.TestCase):
    def make_run(self, root):
        run = root / "old_checkout/outputs/run/cmt_opd"
        run.mkdir(parents=True)
        config = {
            "experiment": {"method": "cmt"},
            "paths": {"storage_root": str(root)},
            "models": {
                "student_path": "models/student",
                "teacher_path": "models/teacher",
            },
            "data": {"path": "train.jsonl"},
            "training_evaluation": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 4096,
                "vllm": {"seed": 1234},
            },
            "evaluation": {
                "benchmarks": {
                    name: {"path": str(root / (name + ".jsonl"))} for name in BENCHMARKS
                }
            },
        }
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        for step in (150, 300):
            checkpoint = run / f"checkpoint-{step:06d}"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
        for metric, filename, score in [
            ("avg@8", "eval_history.jsonl", 0.25),
            ("pass@8", "eval_history_pass_at_8.jsonl", 0.5),
        ]:
            rows = [
                {
                    "step": step,
                    "method": "cmt",
                    "parameters": {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "max_new_tokens": 4096,
                        "num_responses": 8,
                        "seed": 1234,
                        "metric": metric,
                        "limit": None,
                    },
                    "benchmarks": {
                        name: {
                            "accuracy": score,
                            "metric": metric,
                            "samples_per_problem": 8,
                            "problems": 2,
                        }
                        for name in BENCHMARKS
                    },
                }
                for step in (150, 300)
            ]
            (run / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
        return run

    def args(self, run, **kwargs):
        values = dict(
            run_output=run,
            checkpoints=["150"],
            seeds=[43, 44],
            baseline_seed=None,
            tag="extra_seeds",
            dry_run=False,
            world_size=2,
            max_num_seqs=None,
            max_model_len=None,
            gpu_memory_utilization=None,
        )
        values.update(kwargs)
        return argparse.Namespace(**values)

    def evaluator(self, calls):
        def evaluate(name, checkpoint, config, output, settings, resolved, world_size):
            seed = settings["vllm"]["seed"]
            calls.append((Path(checkpoint).name, seed, copy.deepcopy(settings)))
            self.assertEqual(world_size, 2)
            self.assertTrue(resolved.is_file())
            value = 0.375 if seed == 43 else 0.5
            results = {}
            for benchmark in BENCHMARKS:
                predictions = output / (benchmark + ".jsonl")
                predictions.write_text("test predictions\n")
                results[benchmark] = {
                    "avg_at_8": value,
                    "pass_at_8": 0.75 if seed == 43 else 1.0,
                    "accuracy": value,
                    "metric": "avg@8",
                    "samples_per_problem": 8,
                    "problems": 2,
                    "total": 16,
                    "correct": int(16 * value),
                    "predictions": str(predictions),
                }
            return {
                "benchmarks": results,
                "parameters": {
                    key: settings[key]
                    for key in (
                        "temperature",
                        "top_p",
                        "max_new_tokens",
                        "num_responses",
                    )
                }
                | {"seed": seed, "metric": "avg@8"},
            }

        return evaluate

    def test_selected_checkpoint_dual_metrics_summary_resume_and_legacy_preservation(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            before = {p.name: p.read_bytes() for p in run.glob("*.jsonl")}
            calls = []
            # Accept the parent run directory even in a different checkout.
            args = self.args(run.parent)
            summary = execute(args, self.evaluator(calls))
            self.assertEqual(
                [(name, seed) for name, seed, _ in calls],
                [("checkpoint-000150", 43), ("checkpoint-000150", 44)],
            )
            self.assertEqual(len(summary["results"]), 12)
            avg, passed = summary["results"][:2]
            self.assertEqual(avg["seeds"], [1234, 43, 44])
            self.assertAlmostEqual(avg["mean"], 0.375)
            self.assertAlmostEqual(avg["std"], 0.125)  # ddof=1
            self.assertEqual(avg["mean_pm_std"], "37.50 ± 12.50")
            self.assertEqual(passed["mean_pm_std"], "75.00 ± 25.00")
            for filename, content in before.items():
                self.assertEqual((run / filename).read_bytes(), content)
            avg_row = json.loads(
                (run / "eval_history_seed_43_extra_seeds.jsonl").read_text()
            )
            pass_row = json.loads(
                (run / "eval_history_pass_at_8_seed_43_extra_seeds.jsonl").read_text()
            )
            self.assertEqual(avg_row["benchmarks"][BENCHMARKS[0]]["accuracy"], 0.375)
            self.assertEqual(pass_row["benchmarks"][BENCHMARKS[0]]["accuracy"], 0.75)
            self.assertTrue(Path(avg_row["details"]).is_file())
            self.assertTrue(
                Path(avg_row["benchmarks"][BENCHMARKS[0]]["predictions"]).is_file()
            )
            execute(
                args, lambda *a, **k: self.fail("Committed seed result must be reused")
            )
            self.assertTrue((run / "eval_summary_extra_seeds.md").is_file())
            self.assertFalse(
                (run / "eval_multiseed/extra_seeds/checkpoint-000300").exists()
            )

    def test_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            before = sorted(run.rglob("*"))
            result = execute(self.args(run, dry_run=True))
            self.assertTrue(result["dry_run"])
            self.assertEqual(sorted(run.rglob("*")), before)

    def test_invalid_baselines_fail_before_generation(self):
        for mutation, error in [
            ("same_seed", "differ"),
            ("protocol", "different sampling"),
            ("missing_dataset", "Missing old"),
            ("unknown_seed", "actual old EVAL"),
            ("duplicate", "Ambiguous duplicate"),
        ]:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                run = self.make_run(Path(temporary))
                path = run / "eval_history_pass_at_8.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                args = self.args(run)
                if mutation == "same_seed":
                    args.seeds = [1234, 44]
                elif mutation == "protocol":
                    rows[0]["parameters"]["temperature"] = 1.0
                elif mutation == "missing_dataset":
                    del rows[0]["benchmarks"]["AMC23"]
                elif mutation == "unknown_seed":
                    del rows[0]["parameters"]["seed"]
                else:
                    rows.append(rows[0])
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, error):
                    execute(
                        args,
                        lambda *a, **k: self.fail(
                            "Invalid baseline must not start GPU evaluation"
                        ),
                    )

    def test_partial_failure_resumes_only_unfinished_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            calls = []
            evaluate = self.evaluator(calls)

            def interrupted(*args, **kwargs):
                if args[4]["vllm"]["seed"] == 44:
                    raise RuntimeError("interrupted")
                return evaluate(*args, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                execute(self.args(run), interrupted)
            self.assertFalse((run / "eval_summary_extra_seeds.json").exists())
            execute(self.args(run), evaluate)
            self.assertEqual([seed for _, seed, _ in calls], [43, 44])

    def test_existing_seed_cannot_be_reused_with_changed_checkpoint_or_engine_request(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            execute(self.args(run), self.evaluator([]))
            with self.assertRaisesRegex(ValueError, "different request"):
                execute(self.args(run, world_size=1), self.evaluator([]))
            (run / "checkpoint-000150/config.json").write_text('{"modified": true}')
            with self.assertRaisesRegex(ValueError, "different request"):
                execute(self.args(run), self.evaluator([]))

    def test_fresh_three_seeds_without_old_histories_summarizes_all_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            for name in ("eval_history.jsonl", "eval_history_pass_at_8.jsonl"):
                (run / name).unlink()
            args = self.args(
                run, checkpoints=["150", "300"], seeds=[42, 43, 44], fresh_seeds=True
            )
            calls = []
            summary = execute(args, self.evaluator(calls))
            self.assertEqual(
                [(name, seed) for name, seed, _ in calls],
                [
                    (f"checkpoint-{step:06d}", seed)
                    for step in [150, 300]
                    for seed in [42, 43, 44]
                ],
            )
            self.assertEqual(len(summary["results"]), 24)
            for result in summary["results"]:
                self.assertEqual(result["seeds"], [42, 43, 44])
                self.assertEqual(result["n"], 3)
                self.assertIsNone(result["baseline_history"])
            avg, passed = summary["results"][:2]
            self.assertAlmostEqual(avg["mean"], (0.5 + 0.375 + 0.5) / 3)
            self.assertAlmostEqual(avg["std"], 0.125 / (3**0.5))
            self.assertAlmostEqual(passed["std"], 0.25 / (3**0.5))
            self.assertTrue((run / "eval_history_seed_42_extra_seeds.jsonl").is_file())
            self.assertFalse((run / "eval_history.jsonl").exists())
            execute(
                args,
                lambda *a, **k: self.fail("Completed requested seed must be reused"),
            )

    def test_fresh_seeds_ignore_conflicting_old_baselines_and_keep_them_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            path = run / "eval_history_pass_at_8.jsonl"
            path.write_text("old incomplete or malformed history\n")
            before = {p.name: p.read_bytes() for p in run.glob("*.jsonl")}
            args = self.args(run, seeds=[42, 43, 44], fresh_seeds=True)
            execute(args, self.evaluator([]))
            for name, data in before.items():
                self.assertEqual((run / name).read_bytes(), data)

    def test_fresh_seed_sample_std_requires_two_seeds_and_no_baseline_argument(self):
        for overrides, error in [
            ({"seeds": [42]}, "at least two"),
            ({"baseline_seed": 42}, "cannot be combined"),
        ]:
            with (
                self.subTest(overrides=overrides),
                tempfile.TemporaryDirectory() as temporary,
            ):
                run = self.make_run(Path(temporary))
                with self.assertRaisesRegex(ValueError, error):
                    execute(
                        self.args(run, fresh_seeds=True, **overrides),
                        lambda *a, **k: self.fail("Invalid request must not evaluate"),
                    )

    def test_fresh_seed_summary_rejects_changed_dataset_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            evaluate = self.evaluator([])

            def changed_count(*args, **kwargs):
                result = evaluate(*args, **kwargs)
                if result["parameters"]["seed"] == 44:
                    result["benchmarks"][BENCHMARKS[0]]["problems"] = 3
                return result

            with self.assertRaisesRegex(ValueError, "problem count changed"):
                execute(
                    self.args(run, seeds=[42, 43, 44], fresh_seeds=True), changed_count
                )


if __name__ == "__main__":
    unittest.main()
