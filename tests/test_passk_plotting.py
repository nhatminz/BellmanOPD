from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from b200_experiment.passk_plotting import load_passk_rows, plot_passk_results


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib") is not None,
    "matplotlib is optional on non-plotting compute environments",
)
class PassKPlottingTests(unittest.TestCase):
    @staticmethod
    def _synthetic_rows(method: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for benchmark, k_values in (
            ("AIME24", (8, 16, 32)),
            ("AIME25", (8, 16, 32)),
            ("AMC23", (4, 8, 16)),
        ):
            for k in k_values:
                rows.append(
                    {
                        "model_group": "qwen3_4b",
                        "method": method,
                        "label": method.upper(),
                        "checkpoint": f"/tmp/{method}/checkpoint-000600",
                        "benchmark": benchmark,
                        "k": k,
                        "pass_at_k": min(0.95, 0.2 + k / 100.0),
                    }
                )
        return rows

    def test_plotting_from_synthetic_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for model_group in ("qwen3_1.7b", "qwen3_4b"):
                for method in ("opd", "cmt"):
                    for benchmark, k_values in (
                        ("AIME24", (8, 16, 32)),
                        ("AIME25", (8, 16, 32)),
                        ("AMC23", (4, 8, 16)),
                    ):
                        for k in k_values:
                            rows.append(
                                {
                                    "model_group": model_group,
                                    "method": method,
                                    "label": method.upper(),
                                    "checkpoint": f"/tmp/{model_group}/{method}/final",
                                    "benchmark": benchmark,
                                    "k": k,
                                    "pass_at_k": min(0.95, 0.2 + k / 100.0),
                                }
                            )
            summary = root / "summary.json"
            summary.write_text(json.dumps({"results": rows}), encoding="utf-8")
            loaded = load_passk_rows([summary])
            outputs = plot_passk_results(loaded, root / "figures", tag="synthetic")

            self.assertEqual(set(outputs), {"qwen3_1.7b", "qwen3_4b"})
            for paths in outputs.values():
                self.assertTrue(Path(paths["png"]).is_file())
                self.assertTrue(Path(paths["pdf"]).is_file())

    def test_bash_plotter_accepts_output_run_names_and_creates_comparison_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs_root = root / "outputs"
            results_root = root / "results"
            run_names = (
                "opd_checkpoint_000600_qwen3_4b_source",
                "cmt_checkpoint_000600_qwen3_4b_source",
            )
            for run_name, method in zip(run_names, ("opd", "cmt")):
                summaries = outputs_root / run_name / "summaries"
                summaries.mkdir(parents=True)
                (summaries / f"passk_summary_{run_name}.json").write_text(
                    json.dumps({"results": self._synthetic_rows(method)}),
                    encoding="utf-8",
                )

            environment = os.environ.copy()
            environment.update(
                {
                    "MPLBACKEND": "Agg",
                    "PYTHON_BIN": "/usr/bin/python3",
                    "PASSK_OUTPUTS_ROOT": str(outputs_root),
                    "PASSK_RESULTS_ROOT": str(results_root),
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    "scripts/plot_passk_experiment.sh",
                    *run_names,
                ],
                cwd=repo_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            comparison = results_root / f"{run_names[0]}_vs_{run_names[1]}"
            self.assertTrue(comparison.is_dir())
            self.assertEqual(len(list(comparison.glob("*.png"))), 1)
            self.assertEqual(len(list(comparison.glob("*.pdf"))), 1)


if __name__ == "__main__":
    unittest.main()
