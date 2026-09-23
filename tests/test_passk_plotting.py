from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.passk_plotting import load_passk_rows, plot_passk_results


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib") is not None,
    "matplotlib is optional on non-plotting compute environments",
)
class PassKPlottingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
