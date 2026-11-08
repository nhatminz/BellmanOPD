from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from b200_experiment.cli import _configured, build_parser
from b200_experiment.training_cost import comparison_signature


class GemmaLauncherTests(unittest.TestCase):
    def test_four_actual_launchers_resolve_matching_workloads_and_separate_outputs(
        self,
    ):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "commands.jsonl"
            fake_python = root / "python"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import os,sys,json\n"
                f"real={sys.executable!r}\n"
                "if sys.argv[1] == '-': os.execv(real,[real,*sys.argv[1:]])\n"
                "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            )
            fake_python.chmod(0o755)
            env = {
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    "OUTPUT_DIR",
                    "OUTPUT_ROOT",
                    "RESUME",
                    "RESUME_FROM_CHECKPOINT",
                    "COST_OPD_BASELINE",
                    "STUDENT_MODEL",
                    "TEACHER_MODEL",
                    "TRAIN_DATASET",
                    "MICRO",
                    "BATCH_SIZE",
                    "VLLM_RETURN_LOG_PROBS",
                    "TRAIN_NPROC_PER_NODE",
                }
            }
            env.update(
                PYTHON_BIN=str(fake_python),
                CAPTURE=str(capture),
                CUDA_VISIBLE_DEVICES="0,1,2,3",
                RUN_NAME="gemma_test",
                OUTPUT_ROOT=str(root / "outputs"),
            )
            completed = subprocess.run(
                ["bash", str(repo / "scripts/train_gemma2_b200.sh"), "all"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            commands = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(commands), 4)
            configs = [
                _configured(
                    build_parser().parse_args(command[command.index("train") :])
                )
                for command in commands
            ]
            self.assertEqual(
                [c["experiment"]["method"] for c in configs], ["opd", "ta", "iw", "cmt"]
            )
            self.assertEqual(len({comparison_signature(c, 4) for c in configs}), 1)
            self.assertEqual(len({c["experiment"]["output_dir"] for c in configs}), 4)
            for config in configs:
                self.assertTrue(
                    config["models"]["student_path"].endswith("/models/gemma-2-2b-it")
                )
                self.assertEqual(config["rollout"]["num_responses"], 4)
                self.assertEqual(config["training_evaluation"]["max_new_tokens"], 4096)
                self.assertEqual(
                    config["training_evaluation"]["vllm"]["max_model_len"], 8192
                )
            self.assertIsNone(configs[0]["logging"]["cost"]["opd_baseline"])
            for config in configs[1:]:
                self.assertEqual(
                    config["logging"]["cost"]["opd_baseline"], str(root / "outputs/opd")
                )


if __name__ == "__main__":
    unittest.main()
