from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from b200_experiment.cli import _configured, build_parser
from b200_experiment.resume import validate_resume_config
from b200_experiment.training_cost import comparison_signature


class GemmaStandaloneLauncherTests(unittest.TestCase):
    def launch_configs(self, root, methods, overrides=None):
        repo = Path(__file__).resolve().parents[1]
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
            "PATH": os.environ["PATH"],
            "PYTHON_BIN": str(fake_python),
            "CAPTURE": str(capture),
            "RUN_NAME": "gemma_standalone_test",
            "OUTPUT_ROOT": str(root / "outputs"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "STORAGE_ROOT": "/workspace/storage-shared",
        }
        env.update(overrides or {})
        for method in methods:
            command = ["bash", str(repo / f"scripts/train_{method}_b200_gemma.sh")]
            result = subprocess.run(
                command, cwd=repo, env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Cost logs:", result.stdout)
            self.assertIn("OPD cost baseline:", result.stdout)
        commands = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertTrue(all("--overlay" not in command for command in commands))
        return [
            _configured(build_parser().parse_args(command[command.index("train") :]))
            for command in commands
        ]

    def test_four_standalone_scripts_keep_the_supplied_cmt_experiment_fair(self):
        with tempfile.TemporaryDirectory() as temporary:
            configs = self.launch_configs(Path(temporary), ("cmt", "opd", "ta", "iw"))
            self.assertEqual(
                [c["experiment"]["method"] for c in configs], ["cmt", "opd", "ta", "iw"]
            )
            self.assertEqual(len({comparison_signature(c, 4) for c in configs}), 1)
            for config in configs:
                self.assertEqual(
                    config["models"]["teacher_path"],
                    "/workspace/storage-shared/nlp/minhpn19/models/gemma-2-9b-it",
                )
                self.assertEqual(
                    config["models"]["student_path"],
                    "/workspace/storage-shared/nlp/minhpn19/models/gemma-2-2b-it",
                )
                self.assertEqual(config["rollout"]["batch_size"], 64)
                self.assertEqual(config["rollout"]["num_responses"], 4)
                self.assertEqual(config["training"]["ppo_mini_batch_size"], 64)
                self.assertEqual(config["training"]["micro_batch_size_per_gpu"], 4)
                self.assertEqual(config["selector"]["score_micro_batch_size"], 8)
                self.assertEqual(config["training"]["epochs"], 3)
                self.assertEqual(config["training"]["learning_rate"], 5e-6)
                self.assertEqual(config["rollout"]["max_new_tokens"], 4096)
                self.assertEqual(
                    config["rollout"]["vllm"]["gpu_memory_utilization"], 0.5
                )
                self.assertEqual(config["rollout"]["vllm"]["max_model_len"], 5200)
                self.assertEqual(config["training"]["save_interval"], 150)
                self.assertFalse(config["training_evaluation"]["enabled"])
                self.assertTrue(config["logging"]["cost"]["enabled"])
                self.assertTrue(config["logging"]["tensorboard"]["enabled"])
                self.assertTrue(config["logging"]["token_score_stats_enabled"])
                self.assertEqual(config["logging"]["token_score_interval"], 100)
            self.assertEqual(
                configs[0]["selector"]["cmt_allocation_mode"], "direct_bounded_gibbs"
            )
            self.assertEqual(configs[0]["selector"]["cmt_final_allocation_kl"], 0.02)
            self.assertEqual(configs[3]["iw_opd"]["weight_max"], 1.5)

    def test_resume_command_changes_only_allowed_micro_runtime_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "cmt_20261108_103126_485124171/cmt_opd"
            checkpoint = run / "checkpoint-000150"
            checkpoint.mkdir(parents=True)
            configs = self.launch_configs(
                root,
                ("cmt",),
                {
                    "RUN_NAME": "cmt_20261108_103126_485124171",
                    "OUTPUT_DIR": str(run),
                    "RESUME_FROM_CHECKPOINT": "auto",
                    "MAX_STEPS": "2250",
                    "MICRO": "4",
                },
            )
            current = configs[0]
            old = copy.deepcopy(current)
            old["training"]["micro_batch_size_per_gpu"] = 16
            (run / "resolved_config.yaml").write_text(yaml.safe_dump(old))
            validation = validate_resume_config(checkpoint, current)
            self.assertTrue(validation["checked"])
            self.assertEqual(validation["mismatches"], {})
            self.assertEqual(current["experiment"]["output_dir"], str(run))
            self.assertEqual(current["training"]["resume_from_checkpoint"], "auto")
            self.assertEqual(current["training"]["max_steps"], 2250)
            self.assertFalse(current["training"]["resume_allow_config_mismatch"])

    def test_runtime_overrides_and_dapo_are_the_same_across_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            configs = self.launch_configs(
                Path(temporary),
                ("cmt", "opd", "ta", "iw"),
                {
                    "MICRO": "2",
                    "SCORE_MICRO_BATCH_SIZE": "2",
                    "TRAIN_DATASET": "dapo_math",
                    "COST_OPD_BASELINE": "/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/gemma_opd/opd",
                },
            )
            self.assertEqual(len({comparison_signature(c, 4) for c in configs}), 1)
            for config in configs:
                self.assertEqual(config["training"]["micro_batch_size_per_gpu"], 2)
                self.assertEqual(config["selector"]["score_micro_batch_size"], 2)
                self.assertEqual(config["data"]["split"], "all")
                self.assertEqual(config["training"]["epochs"], 2)
                self.assertEqual(config["data"]["prompt_key"], "prompt")
                self.assertTrue(
                    config["logging"]["cost"]["opd_baseline"].endswith(
                        "/outputs/gemma_opd/opd"
                    )
                )


if __name__ == "__main__":
    unittest.main()
