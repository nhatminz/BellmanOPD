from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import AutoModelForCausalLM, Gemma2Config, Gemma2ForCausalLM

from b200_experiment.config import load_with_overlays
from b200_experiment.distributed import DistributedContext
from b200_experiment.fsdp import _transformer_layer_classes
from b200_experiment.models import (
    choose_model_attention_implementation,
    generation_stop_token_ids,
)
from b200_experiment.opd_core import topk_reference_from_logits
from b200_experiment.scoring import (
    RolloutBatch,
    score_original_rollout,
    score_student_teacher_rollout,
    supports_response_only_logits,
)
from b200_experiment.trainer import _opd_train_step, _save_inference_snapshot


class Gemma2Tests(unittest.TestCase):
    def model(self):
        torch.manual_seed(7)
        return Gemma2ForCausalLM(
            Gemma2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                query_pre_attn_scalar=8,
                max_position_embeddings=64,
                sliding_window=16,
                attn_implementation="eager",
            )
        )

    def test_softcap_backend_and_chat_stopping(self):
        config = Gemma2Config()
        with patch(
            "b200_experiment.models.choose_attention_implementation",
            return_value="sdpa",
        ):
            self.assertEqual(
                choose_model_attention_implementation(config, "auto"), "eager"
            )
        with self.assertRaisesRegex(ValueError, "softcapping"):
            choose_model_attention_implementation(config, "sdpa")
        tokenizer = SimpleNamespace(
            eos_token_id=1, get_vocab=lambda: {"<eos>": 1, "<end_of_turn>": 107}
        )
        self.assertEqual(generation_stop_token_ids(tokenizer, {}), [1, 107])
        self.assertEqual(
            generation_stop_token_ids(
                tokenizer, {"models": {"stop_token_ids": [1, 107, 1]}}
            ),
            [1, 107],
        )

    def test_pruned_logits_match_full_logits_and_allow_optimizer_update(self):
        model = self.model().eval()
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8], [2, 4, 3, 5, 9, 7, 8]])
        attention = torch.ones_like(ids)
        valid = torch.ones((2, 4), dtype=torch.bool)
        rollout = RolloutBatch(
            ids, attention, ids[:, 3:], valid, torch.zeros((2, 4)), 3
        )
        with torch.no_grad():
            full = model(
                input_ids=ids[:, :-1], attention_mask=attention[:, :-1], use_cache=False
            ).logits[:, 2:]
        scores = score_original_rollout(model, rollout, top_k=16, micro_batch_size=1)
        joint_student, joint_teacher = score_student_teacher_rollout(
            model, model, rollout, top_k=16, micro_batch_size=1
        )
        torch.testing.assert_close(
            joint_student.top_k_log_probs, scores.top_k_log_probs
        )
        torch.testing.assert_close(
            joint_teacher.candidate_log_probs, scores.top_k_log_probs
        )
        expected = torch.log_softmax(full.float(), dim=-1)
        torch.testing.assert_close(
            scores.sampled_log_probs,
            expected.gather(-1, rollout.response_ids.unsqueeze(-1)).squeeze(-1),
            atol=1e-6,
            rtol=1e-5,
        )
        self.assertTrue(supports_response_only_logits(model))
        self.assertIn(
            type(model.model.layers[0]), _transformer_layer_classes(model, {})
        )
        teacher_logits = full.clone()
        teacher_logits[..., 20] += 3
        reference = topk_reference_from_logits(full, teacher_logits, valid, top_k=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.lm_head.weight.detach().clone()
        config = {
            "experiment": {"method": "opd"},
            "rollout": {"temperature": 1.0},
            "selector": {"score_chunk_steps": 2},
            "training": {
                "micro_batch_size_per_gpu": 1,
                "ppo_mini_batch_size": 2,
                "max_grad_norm": 1.0,
            },
        }
        metrics = _opd_train_step(
            model,
            optimizer,
            rollout,
            valid.float(),
            reference,
            config,
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
        )
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        self.assertGreater(metrics["minibatches"][0]["optimizer_step_wall_time_sec"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            _save_inference_snapshot(model, Mock(), Path(temporary))
            restored = AutoModelForCausalLM.from_pretrained(
                temporary, attn_implementation="eager"
            )
            torch.testing.assert_close(restored.lm_head.weight, model.lm_head.weight)

    def test_overlay_keeps_method_definitions_and_b200_paths(self):
        root = Path(__file__).resolve().parents[1]
        for method in ("opd", "ta", "iw", "cmt"):
            config = load_with_overlays(
                root / f"configs/qwen3_b200_{method}.yaml",
                [root / "configs/gemma2_b200.yaml"],
            )
            self.assertEqual(config["experiment"]["method"], method)
            self.assertEqual(config["models"]["student_path"], "models/gemma-2-2b-it")
            self.assertEqual(config["models"]["teacher_path"], "models/gemma-2-9b-it")
            self.assertEqual(config["models"]["stop_token_ids"], [1, 107])
            self.assertEqual(config["rollout"]["vllm"]["max_model_len"], 8192)
            self.assertEqual(
                config["paths"]["storage_root"], "/workspace/storage-shared"
            )
            self.assertEqual(config["training"]["micro_batch_size_per_gpu"], 1)


if __name__ == "__main__":
    unittest.main()
