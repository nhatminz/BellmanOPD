"""Optional real simulator + GPU toy optimizer check; NOT model task performance."""

import copy
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not os.environ.get("ALFWORLD_REAL_DATA"),
    reason="requires real raw parquet and TextWorld",
)
def test_real_scripted_id_ood_evaluation_three_seed_summary(tmp_path):
    from b200_experiment.alfworld_cli import readiness, DiagnosticTokenizer
    from b200_experiment.config import load_config
    from b200_experiment.alfworld_env import load_task_splits
    from b200_experiment.alfworld_evaluation import (
        evaluate_tasks,
        per_seed_results,
        aggregate,
    )

    root = Path(os.environ["ALFWORLD_REAL_DATA"])
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/alfworld_multiturn.yaml"
    )
    config["data"]["path"] = str(root / "train-00000-of-00001.parquet")
    config["alfworld"]["eval_paths"] = {
        s: str(root / f"{s}-00000-of-00001.parquet")
        for s in ["eval_in_distribution", "eval_out_of_distribution"]
    }
    config["alfworld"]["game_cache_dir"] = str(tmp_path / "games")
    scripts = {
        s: [
            step["action"]
            for step in readiness(config, split=s, smoke=True)[
                "oracle_simulator_smoke"
            ]["steps"]
        ]
        for s in ["eval_in_distribution", "eval_out_of_distribution"]
    }
    tasks = load_task_splits(config)
    results = []
    for seed in [42, 43, 44]:
        traces = []
        for split, commands in scripts.items():
            tok, actions = DiagnosticTokenizer(), iter(commands)

            def generate(ids, sample_seed, budget):
                return tok.encode(next(actions) + ";") + [1]

            traces.extend(
                evaluate_tasks([tasks[split][0]], tok, config, generate, seed)
            )
        assert len(traces) == 2 and all(t["success"] for t in traces)
        results.append(per_seed_results(traces))
    summary = aggregate(results)
    for split in scripts:
        assert summary[split]["task_success_rate"]["mean"] == 1.0
        assert summary[split]["task_success_rate"]["std"] == 0.0
        assert summary[split]["task_success_rate"]["n"] == 3
        assert summary[split]["parser_failure_rate"]["mean"] == 0.0
        assert summary[split]["punctuation_recovery_rate"]["mean"] == 1.0
    (tmp_path / "scripted_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("ALFWORLD_REAL_DATA"),
    reason="requires CUDA and ALFWORLD_REAL_DATA; random tiny models only",
)
@pytest.mark.parametrize("method", ["opd", "cmt"])
def test_real_episode_gpu_optimizer_mechanics(tmp_path, method):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from b200_experiment.config import load_config
    from b200_experiment.alfworld_cli import readiness, DiagnosticTokenizer
    from b200_experiment.alfworld_env import load_task_splits
    from b200_experiment.alfworld_rollout import collect_episode, collate_episodes
    from b200_experiment.scoring import score_student_teacher_rollout
    from b200_experiment.opd_core import build_student_topk_opd_reference
    from b200_experiment.selectors.pgt_selector import PGTSelector
    from b200_experiment.selectors.cmt_selector import CMTSelector
    from b200_experiment.trainer import _opd_train_step
    from b200_experiment.distributed import DistributedContext

    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/alfworld_multiturn.yaml"
    )
    root = Path(os.environ["ALFWORLD_REAL_DATA"])
    cfg["data"]["path"] = str(root / "train-00000-of-00001.parquet")
    cfg["alfworld"]["eval_paths"] = {
        s: str(root / f"{s}-00000-of-00001.parquet")
        for s in ["eval_in_distribution", "eval_out_of_distribution"]
    }
    cfg["alfworld"]["game_cache_dir"] = str(tmp_path / "games")
    # Oracle-assisted ONLY for this explicitly scripted installation/mechanics test.
    smoke = readiness(cfg, smoke=True)
    commands = iter(step["action"] for step in smoke["oracle_simulator_smoke"]["steps"])
    tokenizer_path = os.environ.get("ALFWORLD_TEST_TOKENIZER")
    tok = (
        AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if tokenizer_path
        else DiagnosticTokenizer()
    )

    def generate(ids, seed, budget):
        return tok.encode(next(commands) + ";", add_special_tokens=False) + [
            tok.eos_token_id
        ]

    task = load_task_splits(cfg)["eval_in_distribution"][0]
    episode = collect_episode(task, tok, cfg, generate, 42)
    assert episode["trace"]["success"]
    batch = collate_episodes([episode], tok.pad_token_id, "cuda:0")
    torch.manual_seed(31)
    vocab = max(1024, int(batch.input_ids.max()) + 1)
    mc = Qwen3Config(
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(mc).cuda()
    teacher = Qwen3ForCausalLM(copy.deepcopy(mc)).cuda().requires_grad_(False)
    p, q = score_student_teacher_rollout(
        model, teacher, batch, top_k=4, micro_batch_size=1
    )
    reference = build_student_topk_opd_reference(
        p.top_k_ids,
        p.top_k_log_probs,
        q.candidate_log_probs,
        batch.valid_mask,
        top_k=4,
        full_vocab_ppo=True,
    )
    kwargs = {}
    if method == "cmt":
        support = PGTSelector().compute_scores_from_topk(
            p.top_k_ids,
            q.top_k_ids,
            p.top_k_log_probs,
            q.candidate_log_probs,
            q.top_k_log_probs,
            p.candidate_log_probs,
            batch.valid_mask,
            gain_support="student_topk",
        )
        score = CMTSelector().compute_scores(
            support, batch.response_ids, batch.valid_mask
        )
        kwargs = {
            "gibbs_scores": score.scores,
            "gibbs_epsilon": 0.02,
            "gibbs_mode": "direct_bounded_gibbs",
        }
    before = model.lm_head.weight.detach().clone()
    cfg["experiment"]["method"] = method
    cfg["training"].update(ppo_mini_batch_size=1, micro_batch_size_per_gpu=1)
    result = _opd_train_step(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        batch,
        batch.valid_mask.float(),
        reference,
        cfg,
        torch.device("cuda:0"),
        DistributedContext(0, 0, 1, torch.device("cuda:0")),
        **kwargs,
    )
    assert result["gradient_norm"] > 0
    assert not torch.equal(before, model.lm_head.weight)
    assert abs(result["ratio_mean"] - 1) < 1e-4
    assert all(param.grad is None for param in teacher.parameters())
    print(
        json.dumps(
            {
                "method": method,
                "loss": result["loss"],
                "gradient_norm": result["gradient_norm"],
                "ratio_mean": result["ratio_mean"],
                "parameters_changed": True,
                "kind": "random tiny model, scripted real environment trajectory, NOT trained policy performance",
            }
        )
    )
