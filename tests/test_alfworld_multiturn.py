import copy
import json
import os
from pathlib import Path

import pytest
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from b200_experiment.alfworld_env import (
    EnvironmentSetupError,
    load_task_splits,
    materialize_game,
    normalize_task,
)
from b200_experiment.alfworld_rollout import (
    collect_episode,
    collate_episodes,
    parse_action,
)
from b200_experiment.config import load_config, resolve_runtime_paths, save_config
from b200_experiment.opd_core import (
    compute_iw_opd_weights,
    topk_reference_from_logits,
    build_iw_opd_reference,
)
from b200_experiment.scoring import (
    RolloutBatch,
    rollout_forward_inputs,
    select_rollout_logits,
    score_original_rollout,
    score_student_teacher_rollout,
)
from b200_experiment.selectors.cmt_selector import CMTSelector
from b200_experiment.selectors.pgt_selector import PGTOutput

REPO = Path(__file__).resolve().parents[1]


def make_alfconfig(tmp_path):
    config = load_config(REPO / "configs/alfworld_multiturn.yaml")
    for split in ("train", "eval_in_distribution", "eval_out_of_distribution"):
        game = {
            "pddl_domain": "domain",
            "pddl_problem": f"problem {split}",
            "grammar": "grammar",
            "walkthrough": ["EXPERT MUST NOT LEAK"],
        }
        row = {
            "id": f"id-{split}",
            "task_type": "pick",
            "game_file_path": f"pick-{split}/trial/game.tw-pddl",
            "game_content": json.dumps(game),
        }
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist([row]), path)
        if split == "train":
            config["data"]["path"] = str(path)
        else:
            config["alfworld"]["eval_paths"][split] = str(path)
    config["alfworld"].update(
        game_cache_dir=str(tmp_path / "cache"),
        max_turn_tokens=128,
        max_episode_tokens=2048,
    )
    return resolve_runtime_paths(config)


@pytest.fixture
def alfconfig(tmp_path):
    return make_alfconfig(tmp_path)


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def get_vocab(self):
        return {"EOS": 1}

    def encode(self, text, **kwargs):
        return [ord(c) + 3 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 3) for i in ids if i > 2)

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(f"<{m['role']}>\n{m['content']}</{m['role']}>\n" for m in messages)
            + "<assistant>\n"
        )


class InteractiveEnv:
    instances = []

    def __init__(self, task, settings):
        self.actions = []
        self.closed = False
        self.instances.append(self)

    def reset(self, seed):
        self.seed = seed
        return dict(
            observation="Task: take cup. You see cabinet 1.",
            reward=0.0,
            terminated=False,
            success=False,
            admissible_actions=["open cabinet 1"],
        )

    def step(self, action):
        self.actions.append(action)
        won = self.actions == ["open cabinet 1", "take cup 1 from cabinet 1"]
        return dict(
            observation="You took the cup."
            if won
            else "Cabinet is open. You see cup 1.",
            reward=float(won),
            terminated=won,
            success=won,
            admissible_actions=["take cup 1 from cabinet 1"],
        )

    def close(self):
        self.closed = True


def episode(
    config,
    *,
    responses=("open cabinet 1", "take cup 1 from cabinet 1"),
    env_factory=InteractiveEnv,
):
    task = load_task_splits(config)["train"][0]
    tok = Tokenizer()
    calls = []
    iterator = iter(responses)

    def generate(ids, seed, budget):
        calls.append((list(ids), seed, budget))
        return tok.encode(next(iterator)) + [1]

    return collect_episode(
        task, tok, config, generate, 42, env_factory=env_factory
    ), calls


def test_actual_parquet_schema_materialization_and_isolation(alfconfig):
    splits = load_task_splits(alfconfig)
    assert set(splits) == {"train", "eval_in_distribution", "eval_out_of_distribution"}
    task = splits["train"][0]
    assert task["id"] == "id-train"
    game = materialize_game(task, alfconfig["alfworld"]["game_cache_dir"])
    assert set(json.loads(game.read_text())) == {
        "pddl_domain",
        "pddl_problem",
        "grammar",
    }
    assert "EXPERT" not in game.read_text()
    assert materialize_game(task, alfconfig["alfworld"]["game_cache_dir"]) == game
    alfconfig["alfworld"]["eval_paths"]["eval_in_distribution"] = alfconfig["data"][
        "path"
    ]
    with pytest.raises(EnvironmentSetupError, match="Duplicate/leaked"):
        load_task_splits(alfconfig)


def test_different_ids_cannot_hide_leaked_games(alfconfig):
    train = pq.read_table(alfconfig["data"]["path"]).to_pylist()[0]
    train["id"] = "different-id"
    train["game_file_path"] = "different/path/game.tw-pddl"
    pq.write_table(
        pa.Table.from_pylist([train]),
        alfconfig["alfworld"]["eval_paths"]["eval_in_distribution"],
    )
    with pytest.raises(EnvironmentSetupError, match="Duplicate/leaked"):
        load_task_splits(alfconfig)


def test_missing_assets_and_static_response_are_setup_errors(tmp_path):
    for row in (
        {"id": "x", "response": "fake observation"},
        {"id": "x", "game_file_path": "missing.tw-pddl"},
    ):
        with pytest.raises(EnvironmentSetupError):
            normalize_task(row, "train", tmp_path / "data.parquet")
    asset = tmp_path / "real.tw-pddl"
    asset.write_text(
        json.dumps({"pddl_domain": "d", "pddl_problem": "p", "grammar": "g"})
    )
    task = normalize_task(
        {"game_file_path": asset.name}, "train", tmp_path / "data.parquet"
    )
    assert json.loads(task["game_content"])["pddl_problem"] == "p"


@pytest.mark.parametrize(
    "response,expected",
    [
        ("move cup 1 to table 1", "move cup 1 to table 1"),
        ('{"action":"look"}', "look"),
        ("<action>inventory</action>", "inventory"),
        ("look\ninventory", None),
        ("look; inventory", None),
        ('{"action":"look","other":1}', None),
        ("[]", None),
        ("I will look", None),
    ],
)
def test_one_action_parser(response, expected):
    assert parse_action(response) == expected


def test_real_sequence_of_mock_transitions_and_actor_only_mask(alfconfig):
    item, calls = episode(alfconfig)
    trace = item["trace"]
    assert trace["success"] and len(trace["turns"]) == 2
    assert "Cabinet is open" in Tokenizer().decode(calls[1][0])
    assert "open cabinet 1" in Tokenizer().decode(calls[1][0])
    assert calls[1][0][: len(calls[0][0])] == calls[0][0]
    batch = collate_episodes([item], 0, "cpu")
    expected = [
        token for turn in trace["turns"] for token in turn["response_token_ids"]
    ]
    assert batch.response_ids[0].tolist() == expected
    assert batch.actor_token_mask.sum() == len(expected)
    assert batch.valid_mask.sum() == len(expected)
    for turn in trace["turns"]:
        assert batch.actor_token_mask[0, turn["input_start"] : turn["input_end"]].all()
    gap = slice(trace["turns"][0]["input_end"], trace["turns"][1]["input_start"])
    assert not batch.actor_token_mask[0, gap].any()
    assert (batch.actor_positions[0, 1:] > batch.actor_positions[0, :-1]).all()
    assert InteractiveEnv.instances[-1].closed


def test_malformed_action_does_not_execute_a_simulator_step(alfconfig):
    item, _ = episode(
        alfconfig,
        responses=("look\ninventory", "open cabinet 1", "take cup 1 from cabinet 1"),
    )
    assert item["trace"]["success"]
    assert item["trace"]["invalid_actions"] == 1
    assert item["trace"]["turns"][0]["feedback_source"] == "action_parser"
    assert InteractiveEnv.instances[-1].actions == [
        "open cabinet 1",
        "take cup 1 from cabinet 1",
    ]


def test_eos_budget_and_timeout_are_explicit_episode_boundaries(alfconfig):
    item, _ = episode(alfconfig, responses=("",))
    assert item["trace"]["termination_reason"] == "agent_eos"
    assert not InteractiveEnv.instances[-1].actions
    alfconfig["alfworld"]["max_turns"] = 1
    item, _ = episode(alfconfig, responses=("open cabinet 1",))
    assert item["trace"]["termination_reason"] == "max_turns"
    alfconfig["alfworld"]["max_turns"] = 30
    item, _ = episode(alfconfig)
    alfconfig["alfworld"]["max_episode_tokens"] = (
        item["trace"]["turns"][0]["input_end"] + 1
    )
    item, _ = episode(alfconfig, responses=("open cabinet 1",))
    assert item["trace"]["termination_reason"] == "context_budget"

    class TimedOut(InteractiveEnv):
        def step(self, action):
            raise TimeoutError()

    item, _ = episode(alfconfig, responses=("open cabinet 1",), env_factory=TimedOut)
    assert item["trace"]["termination_reason"] == "environment_timeout"


@pytest.fixture
def tiny_models():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config), Qwen3ForCausalLM(config)


def compact_batch():
    # Observation tokens 20/21/22 occupy three physical slots between turns.
    return collate_episodes(
        [
            {
                "input_ids": [2, 3, 4, 5, 20, 21, 22, 6, 7],
                "actor_positions": [2, 3, 7, 8],
                "trace": {},
            },
            {
                "input_ids": [8, 9, 10, 25, 26, 11],
                "actor_positions": [2, 5],
                "trace": {},
            },
        ],
        0,
        "cpu",
    )


def test_real_qwen_causal_alignment_both_scoring_paths(tiny_models):
    student, teacher = tiny_models
    student.eval()
    teacher.eval()
    batch = compact_batch()
    with torch.no_grad():
        full = student(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits
        expected = (
            torch.log_softmax(full, -1)
            .gather(
                1,
                (batch.actor_positions - 1)
                .clamp_min(0)
                .unsqueeze(-1)
                .expand(-1, -1, 32),
            )
            .gather(-1, batch.response_ids.unsqueeze(-1))
            .squeeze(-1)
        )
    scores = score_original_rollout(student, batch, top_k=4, micro_batch_size=1)
    torch.testing.assert_close(
        scores.sampled_log_probs[batch.valid_mask],
        expected[batch.valid_mask],
        atol=1e-6,
        rtol=1e-6,
    )
    joint = score_student_teacher_rollout(
        student, teacher, batch, top_k=4, micro_batch_size=2
    )
    torch.testing.assert_close(
        joint[0].sampled_log_probs[batch.valid_mask],
        expected[batch.valid_mask],
        atol=1e-6,
        rtol=1e-6,
    )
    # Earlier observation changes the later policy prefix (but cannot alter
    # actor logits from before that observation due to the causal mask).
    changed = copy.deepcopy(batch)
    changed.input_ids[0, 4:7] = torch.tensor([23, 24, 25])
    other = score_original_rollout(student, changed, top_k=4)
    torch.testing.assert_close(
        scores.sampled_log_probs[0, :2], other.sampled_log_probs[0, :2]
    )
    assert not torch.allclose(
        scores.sampled_log_probs[0, 2:], other.sampled_log_probs[0, 2:]
    )


def test_actor_only_logit_gradients_and_frozen_teacher(tiny_models):
    student, teacher = tiny_models
    batch = compact_batch()
    student.config.model_type = "fallback_full_logits"
    kwargs, selection = rollout_forward_inputs(student, batch, torch.arange(2), 4)
    full = student(**kwargs).logits
    full.retain_grad()
    actor_logits = select_rollout_logits(full, selection)
    with torch.no_grad():
        q = teacher(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits
        q = q.gather(
            1, (batch.actor_positions - 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, 32)
        )
    reference = topk_reference_from_logits(
        actor_logits.detach(), q, batch.valid_mask, top_k=4
    )
    from b200_experiment.opd_core import (
        gather_candidate_log_probs,
        topk_candidate_ppo_loss,
    )

    current = gather_candidate_log_probs(
        actor_logits, reference.candidate_ids, temperature=1.0, chunk_steps=2
    )
    loss = topk_candidate_ppo_loss(
        current, reference, clip_low=0.2, clip_high=0.28, dual_clip=3.0
    )
    (loss * batch.valid_mask).sum().backward()
    gradient_rows = full.grad.abs().sum(-1) > 0
    expected = torch.zeros_like(gradient_rows)
    for row in range(2):
        expected[row, batch.actor_positions[row, batch.valid_mask[row]] - 1] = True
    assert torch.equal(gradient_rows, expected)
    assert all(p.grad is None for p in teacher.parameters())


def test_cmt_crosses_turns_resets_episodes_on_actor_clock():
    # The first two actor tokens are turn 1; last two are turn 2. The selector
    # deliberately receives no observation slots, regardless of their length.
    gain = torch.tensor([[1.0, 2.0, 4.0, 8.0], [100.0, 100.0, 0.0, 0.0]])
    log_probs = torch.log(torch.full((2, 4, 1), 0.5))
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    support = PGTOutput(
        gain,
        {
            "gain": gain,
            "gain_support_definition": "student_topk",
            "student_support_mass": torch.ones_like(gain),
            "teacher_support_mass": torch.ones_like(gain),
        },
        torch.zeros(2, 4, 1, dtype=torch.long),
        log_probs,
        log_probs,
        torch.ones(2, 4, 1, dtype=torch.bool),
    )
    output = CMTSelector().compute_scores(
        support, torch.zeros(2, 4, dtype=torch.long), valid
    )
    torch.testing.assert_close(
        output.diagnostics["H"][0], torch.tensor([11.0, 8.0, 4.0, 0.0])
    )
    torch.testing.assert_close(output.diagnostics["H"][1], torch.zeros(4))
    # A truncated single response could only see g_1-g_0=1 at the root.
    assert output.diagnostics["H"][0, 0] > gain[0, 1] - gain[0, 0]
    # With a nonzero root log-ratio, later turns change the actual allocation
    # score at the root, not merely a diagnostic. The other episode stays fixed.
    support.teacher_candidate_log_probs = log_probs.clone()
    support.teacher_candidate_log_probs[0, 0, 0] = torch.log(torch.tensor(0.4))
    before = CMTSelector().compute_scores(
        support, torch.zeros(2, 4, dtype=torch.long), valid
    )
    support.diagnostics["gain"] = gain.clone()
    support.diagnostics["gain"][0, 2:] *= 2
    after = CMTSelector().compute_scores(
        support, torch.zeros(2, 4, dtype=torch.long), valid
    )
    assert before.scores[0, 0] != after.scores[0, 0]
    torch.testing.assert_close(before.scores[1], after.scores[1])


def test_iw_prefix_crosses_turns_and_resets_per_episode():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 1.0, 0.0, 0.0]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    weights = compute_iw_opd_weights(q, torch.zeros_like(q), valid)
    torch.testing.assert_close(weights[0], torch.tensor([1.5, 1.45, 1.35, 1.2]))
    torch.testing.assert_close(weights[1], torch.tensor([1.5, 1.05, 1.0, 1.0]))
    assert weights[0, 2] < 1.5  # next assistant turn must not restart prefix


@pytest.mark.parametrize("method", ["cmt", "opd", "ta", "iw"])
def test_existing_optimizer_updates_real_multiturn_batch(tiny_models, method):
    from b200_experiment.trainer import _opd_train_step
    from b200_experiment.distributed import DistributedContext

    student, teacher = tiny_models
    teacher.requires_grad_(False)
    batch = compact_batch()
    kwargs, selection = rollout_forward_inputs(student, batch, torch.arange(2), 4)
    with torch.no_grad():
        p = select_rollout_logits(student(**kwargs).logits, selection)
        q = select_rollout_logits(teacher(**kwargs).logits, selection)
    if method == "iw":
        plp = p.log_softmax(-1).gather(-1, batch.response_ids.unsqueeze(-1)).squeeze(-1)
        qlp = q.log_softmax(-1).gather(-1, batch.response_ids.unsqueeze(-1)).squeeze(-1)
        reference, _ = build_iw_opd_reference(
            batch.response_ids, plp, qlp, batch.valid_mask
        )
    else:
        reference = topk_reference_from_logits(p, q, batch.valid_mask, top_k=4)
    config = load_config(REPO / "configs/alfworld_multiturn.yaml")
    config["experiment"]["method"] = method
    config["training"].update(micro_batch_size_per_gpu=1, ppo_mini_batch_size=2)
    weights = batch.valid_mask.float()
    # Exercise existing per-PPO constrained Gibbs branch for CMT as well.
    gibbs = (
        {
            "gibbs_scores": torch.arange(8).reshape(2, 4).float(),
            "gibbs_epsilon": 0.02,
            "gibbs_mode": "direct_bounded_gibbs",
        }
        if method == "cmt"
        else {}
    )
    before = student.lm_head.weight.detach().clone()
    result = _opd_train_step(
        student,
        torch.optim.AdamW(student.parameters(), lr=1e-3),
        batch,
        weights,
        reference,
        config,
        torch.device("cpu"),
        DistributedContext(0, 0, 1, torch.device("cpu")),
        **gibbs,
    )
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert not torch.equal(before, student.lm_head.weight)
    assert all(p.grad is None for p in teacher.parameters())


def test_contiguous_actor_scoring_matches_legacy_single_turn(tiny_models):
    student, _ = tiny_models
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    legacy = RolloutBatch(
        ids,
        torch.ones_like(ids),
        ids[:, 2:],
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3),
        2,
    )
    compact = copy.deepcopy(legacy)
    compact.actor_positions = torch.tensor([[2, 3, 4], [2, 3, 4]])
    a = score_original_rollout(student, legacy, top_k=4)
    b = score_original_rollout(student, compact, top_k=4)
    torch.testing.assert_close(a.sampled_log_probs, b.sampled_log_probs)
    torch.testing.assert_close(a.top_k_log_probs, b.top_k_log_probs)
    assert torch.equal(a.top_k_ids, b.top_k_ids)


def test_resume_rejects_changed_episode_protocol(alfconfig, tmp_path):
    from b200_experiment.resume import validate_resume_config, resolve_resume_checkpoint

    checkpoint = tmp_path / "checkpoint-000050"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "optimizer.pt").touch()
    save_config(alfconfig, tmp_path / "resolved_config.yaml")
    assert resolve_resume_checkpoint("auto", tmp_path) == checkpoint
    assert validate_resume_config(checkpoint, alfconfig)["checked"]
    changed = copy.deepcopy(alfconfig)
    changed["alfworld"]["max_turns"] = 20
    with pytest.raises(ValueError, match="alfworld.max_turns"):
        validate_resume_config(checkpoint, changed)


def test_native_qwen_turn_stop_for_base_tokenizer_only_in_multiturn():
    from b200_experiment.models import generation_stop_token_ids

    class BaseTokenizer(Tokenizer):
        def get_vocab(self):
            return {"<|endoftext|>": 1, "<|im_end|>": 2}

    tok = BaseTokenizer()
    assert generation_stop_token_ids(tok, {}) == [1]
    assert generation_stop_token_ids(tok, {"alfworld": {"enabled": True}}) == [1, 2]


def test_alfworld_tensorboard_metrics_and_cost_signature(alfconfig):
    from b200_experiment.tensorboard_logging import production_tensorboard_metrics
    from b200_experiment.training_cost import comparison_signature

    metrics = {
        "alfworld": {
            "success_rate": 0.5,
            "average_turns": 3.0,
            "episode_trace": "trace.jsonl",
        }
    }
    values = production_tensorboard_metrics(metrics, "opd")
    assert values["alfworld/success_rate"] == 0.5
    assert values["alfworld/average_turns"] == 3
    assert "alfworld/episode_trace" not in values
    changed = copy.deepcopy(alfconfig)
    changed["alfworld"]["max_turns"] = 15
    assert comparison_signature(alfconfig, 4) != comparison_signature(changed, 4)


@pytest.mark.skipif(
    not os.environ.get("ALFWORLD_REAL_DATA"),
    reason="optional real TextWorld smoke; set ALFWORLD_REAL_DATA to raw parquet directory",
)
def test_real_simulator_smoke(tmp_path):
    from b200_experiment.alfworld_cli import readiness

    root = Path(os.environ["ALFWORLD_REAL_DATA"])
    config = load_config(REPO / "configs/alfworld_multiturn.yaml")
    config["data"]["path"] = str(root / "train-00000-of-00001.parquet")
    config["alfworld"]["eval_paths"] = {
        s: str(root / f"{s}-00000-of-00001.parquet")
        for s in ("eval_in_distribution", "eval_out_of_distribution")
    }
    config["alfworld"]["game_cache_dir"] = str(tmp_path)
    result = readiness(config, smoke=True)
    assert result["oracle_simulator_smoke"]["success"]
