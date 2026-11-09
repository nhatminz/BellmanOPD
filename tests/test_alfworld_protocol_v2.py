import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from b200_experiment.alfworld_diagnostics import episode_metrics, check_collapse
from b200_experiment.alfworld_protocol import (
    parse_action,
    parse_action_details,
    format_observation,
    resolve_protocol,
    SYSTEM_PROMPT,
)
from b200_experiment.alfworld_rollout import (
    collect_episode,
    collate_episodes,
    AppendOnlyChat,
    generate_training_episodes,
)
from b200_experiment.alfworld_env import load_task_splits, NativeEnvironment
from b200_experiment.config import save_config
from b200_experiment.distributed import DistributedContext
from test_alfworld_multiturn import (
    make_alfconfig,
    InteractiveEnv,
    Tokenizer,
    compact_batch,
)


@pytest.fixture
def config(tmp_path):
    cfg = make_alfconfig(tmp_path)
    cfg["alfworld"]["max_episode_tokens"] = 8192
    return resolve_protocol(cfg)


@pytest.mark.parametrize(
    "response,expected",
    [
        ("look;", "look"),
        ("look", "look"),
        ("look.", "look"),
        ("inventory;", "inventory"),
        ("go to desk 1;", "go to desk 1"),
        ("open drawer 1;", "open drawer 1"),
        ("take cd 1 from drawer 1;", "take cd 1 from drawer 1"),
        ("slice apple 1 with knife 1;", "slice apple 1 with knife 1"),
        ("put apple 1 into bowl 1", "put apple 1 into bowl 1"),
        ("move cd 1 to desk 1.", "move cd 1 to desk 1"),
        ('{"action":"look;"}', "look"),
        ("<action>inventory.</action>", "inventory"),
    ],
)
def test_exact_single_command_optional_punctuation(response, expected):
    assert parse_action(response) == expected
    details = parse_action_details(response)
    assert not details["parser_failure"]


@pytest.mark.parametrize(
    "response",
    [
        "look again",
        "look; open drawer 1",
        "look; look",
        "look;;",
        "look..",
        "go to desk 1; open drawer 1",
        "look\ninventory",
        "go to desk 1 and open drawer 1",
        "slice apple 1 with knife 1; look",
        "slice apple 1",
        "open drawer 1 and look",
        '{"action":"look","action":"inventory"}',
        '{"action":"look\\nlook"}',
        "<action>look</action><action>look</action>",
        "I will go to desk 1",
    ],
)
def test_never_extract_or_execute_first_of_multiple_commands(response):
    assert parse_action(response) is None


def run(config, responses, env=InteractiveEnv):
    task = load_task_splits(config)["train"][0]
    tokenizer = Tokenizer()
    actions = iter(responses)
    prefixes = []

    def generate(ids, seed, budget):
        prefixes.append(list(ids))
        return tokenizer.encode(next(actions)) + [1]

    return collect_episode(
        task, tokenizer, config, generate, 42, env_factory=env
    ), prefixes


def test_punctuation_recovery_preserves_sampled_ids_and_executes_normalized_action(
    config,
):
    episode, prefixes = run(config, ["open cabinet 1;", "take cup 1 from cabinet 1."])
    assert episode["trace"]["success"]
    assert InteractiveEnv.instances[-1].actions == [
        "open cabinet 1",
        "take cup 1 from cabinet 1",
    ]
    batch = collate_episodes([episode], 0, "cpu")
    expected = (
        Tokenizer().encode("open cabinet 1;")
        + [1]
        + Tokenizer().encode("take cup 1 from cabinet 1.")
        + [1]
    )
    assert batch.response_ids[0].tolist() == expected
    assert batch.actor_token_mask.sum() == len(expected)
    metrics = episode_metrics([episode["trace"]])
    assert metrics["punctuation_recovery_rate"] == 1
    assert metrics["parser_failure_rate"] == 0
    assert metrics["task_success_rate"] == 1
    assert "Currently admissible commands" in Tokenizer().decode(prefixes[0])
    assert "take cup 1 from cabinet 1" in Tokenizer().decode(prefixes[1])


def test_parser_errors_keep_last_real_observation_goal_and_game_state(config):
    episode, prefixes = run(
        config,
        [
            "look; open cabinet 1",
            "look again",
            "open cabinet 1",
            "take cup 1 from cabinet 1",
        ],
    )
    env = InteractiveEnv.instances[-1]
    assert env.actions == ["open cabinet 1", "take cup 1 from cabinet 1"]
    trace = episode["trace"]
    assert trace["success"]
    assert trace["turns"][0]["observation"] == trace["initial_observation"]
    assert trace["turns"][1]["observation"] == trace["initial_observation"]
    for prefix in prefixes[1:3]:
        text = Tokenizer().decode(prefix)
        assert "You see cabinet 1" in text and "take cup" in text
        assert "Action parser feedback (no environment step)" in text
    metrics = episode_metrics([trace])
    assert metrics["parser_failure_rate"] == 0.5
    assert metrics["simulator_invalid_action_rate"] == 0
    assert metrics["simulator_steps"] == 2


def test_simulator_invalid_command_is_separate_from_parser_failure(config):
    config["alfworld"]["max_turns"] = 2
    episode, _ = run(config, ["open drawer 99", "take cup 1 from cabinet 1"])
    trace = episode["trace"]
    assert trace["turns"][0]["simulator_invalid"]
    assert not trace["turns"][0]["parser_failure"]
    assert InteractiveEnv.instances[-1].actions[0] == "open drawer 99"
    assert episode_metrics([trace])["simulator_invalid_action_rate"] >= 0.5


def test_all_invalid_tokens_are_retained_and_collapse_guard_is_explicit(config):
    config["alfworld"]["max_turns"] = 2
    episode, _ = run(config, ["look again", "look; look"])
    batch = collate_episodes([episode], 0, "cpu")
    assert batch.valid_mask.all()
    metrics = episode_metrics([episode["trace"]])
    assert metrics["invalid_actor_token_count"] == metrics["actor_token_count"] > 0
    assert metrics["parser_failure_rate"] == 1
    assert not InteractiveEnv.instances[-1].actions
    with pytest.raises(RuntimeError, match="Stopped before the optimizer update"):
        check_collapse(metrics, config["alfworld"])
    settings = copy.deepcopy(config["alfworld"])
    settings["collapse_guard"]["mode"] = "warn"
    with pytest.warns(RuntimeWarning, match="all actor responses"):
        check_collapse(metrics, settings)
    settings["collapse_guard"]["mode"] = "off"
    assert check_collapse(metrics, settings) == []


def test_repeated_look_collapse_is_observable_without_fabricated_success(config):
    class Static(InteractiveEnv):
        def step(self, action):
            self.actions.append(action)
            return self.reset(self.seed)

    config["alfworld"]["max_turns"] = 3
    config["alfworld"]["collapse_guard"]["min_turns"] = 3
    item, _ = run(config, ["look;", "look", "look."], Static)
    metrics = episode_metrics([item["trace"]])
    assert metrics["parser_failure_rate"] == 0
    assert metrics["look_action_rate"] == metrics["unchanged_observation_rate"] == 1
    assert metrics["repeated_action_rate"] == pytest.approx(2 / 3)
    assert metrics["unique_action_count"] == 1 and not item["trace"]["success"]
    with pytest.raises(RuntimeError, match="look_action_rate"):
        check_collapse(metrics, config["alfworld"])


def test_legacy_observation_mode_and_v1_are_explicit(config):
    state = InteractiveEnv({}, {}).reset(42)
    settings = {**config["alfworld"], "action_prompt_mode": "legacy"}
    assert "Currently admissible" not in format_observation(state, "task", settings)
    assert "Currently admissible" in format_observation(
        state, "task", config["alfworld"]
    )
    assert parse_action("look;", protocol_version=1) is None
    assert (
        "look;"
        in __import__(
            "b200_experiment.alfworld_protocol", fromlist=["LEGACY_SYSTEM_PROMPT"]
        ).LEGACY_SYSTEM_PROMPT
    )
    assert "look;" not in SYSTEM_PROMPT


def test_protocol_fingerprint_rejects_incompatible_resume(config, tmp_path):
    from b200_experiment.resume import validate_resume_config

    checkpoint = tmp_path / "checkpoint-000050"
    checkpoint.mkdir()
    legacy = copy.deepcopy(config)
    legacy["alfworld"].update(protocol_version=1, action_prompt_mode="legacy")
    legacy = resolve_protocol(legacy)
    save_config(legacy, tmp_path / "resolved_config.yaml")
    with pytest.raises(ValueError, match="protocol"):
        validate_resume_config(checkpoint, config)
    save_config(config, tmp_path / "resolved_config.yaml")
    changed = copy.deepcopy(config)
    changed["alfworld"]["repeated_action_warning"] = True
    with pytest.raises(ValueError, match="protocol_fingerprint"):
        validate_resume_config(checkpoint, changed)
    assert validate_resume_config(checkpoint, config)["checked"]


def test_context_budget_never_silently_truncates_an_action_list(config):
    config["alfworld"]["max_episode_tokens"] = 10
    with pytest.raises(ValueError, match="No task/action list was truncated"):
        run(config, ["look"])


def test_gemma_style_chat_no_system_role_preserves_native_boundaries():
    class Gemma(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if any(m["role"] == "system" for m in messages):
                raise ValueError("System role not supported")
            text = "<bos>" + "".join(
                f"<start_of_turn>{'model' if m['role'] == 'assistant' else 'user'}\n{m['content']}<end_of_turn>\n"
                for m in messages
            )
            return text + "<start_of_turn>model\n"

    tokenizer = Gemma()
    chat = AppendOnlyChat(tokenizer, {}, 2)
    initial = tokenizer.decode(chat.initial("observation"))
    assert initial.count("<bos>") == 1 and SYSTEM_PROMPT in initial
    suffix = tokenizer.decode(chat.continuation("new observation", 1, [1]))
    assert "<bos>" not in suffix
    assert suffix.startswith("<end_of_turn>\n<start_of_turn>user")


def test_native_success_comes_from_won_not_reward():
    env = NativeEnvironment()
    env.native_site = None
    env.state = SimpleNamespace(
        feedback="won", won=True, lost=False, admissible_commands=[]
    )
    assert env._observation(0, False)["success"]
    env.state.won = False
    assert not env._observation(100, False)["success"]


def test_zero_actor_batch_stops_all_ranks_without_fabricating_targets(config):
    from contextlib import contextmanager

    class Engine:
        @contextmanager
        def multiturn_session(self, model):
            yield lambda *args: []

    # Simulate an empty actor episode on ANOTHER rank while this rank has data.
    class Context:
        rank = 0

        def sum_int(self, count):
            return 1

    task = load_task_splits(config)["train"][0]
    config["experiment"]["output_dir"] = str(
        Path(config["data"]["path"]).parent / "failure-output"
    )
    from unittest.mock import patch

    with patch(
        "b200_experiment.alfworld_rollout.collect_episodes",
        return_value=[{"input_ids": [2, 3], "actor_positions": [1], "trace": {}}],
    ):
        with pytest.raises(
            RuntimeError, match="All ranks stop before scoring/backward"
        ):
            generate_training_episodes(
                None, Engine(), [task], Tokenizer(), config, "cpu", [42], Context()
            )


def test_sampling_logprob_comparison_respects_temperature_and_top_p():
    from b200_experiment.alfworld_logprob_debug import sampled_distribution_log_probs

    logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    raw, sampled = sampled_distribution_log_probs(logits, torch.tensor([[0]]), 2.0, 0.8)
    assert raw.item() == pytest.approx(torch.log_softmax(logits, -1)[0, 0, 0].item())
    assert sampled.item() == pytest.approx(
        torch.log_softmax(torch.tensor([1.0, 0.5, 0.0]), -1)[0].item()
    )
    assert raw.item() != sampled.item()


def test_debug_compares_later_turns_and_fails_on_mismatch(config):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from b200_experiment.scoring import rollout_forward_inputs, select_rollout_logits
    from b200_experiment.alfworld_logprob_debug import (
        compare_multiturn_logprobs,
        sampled_distribution_log_probs,
    )

    torch.manual_seed(2)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    model.eval()
    batch = compact_batch()
    batch.episode_traces = [
        {
            "turns": [
                {"turn": 0, "actor_start": 0, "actor_end": 2},
                {"turn": 1, "actor_start": 2, "actor_end": 4},
            ]
        },
        {
            "turns": [
                {"turn": 0, "actor_start": 0, "actor_end": 1},
                {"turn": 1, "actor_start": 1, "actor_end": 2},
            ]
        },
    ]
    batch.rollout_logprob_kind = "sampling_temperature_top_p"
    config["rollout"].update(temperature=2.0, top_p=1.0)
    config["rollout"]["vllm"]["logprob_sanity"] = {
        "enabled": True,
        "tolerance": 1e-5,
        "fail_on_mismatch": True,
    }
    with torch.no_grad():
        kwargs, selection = rollout_forward_inputs(model, batch, torch.arange(2), 4)
        logits = select_rollout_logits(model(**kwargs).logits, selection)
        _, batch.rollout_log_probs = sampled_distribution_log_probs(
            logits, batch.response_ids, 2.0, 1.0
        )
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    result = compare_multiturn_logprobs(model, batch, config, context)
    assert result["later_turn_tokens"] == 2 and result["passed"]
    assert result["hf_raw_policy_logprob_mean"] != result["hf_sampling_logprob_mean"]
    batch.rollout_log_probs[0, 2] += 0.2
    with pytest.raises(RuntimeError, match="sampling log-prob mismatch"):
        compare_multiturn_logprobs(model, batch, config, context)


@pytest.mark.parametrize("method", ["opd", "ta", "cmt", "iw"])
def test_explicit_off_guard_keeps_invalid_on_policy_loss_finite(config, method):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from b200_experiment.scoring import score_student_teacher_rollout
    from b200_experiment.opd_core import (
        build_student_topk_opd_reference,
        build_iw_opd_reference,
    )
    from b200_experiment.selectors import TASelector
    from b200_experiment.selectors.pgt_selector import PGTSelector
    from b200_experiment.selectors.cmt_selector import CMTSelector
    from b200_experiment.selectors.budget import top_budget_mask
    from b200_experiment.trainer import _opd_train_step

    config["alfworld"]["max_turns"] = 2
    config["alfworld"]["collapse_guard"]["mode"] = "off"
    episode, _ = run(config, ["look again", "look; look"])
    batch = collate_episodes([episode], 0, "cpu")
    assert episode_metrics([episode["trace"]])["parser_failure_rate"] == 1
    check_collapse(episode_metrics([episode["trace"]]), config["alfworld"])
    torch.manual_seed(4)
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    student, teacher = Qwen3ForCausalLM(cfg), Qwen3ForCausalLM(cfg)
    teacher.requires_grad_(False)
    p, q = score_student_teacher_rollout(
        student, teacher, batch, top_k=4, micro_batch_size=1
    )
    valid = batch.valid_mask
    weights = valid.float()
    gibbs = {}
    if method == "iw":
        reference, _ = build_iw_opd_reference(
            batch.response_ids, p.sampled_log_probs, q.sampled_log_probs, valid
        )
    else:
        reference = build_student_topk_opd_reference(
            p.top_k_ids,
            p.top_k_log_probs,
            q.candidate_log_probs,
            valid,
            top_k=4,
            full_vocab_ppo=True,
        )
    if method == "ta":
        scores = TASelector(top_k=4).compute_scores_from_topk(
            p.top_k_ids,
            q.top_k_ids,
            p.top_k_log_probs,
            q.candidate_log_probs,
            q.top_k_log_probs,
            p.candidate_log_probs,
            valid,
        )
        weights = top_budget_mask(scores.scores, valid, rho=0.1).float()
    if method == "cmt":
        support = PGTSelector().compute_scores_from_topk(
            p.top_k_ids,
            q.top_k_ids,
            p.top_k_log_probs,
            q.candidate_log_probs,
            q.top_k_log_probs,
            p.candidate_log_probs,
            valid,
            gain_support="student_topk",
        )
        scores = CMTSelector().compute_scores(support, batch.response_ids, valid)
        gibbs = {
            "gibbs_scores": scores.scores,
            "gibbs_epsilon": 0.02,
            "gibbs_mode": "direct_bounded_gibbs",
        }
    config["experiment"]["method"] = method
    config["training"].update(ppo_mini_batch_size=1, micro_batch_size_per_gpu=1)
    before = student.lm_head.weight.detach().clone()
    result = _opd_train_step(
        student,
        torch.optim.AdamW(student.parameters(), lr=0.001),
        batch,
        weights,
        reference,
        config,
        torch.device("cpu"),
        DistributedContext(0, 0, 1, torch.device("cpu")),
        **gibbs,
    )
    assert torch.isfinite(torch.tensor(result["loss"])) and result["gradient_norm"] > 0
    assert not torch.equal(before, student.lm_head.weight)
    assert all(param.grad is None for param in teacher.parameters())


def test_debug_returns_processed_logprobs_only_when_enabled(
    monkeypatch, config, tmp_path
):
    from b200_experiment.vllm_rollout import VLLMRolloutEngine
    import b200_experiment.models as models

    engine = VLLMRolloutEngine(config, tmp_path)
    monkeypatch.setattr(
        "b200_experiment.vllm_rollout.shutil.which", lambda _: "/venv/bin/vllm"
    )
    assert "--logprobs-mode" not in engine._server_command()
    config["rollout"]["vllm"]["logprob_sanity"]["enabled"] = True
    engine.settings["logprob_sanity"]["enabled"] = True
    command = engine._server_command()
    assert command[command.index("--logprobs-mode") + 1] == "processed_logprobs"
    engine.process = SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(models, "load_student_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(engine, "_release_torch_cache_if_needed", lambda: (False, 0.0))
    for name in ("_wake_weights", "_wake_kv_cache", "_sleep"):
        monkeypatch.setattr(engine, name, lambda: None)
    monkeypatch.setattr(engine, "_sync_weights", lambda _: {})
    calls = []

    def one(*args, **kwargs):
        calls.append(kwargs)
        return 0, [10, 1], [-0.1, -0.2]

    monkeypatch.setattr(engine, "_generate_one", one)
    with engine.multiturn_session(None) as generate:
        result = generate([2, 3], 42, 2)
    assert calls[0]["return_log_probs"]
    assert result["logprob_kind"] == "sampling_temperature_top_p"
    assert result["token_ids"] == [10, 1] and result["log_probs"] == [-0.1, -0.2]
    engine.process = None


def test_alfworld_server_reserves_full_episode_context_not_legacy_prompt_plus_actor(
    monkeypatch, config, tmp_path
):
    from b200_experiment.vllm_rollout import VLLMRolloutEngine

    config["alfworld"]["max_episode_tokens"] = 4096
    config["rollout"]["vllm"]["max_model_len"] = 4096
    monkeypatch.setattr(
        "b200_experiment.vllm_rollout.shutil.which", lambda _: "/venv/bin/vllm"
    )
    engine = VLLMRolloutEngine(config, tmp_path)
    command = engine._server_command()
    assert command[command.index("--max-model-len") + 1] == "4096"
