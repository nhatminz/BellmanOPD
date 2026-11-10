"""Student-only interactive episodes and compact actor-axis collation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .alfworld_env import EnvironmentClient
from .alfworld_protocol import (
    SYSTEM_PROMPT,
    LEGACY_SYSTEM_PROMPT,
    parse_action as parse_action,
    parse_action_details,
    format_observation,
    resolve_protocol,
)


class AppendOnlyChat:
    """Derive turn separators from the tokenizer without re-tokenizing actions.

    Re-rendering a whole chat can rewrite Qwen's thinking preamble or normalize
    generated text. Instead, append native chat envelopes as conditioning tokens,
    preserving every sampled ID and its exact original prefix.
    """

    def __init__(self, tokenizer, data_config, protocol_version=2):
        self.tokenizer = tokenizer
        self.kwargs = dict(data_config.get("chat_template_kwargs", {}))
        self.protocol_version = protocol_version

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def initial(self, observation):
        messages = [
            {
                "role": "system",
                "content": LEGACY_SYSTEM_PROMPT
                if self.protocol_version == 1
                else SYSTEM_PROMPT,
            },
            {"role": "user", "content": observation},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.kwargs
            )
        except Exception as error:
            if "System role not supported" not in str(error):
                raise
            # Gemma2 has no system role. Keep the identical instruction as
            # part of the first user message using its own native template.
            rendered = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": messages[0]["content"] + "\n\n" + observation,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                **self.kwargs,
            )
        return self.encode(rendered)

    def continuation(self, observation, last_token, stop_ids):
        marker = "ALFWORLD_BOUNDARY_9bf6024c"
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Previous observation"},
                {"role": "assistant", "content": marker},
                {"role": "user", "content": observation},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **self.kwargs,
        )
        if rendered.count(marker) != 1:
            raise ValueError(
                "ALFWorld requires a chat template with an identifiable append-only assistant boundary"
            )
        suffix = self.encode(rendered.split(marker, 1)[1])
        if suffix and last_token in stop_ids and suffix[0] == last_token:
            suffix = suffix[1:]
        return suffix


def collect_episode(
    task,
    tokenizer,
    config,
    generate,
    seed,
    *,
    env_factory=EnvironmentClient,
    environment_seed=None,
):
    config = resolve_protocol(config)
    settings = config["alfworld"]
    version = settings["protocol_version"]
    chat = AppendOnlyChat(tokenizer, config["data"], version)
    from .models import generation_stop_token_ids

    stops = generation_stop_token_ids(tokenizer, config)
    env = env_factory(task, settings)
    ids, positions, turns, log_probs = [], [], [], []
    reason, success, invalid = "max_turns", False, 0
    previous_action = None
    reset_seed = int(seed if environment_seed is None else environment_seed)
    trace = {
        "task_id": task["id"],
        "task_type": task["task_type"],
        "split": task["split"],
        "seed": int(seed),
        "environment_seed": reset_seed,
        "game_fingerprint": task["game_fingerprint"],
        "turns": turns,
        "protocol_version": version,
        "protocol_fingerprint": settings["protocol_fingerprint"],
        "action_prompt_mode": settings["action_prompt_mode"],
        "action_projection": False,
        "invalid_token_policy": "retain_all_sampled_actor_tokens",
    }
    try:
        try:
            state = env.reset(reset_seed)
        except TimeoutError:
            state = None
            reason = "environment_reset_timeout"
        if state is not None:
            initial_observation = state["observation"]
            # Goal text is observable at reset; never read hidden PDDL goals or plans.
            task_goal = initial_observation.split("Your task is to:", 1)[-1].strip()
            trace.update(initial_observation=initial_observation, task_goal=task_goal)
            initial_prompt = format_observation(state, task_goal, settings)
            trace["initial_policy_observation"] = initial_prompt
            ids = chat.initial(initial_prompt)
            total_cap = int(settings["max_episode_tokens"])
            actor_cap = int(config["rollout"]["max_new_tokens"])
            if len(ids) >= total_cap:
                raise ValueError(
                    f"Initial ALFWorld task {task['id']} including action menu exceeds the context budget; increase max_episode_tokens/max_model_len. No task/action list was truncated."
                )
            for turn in range(int(settings.get("max_turns", 30))):
                budget = min(
                    int(settings["max_turn_tokens"]),
                    actor_cap - len(positions),
                    total_cap - len(ids),
                )
                if budget <= 0:
                    reason = "episode_token_budget"
                    break
                prompt_length = len(ids)
                try:
                    generated = generate(
                        list(ids), (int(seed) + turn * 1000003) % (2**31 - 1), budget
                    )
                except TimeoutError:
                    reason = "generation_timeout"
                    break
                if isinstance(generated, dict):
                    tokens = list(generated["token_ids"])
                    lp = generated.get("log_probs")
                    if lp is None or len(lp) != len(tokens):
                        raise RuntimeError(
                            "Requested multi-turn log-probs must align with every sampled token"
                        )
                    trace["rollout_logprob_kind"] = generated["logprob_kind"]
                else:
                    tokens, lp = list(generated), None
                if len(tokens) > budget:
                    raise RuntimeError(
                        "Generator exceeded the requested per-turn budget"
                    )
                if not tokens:
                    reason = "empty_generation"
                    break
                begin = len(positions)
                positions.extend(range(len(ids), len(ids) + len(tokens)))
                ids.extend(tokens)  # sampled IDs stay exact, even if parsing fails
                if lp is not None:
                    log_probs.extend(lp)
                response = tokenizer.decode(tokens, skip_special_tokens=True)
                parsed = parse_action_details(response, version)
                action = parsed["action"]
                admissible = list(state.get("admissible_actions", []))
                simulator_invalid = action is not None and action not in admissible
                bad = parsed["parser_failure"] or simulator_invalid
                invalid += int(bad)
                item = {
                    "turn": turn,
                    "prompt_length": prompt_length,
                    "actor_start": begin,
                    "actor_end": len(positions),
                    "input_start": prompt_length,
                    "input_end": len(ids),
                    "response": response,
                    "response_token_ids": tokens,
                    **parsed,
                    "invalid": bool(bad),
                    "simulator_invalid": bool(simulator_invalid),
                    "admissible_actions_before": admissible,
                    "observation_before": state["observation"],
                    "repeated_action": action is not None and action == previous_action,
                    "unchanged_observation": False,
                }
                if lp is not None:
                    item.update(
                        response_log_probs=list(lp),
                        logprob_kind=generated["logprob_kind"],
                    )
                if not response.strip() and tokens[-1] in stops:
                    item.update(
                        observation=state["observation"],
                        reward=0.0,
                        terminated=True,
                        success=False,
                        feedback_source="agent_eos",
                    )
                    turns.append(item)
                    reason = "agent_eos"
                    break
                parser_error = None
                if action is None:
                    # Keep the actual last simulator state/observation separate.
                    parser_error = "Output exactly one ALFWorld command on one line. Environment state is unchanged."
                    item.update(
                        feedback_source="action_parser",
                        parser_feedback=parser_error,
                        observation=state["observation"],
                        reward=0.0,
                        terminated=False,
                        success=False,
                    )
                    if version == 1:
                        # Explicit old-protocol reproduction, never used in v2.
                        state = {
                            **state,
                            "observation": "Protocol error: output exactly one ALFWorld command on one line. Environment state is unchanged.",
                            "reward": 0.0,
                        }
                        item["observation"] = state["observation"]
                else:
                    previous_observation = state["observation"]
                    try:
                        state = env.step(action)
                    except TimeoutError:
                        item.update(
                            observation=previous_observation,
                            reward=0.0,
                            terminated=True,
                            success=False,
                            feedback_source="timeout",
                        )
                        turns.append(item)
                        reason = "environment_timeout"
                        break
                    item.update(
                        feedback_source="simulator",
                        **{
                            key: state[key]
                            for key in (
                                "observation",
                                "reward",
                                "terminated",
                                "success",
                            )
                        },
                    )
                    item["unchanged_observation"] = (
                        state["observation"] == previous_observation
                    )
                    previous_action = action
                turns.append(item)
                success = bool(state["success"])
                if state["terminated"]:
                    reason = "success" if success else "environment_terminal"
                    break
                if turn + 1 == int(settings.get("max_turns", 30)):
                    break
                policy_observation = format_observation(
                    state,
                    task_goal,
                    settings,
                    parser_error=(
                        state["observation"]
                        if version == 1 and parser_error
                        else parser_error
                    ),
                    repeated_warning=item["repeated_action"]
                    and item["unchanged_observation"],
                )
                item["next_policy_observation"] = policy_observation
                suffix = chat.continuation(policy_observation, tokens[-1], stops)
                if len(ids) + len(suffix) >= total_cap:
                    reason = "context_budget"
                    break
                ids.extend(suffix)
    finally:
        env.close()
    trace.update(
        success=success,
        termination_reason=reason,
        average_turns=len(turns),
        invalid_actions=invalid,
        invalid_action_rate=invalid / max(len(turns), 1),
        actor_tokens=len(positions),
        input_tokens=len(ids),
        episode_boundary=[0, len(ids)],
    )
    if settings.get("trace_token_ids", True):
        trace.update(input_ids=ids, actor_positions=positions)
    result = {"input_ids": ids, "actor_positions": positions, "trace": trace}
    if log_probs:
        result["log_probs"] = log_probs
    return result


def collect_episodes(
    tasks, tokenizer, config, generate, seeds, *, env_factory=EnvironmentClient
):
    workers = min(len(tasks), int(config["alfworld"].get("env_workers", 4)))

    def one(pair):
        task, seed = pair
        return collect_episode(
            task, tokenizer, config, generate, seed, env_factory=env_factory
        )

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        return list(executor.map(one, zip(tasks, seeds)))


def collate_episodes(episodes, pad_token_id, device):
    import torch
    from .scoring import RolloutBatch

    if any(not episode["actor_positions"] for episode in episodes):
        raise RuntimeError(
            "ALFWorld produced an episode with no actor tokens (e.g. empty generation). Cannot invent loss tokens; inspect generation/context limits and retry."
        )
    length = max(len(e["input_ids"]) for e in episodes)
    width = max(len(e["actor_positions"]) for e in episodes)
    ids = torch.full(
        (len(episodes), length), int(pad_token_id), dtype=torch.long, device=device
    )
    mask = torch.zeros_like(ids)
    actor_mask = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.zeros((len(episodes), width), dtype=torch.long, device=device)
    valid = torch.zeros_like(positions, dtype=torch.bool)
    responses = torch.full_like(positions, int(pad_token_id))
    log_probs = torch.full_like(responses, float("nan"), dtype=torch.float32)
    for index, episode in enumerate(episodes):
        n, a = len(episode["input_ids"]), len(episode["actor_positions"])
        actor_positions = episode["actor_positions"]
        if not 0 < actor_positions[0] <= actor_positions[-1] < n or any(
            right <= left for left, right in zip(actor_positions, actor_positions[1:])
        ):
            raise ValueError(
                "Actor positions must be strictly increasing, in bounds, and have preceding context"
            )
        ids[index, :n] = torch.tensor(episode["input_ids"], device=device)
        mask[index, :n] = 1
        positions[index, :a] = torch.tensor(episode["actor_positions"], device=device)
        valid[index, :a] = True
        actor_mask[index, positions[index, :a]] = True
        responses[index, :a] = ids[index, positions[index, :a]]
        if "log_probs" in episode:
            if len(episode["log_probs"]) != a:
                raise ValueError("Episode sampled token/log-prob lengths differ")
            log_probs[index, :a] = torch.tensor(
                episode["log_probs"], dtype=torch.float32, device=device
            )
    kinds = {
        e["trace"].get("rollout_logprob_kind") for e in episodes if e.get("log_probs")
    }
    if len(kinds) > 1:
        raise ValueError("Cannot mix raw-policy and sampling-distribution log-probs")
    return RolloutBatch(
        ids,
        mask,
        responses,
        valid,
        log_probs,
        1,
        positions,
        actor_mask,
        [episode["trace"] for episode in episodes],
        next(iter(kinds), None),
    )


def generate_training_episodes(
    model, engine, tasks, tokenizer, config, device, seeds, distributed=None
):
    from .scoring import generate_on_policy
    from .models import generation_stop_token_ids

    if engine is not None:
        with engine.multiturn_session(model) as generate:
            episodes = collect_episodes(tasks, tokenizer, config, generate, seeds)
    else:
        import torch

        # HF fallback is a serial, single-process debug path. Never enter
        # per-turn FSDP collectives with different episode lengths on ranks.
        debug = {**config, "alfworld": {**config["alfworld"], "env_workers": 1}}

        def generate(ids, seed, budget):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            rollout = generate_on_policy(
                model,
                input_ids,
                torch.ones_like(input_ids),
                max_new_tokens=budget,
                temperature=float(config["rollout"]["temperature"]),
                top_p=float(config["rollout"]["top_p"]),
                eos_token_ids=generation_stop_token_ids(tokenizer, config),
                pad_token_id=tokenizer.pad_token_id,
                seed=seed,
            )
            return rollout.response_ids[0][rollout.valid_mask[0]].tolist()

        episodes = collect_episodes(tasks, tokenizer, debug, generate, seeds)
    empty = sum(not e["actor_positions"] for e in episodes)
    global_empty = distributed.sum_int(empty) if distributed is not None else empty
    if global_empty:
        import json
        import uuid
        from pathlib import Path

        if config.get("experiment", {}).get("output_dir"):
            directory = Path(config["experiment"]["output_dir"]) / "alfworld_failures"
            directory.mkdir(parents=True, exist_ok=True)
            path = (
                directory
                / f"rank-{distributed.rank if distributed else 0}-{uuid.uuid4().hex}.jsonl"
            )
            path.write_text(
                "".join(json.dumps(e["trace"]) + "\n" for e in episodes),
                encoding="utf-8",
            )
        raise RuntimeError(
            f"ALFWorld rollout has {global_empty} episodes with zero actor tokens globally. All ranks stop before scoring/backward; no fabricated targets or optimizer update."
        )
    return collate_episodes(episodes, tokenizer.pad_token_id, device)
