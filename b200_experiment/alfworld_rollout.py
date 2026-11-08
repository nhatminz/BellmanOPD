"""Student-only interactive episodes and compact actor-axis collation."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from .alfworld_env import EnvironmentClient


SYSTEM_PROMPT = (
    "You are an agent in an interactive household environment. Read the task and "
    "observations, then output exactly one ALFWorld command per turn, on one line. "
    "Examples of command formats: go to receptacle 1; open receptacle 1; "
    "take object 1 from receptacle 1; move object 1 to receptacle 1; "
    "heat object 1 with microwave 1; clean object 1 with sinkbasin 1; "
    "cool object 1 with fridge 1; use desklamp 1; look; inventory. "
    "Do not invent observations or output multiple commands."
)
_COMMAND = re.compile(
    r"(?:look|inventory|help|(?:go to|open|close|take|move|put|heat|clean|cool|use|examine) [^\n\r;]+)",
    re.IGNORECASE,
)


def parse_action(response):
    text = response.strip()
    if text.startswith("{"):
        try:
            value = json.loads(text)
            if (
                not isinstance(value, dict)
                or set(value) != {"action"}
                or not isinstance(value["action"], str)
            ):
                return None
            text = value["action"].strip()
        except (ValueError, TypeError):
            return None
    elif text.startswith("<action>"):
        match = re.fullmatch(r"<action>([^<>]+)</action>", text, re.DOTALL)
        if not match:
            return None
        text = match.group(1).strip()
    return text if _COMMAND.fullmatch(text) else None


class AppendOnlyChat:
    """Derive turn separators from the tokenizer without re-tokenizing actions.

    Re-rendering a whole chat can rewrite Qwen's thinking preamble or normalize
    generated text. Instead, append native chat envelopes as conditioning tokens,
    preserving every sampled ID and its exact original prefix.
    """

    def __init__(self, tokenizer, data_config):
        self.tokenizer = tokenizer
        self.kwargs = dict(data_config.get("chat_template_kwargs", {}))

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def initial(self, observation):
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": observation},
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
    task, tokenizer, config, generate, seed, *, env_factory=EnvironmentClient
):
    settings = config["alfworld"]
    chat = AppendOnlyChat(tokenizer, config["data"])
    from .models import generation_stop_token_ids

    stops = generation_stop_token_ids(tokenizer, config)
    env = env_factory(task, settings)
    ids, positions, turns = [], [], []
    reason, success, invalid = "max_turns", False, 0
    trace = {
        "task_id": task["id"],
        "task_type": task["task_type"],
        "split": task["split"],
        "seed": int(seed),
        "game_fingerprint": task["game_fingerprint"],
        "turns": turns,
    }
    try:
        try:
            state = env.reset(seed)
        except TimeoutError:
            trace.update(
                initial_observation="",
                success=False,
                termination_reason="environment_reset_timeout",
                average_turns=0,
                invalid_actions=0,
                invalid_action_rate=0.0,
                actor_tokens=0,
                input_tokens=0,
                episode_boundary=[0, 0],
            )
            return {"input_ids": [], "actor_positions": [], "trace": trace}
        trace["initial_observation"] = state["observation"]
        ids = chat.initial(state["observation"])
        total_cap = int(settings["max_episode_tokens"])
        actor_cap = int(config["rollout"]["max_new_tokens"])
        if len(ids) >= total_cap:
            raise ValueError(
                f"Initial ALFWorld task {task['id']} exceeds the episode/context budget; increase max_episode_tokens and vLLM max_model_len together"
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
            # The generator is always the current STUDENT, never the teacher.
            try:
                tokens = list(
                    generate(
                        list(ids), (int(seed) + turn * 1000003) % (2**31 - 1), budget
                    )
                )
            except TimeoutError:
                reason = "generation_timeout"
                break
            if len(tokens) > budget:
                raise RuntimeError("Generator exceeded the requested per-turn budget")
            if not tokens:
                reason = "empty_generation"
                break
            begin = len(positions)
            positions.extend(range(len(ids), len(ids) + len(tokens)))
            ids.extend(tokens)
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            action = parse_action(response)
            admissible = state.get("admissible_actions", [])
            bad = action is None or action not in admissible
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
                "action": action,
                "invalid": bad,
            }
            if not response.strip() and tokens[-1] in stops:
                item.update(observation="", reward=0.0, terminated=True, success=False)
                turns.append(item)
                reason = "agent_eos"
                break
            if action is None:
                # Protocol feedback is explicitly distinguished from simulator
                # observations; no action or invented game transition is applied.
                state = {
                    **state,
                    "observation": "Protocol error: output exactly one ALFWorld command on one line. Environment state is unchanged.",
                    "reward": 0.0,
                }
                item["feedback_source"] = "action_parser"
            else:
                try:
                    state = env.step(action)
                except TimeoutError:
                    item.update(
                        observation="Environment step timed out.",
                        reward=0.0,
                        terminated=True,
                        success=False,
                        feedback_source="timeout",
                    )
                    turns.append(item)
                    reason = "environment_timeout"
                    break
                item["feedback_source"] = "simulator"
            item.update(
                {
                    key: state[key]
                    for key in ("observation", "reward", "terminated", "success")
                }
            )
            turns.append(item)
            success = bool(state["success"])
            if state["terminated"]:
                reason = "success" if success else "environment_terminal"
                break
            if turn + 1 == int(settings.get("max_turns", 30)):
                break
            suffix = chat.continuation(state["observation"], tokens[-1], stops)
            if len(ids) + len(suffix) >= total_cap:
                reason = "context_budget"
                break
            ids.extend(suffix)  # observation/envelope: actor mask remains zero
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
    return {"input_ids": ids, "actor_positions": positions, "trace": trace}


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
    return RolloutBatch(
        ids,
        mask,
        responses,
        valid,
        torch.full_like(responses, float("nan"), dtype=torch.float32),
        1,
        positions,
        actor_mask,
        [episode["trace"] for episode in episodes],
    )


def generate_training_episodes(model, engine, tasks, tokenizer, config, device, seeds):
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
    return collate_episodes(episodes, tokenizer.pad_token_id, device)
