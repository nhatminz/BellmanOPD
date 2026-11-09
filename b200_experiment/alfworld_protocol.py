"""Versioned ALFWorld interaction protocol; no model or simulator imports."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re

LEGACY_SYSTEM_PROMPT = (
    "You are an agent in an interactive household environment. Read the task and "
    "observations, then output exactly one ALFWorld command per turn, on one line. "
    "Examples of command formats: go to receptacle 1; open receptacle 1; "
    "take object 1 from receptacle 1; move object 1 to receptacle 1; "
    "heat object 1 with microwave 1; clean object 1 with sinkbasin 1; "
    "cool object 1 with fridge 1; use desklamp 1; look; inventory. "
    "Do not invent observations or output multiple commands."
)
SYSTEM_PROMPT = """You control an agent in an interactive household environment.
Complete the stated task using the observations and exact object names and numbers.
Respond with exactly ONE command on ONE line. Do not add explanations or more commands.
When available, the admissible commands describe legal actions now, not an optimal plan.
Command format examples (each line is a separate example):
look
inventory
go to desk 1
open drawer 1
close drawer 1
take cd 1 from drawer 1
move cd 1 to desk 1
put apple 1 into bowl 1
heat apple 1 with microwave 1
clean apple 1 with sinkbasin 1
cool apple 1 with fridge 1
use desklamp 1
examine desk 1
slice apple 1 with knife 1
Use the current environment's commands and objects. Do not invent observations.
"""
LEGACY_COMMAND = re.compile(
    r"(?:look|inventory|help|(?:go to|open|close|take|move|put|heat|clean|cool|use|examine) [^\n\r;]+)",
    re.IGNORECASE,
)
# ALFWorld's demangler emits one entity type followed by its numbered instance.
ENTITY = r"[a-z][a-z0-9_]* [1-9][0-9]*"
COMMAND = re.compile(
    rf"(?:look|inventory|help|go to {ENTITY}|(?:open|close|use|examine) {ENTITY}"
    rf"|take {ENTITY} from {ENTITY}|move {ENTITY} to {ENTITY}"
    rf"|put {ENTITY} (?:in|into|on|to) {ENTITY}"
    rf"|(?:heat|clean|cool|slice) {ENTITY} with {ENTITY})",
    re.IGNORECASE,
)


def _unwrap(response):
    text = response.strip()
    if text.startswith("{"):

        def unique_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Duplicate JSON action field")
                value[key] = item
            return value

        try:
            value = json.loads(text, object_pairs_hook=unique_pairs)
        except (ValueError, TypeError):
            return None
        if (
            not isinstance(value, dict)
            or set(value) != {"action"}
            or not isinstance(value["action"], str)
        ):
            return None
        return value["action"].strip()
    if text.startswith("<action>"):
        match = re.fullmatch(r"<action>([^<>]+)</action>", text, re.DOTALL)
        return match.group(1).strip() if match else None
    return text


def parse_action_details(response, version=2):
    text = _unwrap(response)
    recovered = False
    if text is not None and version == 2 and text.endswith((";", ".")):
        text = text[:-1].rstrip()
        recovered = True
    pattern = LEGACY_COMMAND if version == 1 else COMMAND
    action = text if text is not None and pattern.fullmatch(text) else None
    return {
        "action": action,
        "parser_failure": action is None,
        "punctuation_recovered": bool(recovered and action is not None),
    }


def parse_action(response, protocol_version=2):
    return parse_action_details(response, protocol_version)["action"]


def format_observation(
    state, task_goal, settings, *, parser_error=None, repeated_warning=False
):
    if int(settings.get("protocol_version", 1)) == 1:
        return parser_error or state["observation"]
    mode = settings.get("action_prompt_mode", "admissible")
    if mode not in {"admissible", "legacy"}:
        raise ValueError(f"Unknown ALFWorld action_prompt_mode: {mode}")
    parts = (
        [state["observation"]]
        if mode == "legacy"
        else [
            f"Task goal: {task_goal}",
            f"Environment observation:\n{state['observation']}",
        ]
    )
    if parser_error:
        parts.append(
            f"Action parser feedback (no environment step): {parser_error}\nThe game state is unchanged; the observation above is the last real simulator observation."
        )
    if repeated_warning and settings.get("repeated_action_warning", False):
        parts.append(
            "Interaction warning: the same command was repeated without a changed observation. Reconsider how to complete the task."
        )
    if mode == "admissible":
        actions = state.get("admissible_actions", [])
        parts.append(
            "Currently admissible commands (legal, not necessarily useful):\n"
            + "\n".join(actions)
        )
    parts.append("Output exactly one command on one line.")
    return "\n\n".join(parts)


def protocol_manifest(config):
    settings = config["alfworld"]
    version = int(settings.get("protocol_version", 1))
    if version not in {1, 2}:
        raise ValueError(f"Unsupported ALFWorld protocol_version={version}")
    mode = settings.get(
        "action_prompt_mode", "legacy" if version == 1 else "admissible"
    )
    if mode not in {"admissible", "legacy"} or (version == 1 and mode != "legacy"):
        raise ValueError(
            "Protocol v1 requires legacy observation-only mode; v2 supports admissible or legacy"
        )
    if settings.get("invalid_trajectory_filter", "none") != "none":
        raise ValueError(
            "This OPD implementation preserves all sampled actor tokens; invalid_trajectory_filter must be none. Filtering would be a separate estimator ablation."
        )
    rollout = config["rollout"]
    from .alfworld_rollout import AppendOnlyChat

    implementation = "\n".join(
        inspect.getsource(fn)
        for fn in (_unwrap, parse_action_details, format_observation, AppendOnlyChat)
    )
    return {
        "version": version,
        "system_prompt": LEGACY_SYSTEM_PROMPT if version == 1 else SYSTEM_PROMPT,
        "observation_formatter": "observation_only_v1"
        if version == 1
        else "task_observation_actions_feedback_v2",
        "parser": "legacy_regex_v1"
        if version == 1
        else "single_command_optional_one_punctuation_v2",
        "parser_expression": (LEGACY_COMMAND if version == 1 else COMMAND).pattern,
        "implementation_sha256": hashlib.sha256(implementation.encode()).hexdigest(),
        "chat_boundary": "append_only_native_v2"
        if version == 2
        else "append_only_native_v1",
        "action_prompt_mode": mode,
        "ppo_normalizer": "full_vocabulary_consistent"
        if version == 2
        else "legacy_topk_backward",
        "action_projection": False,
        "repeated_action_warning": bool(settings.get("repeated_action_warning", False)),
        "context_policy": "full_history_stop_no_truncation",
        "limits": {
            k: settings.get(k)
            for k in ("max_turns", "max_turn_tokens", "max_episode_tokens")
        },
        "generation": {
            k: rollout.get(k) for k in ("max_new_tokens", "temperature", "top_p")
        },
        "stop_token_ids": config.get("models", {}).get("stop_token_ids"),
        "chat_template_kwargs": config.get("data", {}).get("chat_template_kwargs", {}),
    }


def resolve_protocol(config):
    result = copy.deepcopy(config)
    manifest = protocol_manifest(result)
    settings = result["alfworld"]
    settings["protocol_version"] = manifest["version"]
    settings["action_prompt_mode"] = manifest["action_prompt_mode"]
    settings["protocol_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    settings["protocol_manifest"] = manifest
    return result
