"""Episode accounting and explicit protocol-collapse guards, never loss weights."""

from __future__ import annotations

import math
import warnings
from collections import Counter

RATE_METRICS = (
    "success_rate",
    "task_success_rate",
    "average_turns",
    "invalid_action_rate",
    "total_invalid_action_rate",
    "parser_failure_rate",
    "simulator_invalid_action_rate",
    "punctuation_recovery_rate",
    "valid_action_rate",
    "action_diversity",
    "unique_action_count",
    "look_action_rate",
    "first_action_look_rate",
    "repeated_action_rate",
    "unchanged_observation_rate",
    "actor_token_count",
)


def episode_counts(traces):
    counts = Counter(episodes=len(traces))
    hist = {
        name: Counter()
        for name in (
            "response_counts",
            "action_counts",
            "termination_reason_distribution",
        )
    }
    for trace in traces:
        turns = trace["turns"]
        counts.update(
            successes=int(bool(trace["success"])),
            turns=len(turns),
            invalid_actions=int(trace.get("invalid_actions", 0)),
            actor_token_count=int(trace.get("actor_tokens", 0)),
        )
        counts["invalid_actor_token_count"] += sum(
            t.get("actor_end", 0) - t.get("actor_start", 0)
            for t in turns
            if t.get("invalid", False)
        )
        hist["termination_reason_distribution"][
            trace.get("termination_reason", "unknown")
        ] += 1
        for index, turn in enumerate(turns):
            counts.update(
                parser_failures=int(turn.get("parser_failure", False)),
                simulator_invalid_actions=int(turn.get("simulator_invalid", False)),
                punctuation_recoveries=int(turn.get("punctuation_recovered", False)),
                repeated_actions=int(turn.get("repeated_action", False)),
                unchanged_observations=int(turn.get("unchanged_observation", False)),
                simulator_steps=int(turn.get("feedback_source") == "simulator"),
            )
            if "response" in turn:
                hist["response_counts"][turn["response"]] += 1
            action = turn.get("action")
            if action is not None:
                hist["action_counts"][action] += 1
                if action == "look":
                    counts["look_actions"] += 1
                    counts["first_action_look"] += int(index == 0)
    return {
        "counts": dict(counts),
        **{name: dict(value) for name, value in hist.items()},
    }


def merge_counts(reports):
    result = {
        name: Counter()
        for name in (
            "counts",
            "response_counts",
            "action_counts",
            "termination_reason_distribution",
        )
    }
    for report in reports:
        for key in result:
            result[key].update(report.get(key, {}))
    return {name: dict(values) for name, values in result.items()}


def metrics_from_counts(report):
    counts = report["counts"]
    n, turns = counts.get("episodes", 0), counts.get("turns", 0)
    invalid = counts.get("invalid_actions", 0)
    actions = report["action_counts"]
    success = counts.get("successes", 0) / max(n, 1)
    return {
        **counts,
        "success_rate": success,
        "task_success_rate": success,
        "average_turns": turns / max(n, 1),
        "invalid_action_rate": invalid / max(turns, 1),
        "total_invalid_action_rate": invalid / max(turns, 1),
        "parser_failure_rate": counts.get("parser_failures", 0) / max(turns, 1),
        "simulator_invalid_action_rate": counts.get("simulator_invalid_actions", 0)
        / max(turns, 1),
        "punctuation_recovery_rate": counts.get("punctuation_recoveries", 0)
        / max(turns, 1),
        "valid_action_rate": (turns - invalid) / max(turns, 1),
        "unique_action_count": len(actions),
        "action_diversity": len(actions) / max(sum(actions.values()), 1),
        "look_action_rate": counts.get("look_actions", 0) / max(turns, 1),
        "first_action_look_rate": counts.get("first_action_look", 0) / max(n, 1),
        "repeated_action_rate": counts.get("repeated_actions", 0) / max(turns, 1),
        "unchanged_observation_rate": counts.get("unchanged_observations", 0)
        / max(counts.get("simulator_steps", 0), 1),
        "actor_token_count": counts.get("actor_token_count", 0),
        "termination_reason_distribution": report["termination_reason_distribution"],
        "top_responses": [
            {"response": text, "count": count}
            for text, count in Counter(report["response_counts"]).most_common(20)
        ],
        "top_actions": [
            {"action": text, "count": count}
            for text, count in Counter(actions).most_common(20)
        ],
        "raw_counts": report,
    }


def episode_metrics(traces):
    return metrics_from_counts(episode_counts(traces))


def distributed_episode_metrics(traces, distributed):
    local = {
        "all": episode_counts(traces),
        "task_types": {
            kind: episode_counts([t for t in traces if t["task_type"] == kind])
            for kind in sorted({t["task_type"] for t in traces})
        },
    }
    gathered = distributed.all_gather_objects(local)
    result = metrics_from_counts(merge_counts([r["all"] for r in gathered]))
    kinds = sorted({k for r in gathered for k in r["task_types"]})
    result["task_types"] = {
        k: metrics_from_counts(
            merge_counts([r["task_types"][k] for r in gathered if k in r["task_types"]])
        )
        for k in kinds
    }
    return result


def check_collapse(metrics, settings):
    guard = settings.get("collapse_guard", {})
    mode = guard.get("mode", "warn")
    if mode not in {"off", "warn", "error"}:
        raise ValueError("alfworld.collapse_guard.mode must be off, warn or error")
    if mode == "off":
        return []
    turns = metrics.get("turns", 0)
    issues = []
    if turns and metrics.get("parser_failures", 0) == turns:
        issues.append("all actor responses failed parsing")
    if turns >= int(guard.get("min_turns", 32)):
        for metric, key, default in (
            ("parser_failure_rate", "parser_failure_threshold", 0.95),
            ("look_action_rate", "look_action_threshold", 0.98),
            ("repeated_action_rate", "repeated_action_threshold", 0.98),
        ):
            threshold = float(guard.get(key, default))
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid collapse guard threshold {key}")
            if metrics.get(metric, 0) >= threshold:
                issues.append(f"{metric}={metrics[metric]:.4f} >= {threshold}")
    if issues:
        message = (
            "ALFWorld protocol collapse: "
            + "; ".join(issues)
            + ". No action projection or token filtering was applied. Inspect episode traces and compare untouched student/teacher action distributions."
        )
        if mode == "error":
            raise RuntimeError(message + " Stopped before the optimizer update.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return issues
