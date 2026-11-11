"""Lossless ALFWorld episode JSONL reading and observed-outcome diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .alfworld_diagnostics import merge_counts, metrics_from_counts
from .alfworld_protocol import parse_action_details


def read_episode_jsonl(path):
    """Iterate physical LF-delimited records, preserving Unicode in responses.

    str.splitlines() also splits U+0085/U+2028/U+2029 *inside JSON strings*.
    json.dumps(ensure_ascii=False) legitimately leaves those characters intact.
    """
    path = Path(path)
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except (ValueError, UnicodeError) as error:
                raise ValueError(
                    f"Invalid episode JSONL record in {path}, physical line {number}: {error}. "
                    "Record was not skipped or repaired; inspect the source file."
                ) from error
            if not isinstance(trace, dict):
                raise ValueError(
                    f"Expected an episode object in {path}, physical line {number}"
                )
            yield trace


def read_completed_seed(directory, worker_count):
    """Return only completed artifacts; partial/missing seeds need a fresh run.

    Corrupt *completed* artifacts raise with file/line instead of becoming zeros.
    The caller still verifies every task/sample/seed/protocol before reuse.
    """
    directory = Path(directory)
    shards = [directory / f"rank-{rank:05d}.jsonl" for rank in range(worker_count)]
    if shards and all(p.is_file() and p.with_suffix(".done").is_file() for p in shards):
        return [trace for path in shards for trace in read_episode_jsonl(path)]
    episodes = directory / "episodes.jsonl"
    if episodes.is_file() and (directory / "summary.json").is_file():
        return list(read_episode_jsonl(episodes))
    return None


def seed_diagnostics(seed_results, traces, protocol_version):
    """Describe observed failures without modifying actions or success labels."""
    overall = metrics_from_counts(
        merge_counts([r["raw_counts"] for r in seed_results.values()])
    )
    counts = overall["raw_counts"]["counts"]
    episodes, tokens = counts.get("episodes", 0), counts.get("actor_token_count", 0)
    reasons = overall["termination_reason_distribution"]
    invalid_responses = Counter(
        {
            response: count
            for response, count in overall["raw_counts"]["response_counts"].items()
            if parse_action_details(response, protocol_version)["parser_failure"]
        }
    )
    return {
        "episodes": episodes,
        "successes": counts.get("successes", 0),
        "zero_success": episodes > 0 and counts.get("successes", 0) == 0,
        "success_rate": overall["success_rate"],
        "parser_failure_rate": overall["parser_failure_rate"],
        "simulator_invalid_action_rate": overall["simulator_invalid_action_rate"],
        "repeated_action_rate": overall["repeated_action_rate"],
        "look_action_rate": overall["look_action_rate"],
        "invalid_actor_token_fraction": counts.get("invalid_actor_token_count", 0)
        / tokens
        if tokens
        else None,
        "termination_reason_counts": reasons,
        "termination_reason_fractions": {
            reason: count / episodes for reason, count in reasons.items()
        }
        if episodes
        else {},
        "top_parser_failures": [
            {"response": text, "count": n}
            for text, n in invalid_responses.most_common(20)
        ],
        "response_unicode_line_separator_occurrences": {
            f"U+{ord(separator):04X}": sum(
                response.count(separator) * count
                for response, count in overall["raw_counts"]["response_counts"].items()
            )
            for separator in ("\u0085", "\u2028", "\u2029")
        },
        "top_actions": overall["top_actions"],
        "episode_vs_turn_success_mismatches": sum(
            bool(trace["success"])
            != any(turn.get("success", False) for turn in trace["turns"])
            for trace in traces
        ),
        "interpretation": "Observed simulator success labels and response statistics; aggregate diagnostics alone do not establish a training or simulator defect.",
    }
