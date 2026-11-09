"""Opt-in multi-turn vLLM/HF comparison on equivalent sampled distributions."""

from __future__ import annotations

from dataclasses import replace

import torch

from .scoring import rollout_forward_inputs, select_rollout_logits


def sampled_distribution_log_probs(logits, token_ids, temperature, top_p):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError(
            "Sampling log-prob diagnostics require temperature>0, top_p in (0,1]"
        )
    raw = logits.float().log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    scaled = logits.float() / float(temperature)
    if top_p < 1:
        sorted_logits, order = scaled.sort(-1, descending=True)
        probabilities = sorted_logits.softmax(-1)
        remove = probabilities.cumsum(-1) - probabilities >= float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(scaled, -torch.inf)
        scaled = filtered.scatter(-1, order, sorted_logits)
    sampled = scaled.log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return raw, sampled


@torch.inference_mode()
def compare_multiturn_logprobs(model, rollout, config, distributed):
    sanity = config["rollout"]["vllm"]["logprob_sanity"]
    missing = rollout.rollout_logprob_kind != "sampling_temperature_top_p" or not bool(
        torch.isfinite(rollout.rollout_log_probs[rollout.valid_mask]).all()
    )
    if distributed.any(missing):
        raise RuntimeError(
            "ALFWorld debug needs vLLM processed_logprobs for every sampled token; check --logprobs-mode processed_logprobs and token IDs in the server response"
        )
    cap = max(1, int(sanity.get("max_tokens_per_rank", 32)))
    per_turn = max(1, int(sanity.get("max_tokens_per_turn", 1)))
    candidates = []
    for row, trace in enumerate(rollout.episode_traces):
        for turn in trace["turns"]:
            candidates.extend(
                (row, i, turn["turn"])
                for i in range(
                    turn["actor_start"],
                    min(turn["actor_end"], turn["actor_start"] + per_turn),
                )
            )
    # The first token of EVERY turn is considered before additional local rows.
    # With cap=32 and default max_turns=30, later turns are checked as well.
    selected = candidates[:cap]
    differences, raw_values, sampled_values, later = [], [], [], 0
    model.eval()
    for row in sorted({r for r, _, _ in selected}):
        entries = [(i, turn) for r, i, turn in selected if r == row]
        indices = torch.tensor([i for i, _ in entries], device=rollout.input_ids.device)
        mini = replace(
            rollout,
            input_ids=rollout.input_ids[row : row + 1],
            attention_mask=rollout.attention_mask[row : row + 1],
            response_ids=rollout.response_ids[row : row + 1].index_select(1, indices),
            actor_positions=rollout.actor_positions[row : row + 1].index_select(
                1, indices
            ),
            valid_mask=torch.ones(
                (1, len(entries)), dtype=torch.bool, device=indices.device
            ),
        )
        kwargs, selection = rollout_forward_inputs(
            model, mini, torch.tensor([0], device=indices.device), len(entries)
        )
        logits = select_rollout_logits(model(**kwargs).logits, selection)
        raw, sampled = sampled_distribution_log_probs(
            logits,
            mini.response_ids,
            float(config["rollout"]["temperature"]),
            float(config["rollout"]["top_p"]),
        )
        observed = rollout.rollout_log_probs[row].index_select(0, indices)
        differences.extend((sampled.squeeze(0) - observed).abs().tolist())
        raw_values.extend(raw.flatten().tolist())
        sampled_values.extend(sampled.flatten().tolist())
        later += sum(turn > 0 for _, turn in entries)
    count = distributed.sum_int(len(differences))
    if not count:
        raise RuntimeError("ALFWorld log-prob debug found no actor tokens")
    maximum = distributed.max_float(max(differences, default=0.0))
    tolerance = float(sanity.get("tolerance", 0.05))
    result = {
        "enabled": True,
        "distribution": "sampling_temperature_top_p",
        "compared_tokens": count,
        "later_turn_tokens": distributed.sum_int(later),
        "mean_abs_error": distributed.sum_float(sum(differences)) / count,
        "max_abs_error": maximum,
        "tolerance": tolerance,
        "passed": maximum <= tolerance,
        "hf_raw_policy_logprob_mean": distributed.sum_float(sum(raw_values)) / count,
        "hf_sampling_logprob_mean": distributed.sum_float(sum(sampled_values)) / count,
    }
    if not result["passed"] and sanity.get("fail_on_mismatch", True):
        raise RuntimeError(
            f"ALFWorld vLLM/HF sampling log-prob mismatch: max={maximum:.6f}, tolerance={tolerance}; inspect weights, later-turn prefixes, temperature/top-p and EOS"
        )
    return result
