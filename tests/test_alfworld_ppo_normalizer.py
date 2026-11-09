"""Identity PPO ratio must be one before an update, including truncated Top-K."""

import torch

from b200_experiment.opd_core import (
    build_student_topk_opd_reference,
    gather_candidate_log_probs,
)


def test_unchanged_policy_ratio_is_one_on_student_topk():
    logits = torch.zeros(1, 1, 8)
    log_p = logits.log_softmax(-1)
    ids = torch.tensor([[[0, 1]]])
    old = log_p.gather(-1, ids)
    reference = build_student_topk_opd_reference(
        ids,
        old,
        old - 0.1,
        torch.ones(1, 1, dtype=torch.bool),
        top_k=2,
        full_vocab_ppo=True,
    )
    current = gather_candidate_log_probs(
        logits, ids, temperature=1.0, chunk_steps=1, support_mask=reference.support_mask
    )
    ratio = (current - reference.old_student_log_probs).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio))


def test_v2_normalizer_changes_no_candidates_rewards_or_legacy_reference():
    ids = torch.tensor([[[0, 1]]])
    p = torch.tensor([[[-1.0, -2.0]]])
    q = torch.tensor([[[-2.0, -1.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    legacy = build_student_topk_opd_reference(ids, p, q, valid, top_k=2)
    repaired = build_student_topk_opd_reference(
        ids, p, q, valid, top_k=2, full_vocab_ppo=True
    )
    assert legacy.support_mask is not None and repaired.support_mask is None
    for field in (
        "candidate_ids",
        "old_student_log_probs",
        "teacher_log_probs",
        "student_weights",
        "advantages",
    ):
        torch.testing.assert_close(getattr(legacy, field), getattr(repaired, field))
