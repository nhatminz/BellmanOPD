import torch
import pytest

from b200_experiment.opd_core import build_student_topk_opd_reference
from b200_experiment.selectors.cmt_selector import CMTSelector
from b200_experiment.selectors.pgt_selector import PGTSelector


def _inputs():
    student_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    teacher_ids = torch.tensor([[[1, 2, 7], [4, 8, 9]]])
    student = torch.log_softmax(
        torch.tensor([[[2.0, 1.0, -1.0], [1.5, 0.0, -0.5]]]), dim=-1
    )
    teacher = torch.log_softmax(
        torch.tensor([[[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]]]), dim=-1
    )
    # Cross scores are global model log-probabilities on both Top-K sets; the
    # selector conditionalizes both models on their literal union.
    teacher_on_student = torch.tensor(
        [[[-1.0, -0.5, -2.5], [-1.2, -2.0, -0.7]]]
    )
    student_on_teacher = torch.tensor(
        [[[-0.4, -1.1, -2.0], [-0.9, -1.8, -0.3]]]
    )
    valid = torch.tensor([[True, True]])
    return (
        student_ids,
        teacher_ids,
        student,
        teacher_on_student,
        teacher,
        student_on_teacher,
        valid,
    )


def test_pgt_builds_union_scoring_support_and_finite_gain():
    output = PGTSelector().compute_scores_from_topk(*_inputs())
    student_ids, teacher_ids, *_ = _inputs()
    assert output.candidate_ids.shape[-1] == 6
    assert output.support_mask[0, 0].sum().item() == 4
    assert output.support_mask[0, 1].sum().item() == 5
    teacher_only = set(teacher_ids.flatten().tolist()) - set(student_ids.flatten().tolist())
    assert teacher_only.intersection(output.candidate_ids.flatten().tolist())
    assert torch.isfinite(output.scores).all()
    assert (output.scores >= 0).all()
    assert torch.equal(output.scores, output.diagnostics["s_PGT"])
    p_mass = (output.student_candidate_log_probs.exp() * output.support_mask).sum(-1)
    q_mass = (output.teacher_candidate_log_probs.exp() * output.support_mask).sum(-1)
    assert torch.allclose(p_mass, torch.ones_like(p_mass), atol=1e-6)
    assert torch.allclose(q_mass, torch.ones_like(q_mass), atol=1e-6)
    assert output.diagnostics["support_definition"] == (
        "literal_union_student_topk_teacher_topk"
    )
    assert "student_union_mass" not in output.diagnostics
    assert "teacher_union_mass" not in output.diagnostics


def test_teacher_only_topk_values_change_local_teachability():
    args = list(_inputs())
    reference = PGTSelector().compute_scores_from_topk(*args)
    # The last teacher slot is teacher-only at both positions. Changing its
    # teacher probability changes the union-normalized local gain.
    args[4] = args[4].clone()
    args[4][..., -1] += 3.0
    changed = PGTSelector().compute_scores_from_topk(*args)
    assert torch.equal(changed.candidate_ids, reference.candidate_ids)
    assert not torch.allclose(changed.scores, reference.scores)


def test_cmt_gain_is_exact_student_topk_variance():
    args = list(_inputs())
    support = PGTSelector().compute_scores_from_topk(
        *args, gain_support="student_topk"
    )
    student_logp = args[2].float()
    teacher_on_student = args[3].float()
    p_log = student_logp - torch.logsumexp(student_logp, dim=-1, keepdim=True)
    q_log = teacher_on_student - torch.logsumexp(
        teacher_on_student, dim=-1, keepdim=True
    )
    p = p_log.exp()
    r = q_log - p_log
    expected = (p * (r - (p * r).sum(-1, keepdim=True)).square()).sum(-1)
    assert torch.allclose(support.diagnostics["gain"], expected, atol=1e-6)
    assert support.diagnostics["gain_support_definition"] == "student_topk"
    assert support.diagnostics["support_geometry"] == (
        "conditional_student_teacher_distributions_on_student_topk"
    )


def test_teacher_only_topk_values_do_not_change_cmt_local_gain():
    args = list(_inputs())
    base_support = PGTSelector().compute_scores_from_topk(
        *args, gain_support="student_topk"
    )
    args[4] = args[4].clone()
    args[4][..., -1] += 3.0
    changed_support = PGTSelector().compute_scores_from_topk(
        *args, gain_support="student_topk"
    )
    sampled = torch.tensor([[1, 4]])
    valid = torch.tensor([[True, True]])
    base = CMTSelector().compute_scores(base_support, sampled, valid)
    changed = CMTSelector().compute_scores(changed_support, sampled, valid)
    assert torch.allclose(changed.diagnostics["gain"], base.diagnostics["gain"])


def test_teacher_on_student_topk_values_change_cmt_local_gain():
    args = list(_inputs())
    reference = PGTSelector().compute_scores_from_topk(
        *args, gain_support="student_topk"
    )
    args[3] = args[3].clone()
    args[3][..., -1] += 2.0
    changed = PGTSelector().compute_scores_from_topk(
        *args, gain_support="student_topk"
    )
    assert not torch.allclose(
        changed.diagnostics["gain"], reference.diagnostics["gain"]
    )


def test_cmt_rejects_union_defined_local_gain():
    union_support = PGTSelector().compute_scores_from_topk(*_inputs())
    with pytest.raises(
        ValueError, match="exact Student Top-K support"
    ):
        CMTSelector().compute_scores(
            union_support,
            sampled_token_ids=torch.tensor([[1, 4]]),
            valid_mask=torch.tensor([[True, True]]),
        )


def test_union_score_is_zero_on_padding():
    args = list(_inputs())
    args[-1] = torch.tensor([[True, False]])
    output = PGTSelector().compute_scores_from_topk(*args)
    assert torch.all(output.scores[:, 1] == 0)


def test_constant_log_ratio_has_zero_projected_policy_gain():
    args = list(_inputs())
    # Make both cross-scored distributions equal on every union action.
    # The expected policy gradient is then zero even though the raw values are
    # non-zero, which is exactly the distinction from a divergence heuristic.
    args[3] = args[2].clone()
    args[1] = args[0].clone()
    args[4] = args[2].clone()
    args[5] = args[2].clone()
    output = PGTSelector().compute_scores_from_topk(*args)
    assert torch.allclose(output.scores, torch.zeros_like(output.scores), atol=1e-6)


def test_union_is_not_passed_to_policy_loss():
    inputs = _inputs()
    output = PGTSelector().compute_scores_from_topk(*inputs)
    student_ids, _, student_logp, teacher_on_student, *_ = inputs
    top_k = student_ids.shape[-1]
    reference = build_student_topk_opd_reference(
        student_ids,
        student_logp,
        teacher_on_student,
        torch.tensor([[True, True]]),
        top_k=top_k,
    )
    assert torch.equal(reference.candidate_ids, student_ids)
    assert reference.candidate_ids.shape[-1] == top_k
    assert output.candidate_ids.shape[-1] == 2 * top_k


def test_cmt_preserves_union_only_as_transition_support():
    support = PGTSelector().compute_scores_from_topk(
        *_inputs(), gain_support="student_topk"
    )
    result = CMTSelector().compute_scores(
        support,
        sampled_token_ids=torch.tensor([[1, 4]]),
        valid_mask=torch.tensor([[True, True]]),
    )
    assert torch.equal(result.candidate_ids, support.candidate_ids)
    assert result.candidate_ids.shape[-1] == 2 * _inputs()[0].shape[-1]
    assert result.diagnostics["gain_support_definition"] == "student_topk"
    assert result.diagnostics["transition_support_definition"] == (
        "literal_union_student_topk_teacher_topk"
    )
