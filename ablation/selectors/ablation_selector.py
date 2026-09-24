"""Description and validation of CMT ablation arms.

The numerical score remains in ``b200_experiment.selectors.CMTSelector``;
this module deliberately does not duplicate the production formula.
"""

from __future__ import annotations

from typing import Final


ARMS: Final = ("g", "g_x", "g_d", "d_only")


def normalize_arm(value: str) -> str:
    arm = str(value).strip().lower()
    arm = {
        "gx": "g_x",
        "gd": "g_d",
        "cmt": "g_d",
        "canonical": "g_d",
        "d": "d_only",
        "d-only": "d_only",
    }.get(arm, arm)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {', '.join(ARMS)}; got {value!r}")
    return arm


def score_definition(arm: str) -> str:
    return {
        "g": "S_t = g_t",
        "g_x": "S_t = g_t + successor_excess_t",
        "g_d": "S_t = canonical CMT learning_value_t = g_t + sequential_gain_t",
        "d_only": "S_t = D_t = sequential_gain_t (without g_t)",
    }[normalize_arm(arm)]
