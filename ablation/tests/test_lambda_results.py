from __future__ import annotations

import json
from pathlib import Path

from ablation.analysis.lambda_results import BENCHMARK_ORDER, collect_lambda_results
from b200_experiment.config import apply_overrides, load_with_overlays


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_run(root, seed: int, lambda_value: float) -> None:
    run = root / f"lambda_{lambda_value}_seed{seed}"
    run.mkdir(parents=True)
    spec = {
        "ablation_kind": "lambda_successor",
        "arm": "g_d",
        "seed": seed,
        "rollout_seed": seed,
        "cmt_successor_lambda": lambda_value,
        "cmt_gain_support": "student_topk",
        "cmt_correction_mode": "tanh_q99",
        "cmt_correction_quantile": 0.99,
        "cmt_allocation_mode": "direct_bounded_gibbs",
        "top_k": 16,
        "learning_rate": 5e-6,
    }
    (run / "ablation_spec.json").write_text(
        json.dumps(spec) + "\n", encoding="utf-8"
    )
    eval_rows = [
        {
            "step": step,
            "benchmarks": {
                name: {"accuracy": 0.5 + 0.01 * step + 0.001 * lambda_value}
                for name in BENCHMARK_ORDER
            },
        }
        for step in (0, 10)
    ]
    (run / "eval_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in eval_rows), encoding="utf-8"
    )
    metric_rows = []
    for step in (1, 2):
        metric_rows.append(
            {
                "step": step,
                "allocation_kl_final": 0.02,
                "normalized_ess": 0.8,
                "fraction_at_weight_min": 0.1,
                "fraction_at_weight_max": 0.2,
                "weight_final_min": 0.5,
                "weight_final_max": 2.0,
                "correction_kappa": 2.0,
                "sequential_gain_raw_abs_q95": lambda_value,
                "sequential_gain_raw_abs_q99": 2.0 * lambda_value,
                "correction_saturation_rate_1kappa": 0.1,
                "correction_saturation_rate_2kappa": 0.05,
                "selector": {"w": {"mean": 1.0, "std": 0.2, "max": 2.0}},
            }
        )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows), encoding="utf-8"
    )


def test_lambda_aggregation_validates_config_and_supports_multiple_seeds(tmp_path):
    values = (0.25, 0.5, 1.0, 2.0)
    for seed in (42, 43):
        for value in values:
            _write_run(tmp_path, seed, value)
    summary = collect_lambda_results(
        tmp_path, expected_lambdas=values, metric="accuracy"
    )
    assert len(summary["runs"]) == 8
    assert "only cmt_successor_lambda differs" in summary["config_validation"]
    point = next(
        row
        for row in summary["final_eval_summary"]
        if row["lambda"] == 1.0 and row["benchmark"] == "MATH-500"
    )
    assert point["n"] == 2
    ratio = next(
        row
        for row in summary["training_summary"]
        if row["lambda"] == 2.0
        and row["step"] == 1
        and row["diagnostic"] == "lambda_d_over_kappa_q99"
    )
    assert ratio["mean"] == 2.0
    assert ratio["n"] == 2


def test_resolved_lambda_configs_differ_only_in_successor_lambda():
    base = load_with_overlays(
        REPO_ROOT / "configs/qwen3_b200_cmt.yaml",
        [REPO_ROOT / "ablation/configs/common.yaml"],
    )
    resolved = [
        apply_overrides(base, [f"selector.cmt_successor_lambda={value}"])
        for value in (0.25, 0.5, 1.0, 2.0)
    ]
    assert all(
        config["selector"]["cmt_correction_mode"] == "tanh_q99"
        and config["selector"]["cmt_allocation_mode"]
        == "direct_bounded_gibbs"
        for config in resolved
    )
    for config in resolved:
        config["selector"].pop("cmt_successor_lambda")
    assert all(config == resolved[0] for config in resolved[1:])
