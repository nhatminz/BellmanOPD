from pathlib import Path
import os
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_launchers_resolve_paths_from_script_directory():
    for path in (ROOT / "scripts").glob("*.sh"):
        text = path.read_text(encoding="utf-8")
        assert "SCRIPT_DIR" in text
    train = (ROOT / "scripts/train.sh").read_text(encoding="utf-8")
    assert "MAX_RESPONSE_LEN" in train and "TRAIN_EVAL_MAX_NEW_TOKENS" in train
    assert "TRAIN_EVAL_SEED" in train
    assert "TRAIN_EVAL_BENCHMARKS" in train
    for benchmark in (
        "Competition-MATH",
        "MATH-500",
        "AIME24",
        "AIME25",
        "GPQA-Diamond",
        "AMC23",
    ):
        assert benchmark in train
    assert "selector.cmt_ablation_arm" in train


def test_common_defaults_are_canonical():
    text = (ROOT / "configs/common.yaml").read_text(encoding="utf-8")
    for expected in (
        "cmt_allocation_kl: 0.5",
        "cmt_allocation_mode: direct_bounded_gibbs",
        "cmt_correction_mode: tanh_q99",
        "cmt_gamma: 1.0",
        "cmt_successor_lambda: 1.0",
        "top_k: 16",
        "max_new_tokens: 4096",
        "learning_rate: 5.0e-6",
    ):
        assert expected in text
    gamma_script = (ROOT / "scripts/sweep_gamma.sh").read_text(encoding="utf-8")
    assert "CMT_GAMMA" in gamma_script
    assert "train.sh\" g_d" in gamma_script


def test_train_launcher_is_cwd_independent_and_keeps_token_budgets():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "run"
        env = dict(os.environ, ABLATION_DRY_RUN="true", OUTPUT_DIR=str(output))
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/train.sh"), "g_x"],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "train_max_new_tokens=4096 eval_max_new_tokens=7168" in result.stdout
        assert "CMT gain support: student_topk" in result.stdout
        assert "CMT robust correction: tanh_q99" in result.stdout
        assert "CMT allocation mode: direct_bounded_gibbs" in result.stdout
        assert (output / "ablation_spec.json").is_file()


def test_g_arm_dry_run_records_student_topk_gain_support():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "g"
        env = dict(os.environ, ABLATION_DRY_RUN="true", OUTPUT_DIR=str(output))
        subprocess.run(
            ["bash", str(ROOT / "scripts/train.sh"), "g"],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        import json

        spec = json.loads((output / "ablation_spec.json").read_text(encoding="utf-8"))
        assert spec["arm"] == "g"
        assert spec["cmt_gain_support"] == "student_topk"
        assert spec["cmt_correction_mode"] == "tanh_q99"
        assert spec["cmt_allocation_mode"] == "direct_bounded_gibbs"


def test_topk_ablation_resolves_multiple_student_topk_values():
    import json

    with tempfile.TemporaryDirectory() as temp:
        for top_k in (8, 16, 32):
            output = Path(temp) / f"topk-{top_k}"
            env = dict(
                os.environ,
                ABLATION_DRY_RUN="true",
                OUTPUT_DIR=str(output),
                TOP_K=str(top_k),
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/train.sh"), "g_d"],
                cwd="/tmp",
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            spec = json.loads(
                (output / "ablation_spec.json").read_text(encoding="utf-8")
            )
            assert spec["top_k"] == top_k
            assert spec["cmt_gain_support"] == "student_topk"
            assert f"top_k={top_k}" in result.stdout


def test_topk_launcher_keeps_current_learning_rate_default():
    text = (ROOT / "scripts/sweep_topk.sh").read_text(encoding="utf-8")
    assert 'LEARNING_RATE="${LEARNING_RATE:-5e-6}"' in text


def test_lambda_launcher_has_exact_four_defaults_and_forced_modes():
    text = (ROOT / "scripts/run_lambda_ablation.sh").read_text(encoding="utf-8")
    assert "0.25 0.5 1.0 2.0" in text
    assert "CMT_CORRECTION_MODE=tanh_q99" in text
    assert "CMT_ALLOCATION_MODE=direct_bounded_gibbs" in text
    assert '"${SCRIPT_DIR}/train.sh" g_d' in text


def test_plot_launcher_uses_a_fresh_named_directory():
    text = (ROOT / "scripts/plot_ablation.sh").read_text(encoding="utf-8")
    assert "FIGURE_ROOT" in text
    assert "PLOT_TAG" in text
    assert "BENCHMARKS" in text
    assert "--benchmarks" in text
    assert "date +%Y%m%d_%H%M%S_%N" in text
    assert "GD_CMT_RUN_NAME" in text
    assert "--g-d-output" in text
    assert "--g-d-run-name" in text
