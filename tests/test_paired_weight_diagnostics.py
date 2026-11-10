import copy
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from b200_experiment.cmt_random_control import permute_final_cmt_weights
from b200_experiment.distributed import DistributedContext
from b200_experiment.paired_weight_diagnostics import (
    PAIRED_FIELDS,
    TRAJECTORY_SOURCE,
    PairedWeightDiagnosticsLogger,
    flatten_paired,
    measure_paired_weights,
)
from b200_experiment.selectors.cmt_selector import cmt_allocation
from b200_experiment.trainer import _opd_train_step, _append_train_metrics_csv
from test_cmt_random_control import batch, config_for_control, reference_for
from test_distributed import _TinyCausalLM, _parameters


def measure(prow, gain_only, d=None, coverage=None):
    count = prow.numel()
    d = torch.zeros(count) if d is None else d
    coverage = torch.zeros(count) if coverage is None else coverage
    _, shuffled, _ = permute_final_cmt_weights(
        prow, prow, experiment_seed=42, optimizer_step=1
    )
    return measure_paired_weights(
        torch.arange(count).float(),
        d,
        coverage,
        prow,
        gain_only,
        shuffled,
        weight_min=0.5,
        weight_max=2.0,
        allocation_mode="direct_bounded_gibbs",
    )


def test_exact_zero_outside_intersection_and_paired_token_statistics():
    prow = torch.tensor([0.5, 0.5, 1, 1, 1, 2])
    gain_only = torch.tensor([1, 1, 0.5, 0.5, 1, 2])
    d = torch.tensor([0, 0, 0, 1, -1, 1e-12])
    coverage = torch.tensor([0, 1, 0, 1, 1, 1])
    result, payload = measure(prow, gain_only, d, coverage)
    counts = result["zero_score_statistics"]
    assert counts["valid_token_count"] == 6
    assert counts["zero_prospective_count"] == 3
    assert counts["out_of_support_count"] == counts["out_of_support_zero_count"] == 2
    assert counts["in_support_zero_count"] == 1
    assert counts["zero_prospective_fraction"] == 0.5
    assert counts["zero_attributable_to_out_of_support_fraction"] == pytest.approx(
        2 / 3
    )
    assert counts["in_support_zero_fraction"] == pytest.approx(1 / 6)
    assert counts["in_support_zero_conditional_fraction"] == 0.25
    zero = result["paired_comparison"]["zero_prospective"]
    assert zero["count"] == 3
    assert zero["prow_weight_mean"] == pytest.approx(2 / 3)
    assert zero["g_only_weight_mean"] == pytest.approx(2.5 / 3)
    assert zero["signed_diff_mean"] == pytest.approx(-0.5 / 3)
    assert (
        zero["abs_diff_mean"] == zero["abs_diff_median"] == zero["abs_diff_p95"] == 0.5
    )
    assert zero["different_fraction"] == 1
    assert torch.equal(
        payload["out_of_support"], torch.tensor([0.5, 0.5], dtype=torch.float64)
    )
    assert all(result["checks"].values())
    assert (
        result["allocations"]["prow"]["realized_kl"]
        == result["allocations"]["random"]["realized_kl"]
    )
    assert set(flatten_paired(result)) == set(PAIRED_FIELDS) - {
        "paired_weight_diagnostics_time_sec"
    }


def test_outside_and_zero_are_not_silently_equated_and_empty_subsets_are_null():
    weights = torch.ones(4)
    result, _ = measure(
        weights,
        weights,
        torch.tensor([0.0, 1e-30, 2.0, 0.0]),
        torch.tensor([0, 0, 1, 1]),
    )
    counts = result["zero_score_statistics"]
    assert (
        counts["out_of_support_count"] == 2 and counts["out_of_support_zero_count"] == 1
    )
    assert (
        counts["out_of_support_nonzero_count"] == 1
        and counts["in_support_zero_count"] == 1
    )
    empty, _ = measure(weights, weights, torch.ones(4), torch.ones(4))
    for pair in empty["paired_comparison"].values():
        assert pair["count"] == 0
        assert (
            pair["prow_weight_mean"]
            is pair["abs_diff_median"]
            is pair["different_fraction"]
            is None
        )
    json.dumps(empty, allow_nan=False)


def test_diagnostics_and_file_logger_do_not_modify_gradients_or_rng(tmp_path):
    prow = torch.tensor([0.5, 0.5, 1, 2], requires_grad=True)
    gain_only = torch.ones(4, requires_grad=True)
    torch_state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    result, payload = measure(prow, gain_only)
    assert prow.grad is gain_only.grad is None
    assert all(not value.requires_grad for value in payload.values())
    logger = PairedWeightDiagnosticsLogger(tmp_path, enabled=True)
    logger.write(
        1, result, payload, allocation_id="r0:g0", rollout_id=0, ppo_group_index=0
    )
    logger.finalize()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and np.array_equal(
        numpy_state[1], after_numpy[1]
    )
    assert numpy_state[2:] == after_numpy[2:]
    assert all(
        torch.equal(old, new)
        for old, new in zip(cuda_state, torch.cuda.get_rng_state_all())
    )


def test_token_weighted_pooling_exact_quantiles_deduplication_and_resume(tmp_path):
    first, first_payload = measure(torch.tensor([0.5, 0.5, 1.0, 2.0]), torch.ones(4))
    second, second_payload = measure(torch.ones(12), torch.ones(12))
    logger = PairedWeightDiagnosticsLogger(tmp_path, enabled=True)
    logger.write(
        1, first, first_payload, allocation_id="r0:g0", rollout_id=0, ppo_group_index=0
    )
    logger.write(
        2,
        second,
        second_payload,
        allocation_id="r0:g1",
        rollout_id=0,
        ppo_group_index=1,
    )
    logger.write(
        2,
        second,
        second_payload,
        allocation_id="r0:g1",
        rollout_id=0,
        ppo_group_index=1,
    )
    result = logger.finalize()
    assert result["observed_optimizer_steps"] == 2 and result["quantiles_finalized"]
    assert result["trajectory_source"] == TRAJECTORY_SOURCE
    assert result["overall"]["zero_score_statistics"]["valid_token_count"] == 16
    pair = result["overall"]["paired_comparison"]["zero_prospective"]
    pooled = np.array([0.5, 0.5, 0, 1] + [0] * 12)
    assert pair["abs_diff_mean"] == 0.125  # not a mean of group means (0.25)
    assert pair["abs_diff_median"] == np.median(pooled) == 0
    assert pair["abs_diff_p95"] == np.quantile(pooled, 0.95)
    rows = list(csv.DictReader((tmp_path / "rebuttal_weight_diagnostics.csv").open()))
    assert [row["scope"] for row in rows] == ["overall", "step", "step"]
    assert float(rows[0]["paired_zero_prospective_abs_diff_mean"]) == 0.125
    resumed = PairedWeightDiagnosticsLogger(tmp_path, enabled=True, resume_step=1)
    assert (
        resumed.finalize()["overall"]["zero_score_statistics"]["valid_token_count"] == 4
    )
    resumed.write(
        2,
        second,
        second_payload,
        allocation_id="r0:g1",
        rollout_id=0,
        ppo_group_index=1,
    )
    assert (
        resumed.finalize()["overall"]["zero_score_statistics"]["valid_token_count"]
        == 16
    )


def test_disabled_logger_writes_nothing(tmp_path):
    logger = PairedWeightDiagnosticsLogger(tmp_path / "disabled", enabled=False)
    assert logger.finalize() is None
    assert not (tmp_path / "disabled").exists()


def test_csv_schema_when_diagnostics_enabled_or_disabled_mid_run(tmp_path):
    path = tmp_path / "train_metrics.csv"
    _append_train_metrics_csv(path, {"step": 1, "train_loss": 0.75})
    with path.open() as handle:
        original_fields = csv.DictReader(handle).fieldnames
    assert not any(name.startswith("paired_") for name in original_fields)
    report, _ = measure(torch.ones(4), torch.ones(4))
    _append_train_metrics_csv(
        path, {"step": 2, "train_loss": 0.5, **flatten_paired(report)}
    )
    _append_train_metrics_csv(path, {"step": 3, "train_loss": 0.25})
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames[: len(original_fields)] == original_fields
    assert [row["train_loss"] for row in rows] == ["0.75", "0.5", "0.25"]
    assert [row["paired_valid_token_count"] for row in rows] == ["", "4", ""]
    assert all(None not in row for row in rows)  # no mismatched header/row widths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optional")
def test_cuda_measurement_is_detached_and_keeps_cuda_rng():
    prow = torch.tensor([0.5, 0.5, 1.0, 2.0], device="cuda", requires_grad=True)
    gain_only = torch.ones(4, device="cuda", requires_grad=True)
    rng = torch.cuda.get_rng_state().clone()
    cpu_rng = torch.get_rng_state().clone()
    report, payload = measure(
        prow,
        gain_only,
        torch.tensor([0.0, 1e-12, 0.0, 1.0], device="cuda"),
        torch.tensor([0.0, 1.0, 1.0, 0.0], device="cuda"),
    )
    assert all(report["checks"].values())
    assert report["zero_score_statistics"]["zero_prospective_count"] == 2
    assert all(value.is_cuda and not value.requires_grad for value in payload.values())
    assert prow.grad is gain_only.grad is None
    assert torch.equal(rng, torch.cuda.get_rng_state()) and torch.equal(
        cpu_rng, torch.get_rng_state()
    )


def test_paired_allocation_validation_rejects_corrupt_random_multiset():
    weights = torch.tensor([0.5, 0.5, 1.0, 2.0])
    with pytest.raises(AssertionError, match="invariants"):
        measure_paired_weights(
            torch.ones(4),
            torch.zeros(4),
            torch.ones(4),
            weights,
            weights,
            torch.ones(4),
            weight_min=0.5,
            weight_max=2,
            allocation_mode="direct_bounded_gibbs",
        )


def paired_inputs(rollout, rank=0):
    gain = (
        torch.arange(rollout.valid_mask.numel())
        .reshape(rollout.valid_mask.shape)
        .float()
        + rank * 10
    )
    d = torch.where(gain.remainder(3).eq(0), 0, gain * 0.1)
    coverage = gain.remainder(2).ne(0).float()
    return gain, d, coverage


@pytest.mark.parametrize("micro_batch", [1, 2])
def test_paired_diagnostics_do_not_change_random_updates_or_model_forward_count(
    monkeypatch,
    micro_batch,
):
    import b200_experiment.trainer as trainer

    torch.manual_seed(73)
    initial = _TinyCausalLM()
    rollout = batch()
    reference = reference_for(initial, rollout)
    gain, d, coverage = paired_inputs(rollout)
    # Poison padding so including invalid tokens cannot silently pass.
    gain[~rollout.valid_mask] = 999
    d[~rollout.valid_mask] = 0
    coverage[~rollout.valid_mask] = 0
    corrected = gain + d.tanh()
    initial_ids, initial_scores = rollout.input_ids.clone(), corrected.clone()
    results, parameters, forward_counts = [], [], []
    allocated_inputs = []
    original_allocate = trainer.cmt_allocation

    def allocate(values, epsilon, **kwargs):
        allocated_inputs.append((values.detach().clone(), epsilon, kwargs))
        return original_allocate(values, epsilon, **kwargs)

    monkeypatch.setattr(trainer, "cmt_allocation", allocate)
    for enabled in [False, True]:
        model = copy.deepcopy(initial)
        config = config_for_control(True)
        config["training"]["micro_batch_size_per_gpu"] = micro_batch
        config["selector"]["cmt_paired_weight_diagnostics"] = enabled
        extra = (
            {
                "paired_local_gain": gain.clone().requires_grad_(),
                "paired_sequential_gain_raw": d.clone().requires_grad_(),
                "paired_coverage_correction": coverage.clone().requires_grad_(),
            }
            if enabled
            else {}
        )
        state = torch.get_rng_state().clone()
        result = _opd_train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            rollout,
            rollout.valid_mask.float(),
            reference,
            config,
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
            gibbs_scores=corrected,
            gibbs_epsilon=0.5,
            gibbs_mode="direct_bounded_gibbs",
            **extra,
        )
        assert torch.equal(state, torch.get_rng_state())
        assert all(value.grad is None for value in extra.values())
        results.append(result)
        parameters.append(_parameters(model))
        forward_counts.append(model.forward_calls)
    assert torch.equal(parameters[0], parameters[1])
    assert forward_counts[0] == forward_counts[1]
    assert results[0]["loss"] == results[1]["loss"]
    assert torch.equal(
        results[0]["allocated_position_weights"],
        results[1]["allocated_position_weights"],
    )
    assert len(allocated_inputs) == 3  # PROW, then PROW + diagnostic G-only
    order = torch.arange(len(gain))
    if micro_batch > 1:
        order = torch.argsort(rollout.valid_mask.sum(-1), descending=True, stable=True)
    assert torch.equal(allocated_inputs[-1][0], gain[order][rollout.valid_mask[order]])
    assert allocated_inputs[-1][1:] == allocated_inputs[-2][1:]
    assert torch.equal(rollout.input_ids, initial_ids) and torch.equal(
        corrected, initial_scores
    )
    assert "paired_weight_diagnostics" not in results[0]
    report = results[1]["paired_weight_diagnostics"]
    assert report["zero_score_statistics"]["valid_token_count"] == int(
        rollout.valid_mask.sum()
    )


def test_metrics_csv_tensorboard_and_preupdate_measurement(monkeypatch, tmp_path):
    import b200_experiment.trainer as trainer
    from b200_experiment.tensorboard_logging import production_tensorboard_metrics
    from test_tensorboard_logging import _cmt_metrics

    torch.manual_seed(73)
    model = _TinyCausalLM()
    rollout = batch()
    reference = reference_for(model, rollout)
    gain, d, coverage = paired_inputs(rollout)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = config_for_control(True)
    config["selector"]["cmt_paired_weight_diagnostics"] = True
    stepped = []
    original_step, original_measure = optimizer.step, trainer.measure_paired_weights

    def measure_before_update(*args, **kwargs):
        assert not stepped
        return original_measure(*args, **kwargs)

    def step(*args, **kwargs):
        stepped.append(True)
        return original_step(*args, **kwargs)

    optimizer.step = step
    monkeypatch.setattr(trainer, "measure_paired_weights", measure_before_update)
    logger = PairedWeightDiagnosticsLogger(tmp_path, enabled=True)
    result = _opd_train_step(
        model,
        optimizer,
        rollout,
        rollout.valid_mask.float(),
        reference,
        config,
        torch.device("cpu"),
        DistributedContext(0, 0, 1, torch.device("cpu")),
        gibbs_scores=gain + d,
        gibbs_epsilon=0.5,
        gibbs_mode="direct_bounded_gibbs",
        paired_local_gain=gain,
        paired_sequential_gain_raw=d,
        paired_coverage_correction=coverage,
        on_paired_weight_diagnostics=logger.write,
    )
    logger.finalize()
    event = {"step": 1, "method": "cmt", **{key: result[key] for key in PAIRED_FIELDS}}
    _append_train_metrics_csv(tmp_path / "train_metrics.csv", event)
    csv_row = list(csv.DictReader((tmp_path / "train_metrics.csv").open()))[0]
    assert int(csv_row["paired_valid_token_count"]) == int(rollout.valid_mask.sum())
    tb_input = _cmt_metrics()
    tb_input.update(event)
    tags = production_tensorboard_metrics(tb_input, "cmt")
    assert (
        tags["cmt/paired_weights/zero_prospective_count"]
        == result["paired_zero_prospective_count"]
    )
    assert (
        tags["cmt/paired_weights/prow_realized_kl"] == result["paired_prow_realized_kl"]
    )


def _distributed_worker(rank, size, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=size
    )
    try:
        torch.manual_seed(55)
        model = _TinyCausalLM()
        wrapped = DistributedDataParallel(model)
        counts = (5, 4) if size == 2 else (3, 3, 2, 2)
        rollout = batch(rank, counts[rank])
        reference = reference_for(model, rollout)
        gain, d, coverage = paired_inputs(rollout, rank)
        config = config_for_control(True, 5 if size == 2 else 6)
        config["selector"]["cmt_paired_weight_diagnostics"] = True
        context = DistributedContext(rank, rank, size, torch.device("cpu"))
        callbacks = []
        result = _opd_train_step(
            wrapped,
            torch.optim.SGD(model.parameters(), lr=0.001),
            rollout,
            rollout.valid_mask.float(),
            reference,
            config,
            torch.device("cpu"),
            context,
            trajectory_group_ids=[0] * counts[rank],
            gibbs_scores=gain + d.tanh(),
            gibbs_epsilon=0.5,
            gibbs_mode="direct_bounded_gibbs",
            paired_local_gain=gain,
            paired_sequential_gain_raw=d,
            paired_coverage_correction=coverage,
            on_paired_weight_diagnostics=lambda *args, **kwargs: callbacks.append(
                args[0]
            ),
        )
        assert callbacks == ([1, 2] if rank == 0 else [])
        reports = []
        for index, entry in enumerate(result["minibatches"]):
            active = rollout.valid_mask & result["allocated_group_indices"].eq(index)
            G, _, _, _ = context.all_gather_variable_1d(gain[active])
            D, _, _, _ = context.all_gather_variable_1d(d[active])
            C, _, _, _ = context.all_gather_variable_1d(coverage[active])
            W, _, _, _ = context.all_gather_variable_1d(
                result["allocated_position_weights"][active]
            )
            _, prow, _, _ = cmt_allocation(
                G + D.tanh(), 0.5, mode="direct_bounded_gibbs", final_epsilon=0.02
            )
            _, g_only, _, _ = cmt_allocation(
                G, 0.5, mode="direct_bounded_gibbs", final_epsilon=0.02
            )
            expected, _ = measure_paired_weights(
                G,
                D,
                C,
                prow,
                g_only,
                W,
                weight_min=0.5,
                weight_max=2,
                allocation_mode="direct_bounded_gibbs",
            )
            report = entry["paired_weight_diagnostics"]
            for key in [
                "zero_score_statistics",
                "paired_comparison",
                "allocations",
                "checks",
            ]:
                assert report[key] == expected[key]
            reports.append(report)
        torch.save(
            {"parameters": _parameters(model), "reports": reports},
            Path(destination) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_global_group_diagnostics_with_uneven_ranks_and_dummy_padding(tmp_path, size):
    mp.spawn(
        _distributed_worker,
        args=(size, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=size,
        join=True,
    )
    records = [
        torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for rank in range(size)
    ]
    assert all(records[0]["reports"] == record["reports"] for record in records[1:])
    assert all(
        torch.equal(records[0]["parameters"], record["parameters"])
        for record in records[1:]
    )


def test_existing_launcher_one_gpu_random_enables_paired_flag_and_keeps_final_eval(
    tmp_path,
):
    from b200_experiment.cli import _configured, build_parser

    root = Path(__file__).resolve().parents[1]
    capture = tmp_path / "commands.jsonl"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        f"real={sys.executable!r}\n"
        "if sys.argv[1] == '-': os.execv(real,[real,*sys.argv[1:]])\n"
        "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
    )
    executable.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": str(executable),
        "CAPTURE": str(capture),
        "CUDA_VISIBLE_DEVICES": "0",
        "TRAIN_NPROC_PER_NODE": "1",
        "CMT_PAIRED_WEIGHT_DIAGNOSTICS": "true",
        "RUN_PREFIX": "unit_paired",
        "STORAGE_ROOT": "/workspace/storage-shared",
    }
    process = subprocess.run(
        [
            "bash",
            str(root / "scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh"),
            "random",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(commands) == 2
    assert commands[0][:2] == ["-m", "b200_experiment.cli"]
    config = _configured(
        build_parser().parse_args(commands[0][commands[0].index("train") :])
    )
    assert config["selector"]["cmt_random_weight_control"] is True
    assert config["selector"]["cmt_paired_weight_diagnostics"] is True
    assert config["training"]["epochs"] == 1 and config["training"]["max_steps"] is None
    assert not config["training_evaluation"]["enabled"]
    assert (
        config["models"]["teacher_path"] == "/workspace/storage-shared/models/Qwen3-4B"
    )
    assert (
        config["models"]["student_path"]
        == "/workspace/storage-shared/models/Qwen3-1.7B"
    )
    assert "--fresh-seeds" in commands[1]
    assert commands[1][
        commands[1].index("--seeds") + 1 : commands[1].index("--seeds") + 4
    ] == ["42", "43", "44"]
