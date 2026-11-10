import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from b200_experiment.cmt_random_control import (
    permute_final_cmt_weights,
    permutation_seed,
    random_weight_control_enabled,
)
from b200_experiment.config import load_config, save_config
from b200_experiment.distributed import DistributedContext
from b200_experiment.opd_core import topk_reference_from_logits
from b200_experiment.resume import validate_resume_config
from b200_experiment.scoring import RolloutBatch
from b200_experiment.selectors.cmt_selector import cmt_allocation
from b200_experiment.trainer import _opd_train_step
from test_distributed import _TinyCausalLM, _tiny_config, _parameters


def moments(weights):
    values = weights.detach().double().cpu().tolist()
    total = math.fsum(values)
    return {
        "mean": total / len(values),
        "kl": math.fsum(w * math.log(w) if w else 0 for w in values) / len(values),
        "ess": total**2 / math.fsum(w * w for w in values),
    }


@pytest.mark.parametrize("mode", ["gibbs", "bounded_gibbs", "direct_bounded_gibbs"])
def test_exact_multiset_mean_kl_ess_clipping_and_raw_pair_invariance(mode):
    scores = torch.linspace(-5, 5, 128).square() + torch.linspace(0, 0.7, 128)
    raw, final, _, _ = cmt_allocation(
        scores, 0.5, mode=mode, weight_min=0.5, weight_max=2, final_epsilon=0.02
    )
    raw_before, final_before = raw.clone(), final.clone()
    changed_raw, changed_final, stats = permute_final_cmt_weights(
        raw, final, experiment_seed=42, optimizer_step=17
    )
    assert torch.equal(torch.sort(changed_final).values, torch.sort(final).values)
    assert torch.equal(torch.sort(changed_raw).values, torch.sort(raw).values)
    assert moments(changed_final) == moments(final)
    assert abs(moments(changed_final)["mean"] - 1) < 1e-6
    assert float(changed_final.min()) == float(final.min())
    assert float(changed_final.max()) == float(final.max())
    for boundary in [0.5, 2.0]:
        assert int(changed_final.eq(boundary).sum()) == int(final.eq(boundary).sum())
    assert raw.dtype == changed_raw.dtype and final.dtype == changed_final.dtype
    assert torch.equal(raw, raw_before) and torch.equal(final, final_before)
    generator = torch.Generator(device="cpu").manual_seed(stats["cmt_permutation_seed"])
    expected = torch.randperm(len(final), generator=generator)
    assert torch.equal(changed_raw, raw[expected])
    assert torch.equal(changed_final, final[expected])
    assert (
        stats["cmt_permutation_sha256"]
        == hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    )
    assert stats["cmt_permutation_moved_fraction"] > 0.9


def test_reproducibility_step_variation_and_global_rng_states_untouched():
    raw, final = torch.arange(128).float(), torch.linspace(0.5, 1.5, 128)
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    first = permute_final_cmt_weights(raw, final, experiment_seed=42, optimizer_step=1)
    again = permute_final_cmt_weights(raw, final, experiment_seed=42, optimizer_step=1)
    next_step = permute_final_cmt_weights(
        raw, final, experiment_seed=42, optimizer_step=2
    )
    other_seed = permute_final_cmt_weights(
        raw, final, experiment_seed=43, optimizer_step=1
    )
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    assert first[2]["cmt_permutation_seed"] == again[2]["cmt_permutation_seed"]
    assert not torch.equal(first[1], next_step[1])
    assert not torch.equal(first[1], other_seed[1])
    assert torch.equal(cpu_state, torch.get_rng_state())
    if cuda_states:
        assert all(
            torch.equal(old, new)
            for old, new in zip(cuda_states, torch.cuda.get_rng_state_all())
        )


@pytest.mark.parametrize("count", [1, 64])
def test_uniform_weights_remain_uniform_and_report_zero_changed_values(count):
    weights = torch.ones(count)
    raw, final, stats = permute_final_cmt_weights(
        weights, weights, experiment_seed=42, optimizer_step=1
    )
    assert torch.equal(raw, weights) and torch.equal(final, weights)
    assert stats["cmt_permutation_changed_weight_fraction"] == 0
    assert stats["cmt_permutation_changed_raw_weight_fraction"] == 0


def test_active_clipping_masses_and_exact_realized_kl_remain_unchanged():
    final = torch.tensor([0.5, 0.5, 1.0, 2.0] * 8)
    raw = torch.linspace(0.1, 1.9, 32)
    shuffled_raw, shuffled, _ = permute_final_cmt_weights(
        raw, final, experiment_seed=42, optimizer_step=9
    )
    assert int(shuffled.eq(0.5).sum()) == 16
    assert int(shuffled.eq(2.0).sum()) == 8
    assert moments(shuffled) == moments(final)
    assert moments(shuffled)["mean"] == 1.0
    assert torch.equal(torch.sort(shuffled_raw).values, torch.sort(raw).values)


def batch(rank=0, rows=4):
    ids = torch.tensor([[1 + (rank + row) % 7, 2, 3, 4, 5] for row in range(rows)])
    valid = torch.arange(3).unsqueeze(0) < torch.tensor(
        [1 + (rank + i) % 3 for i in range(rows)]
    ).unsqueeze(1)
    attention = torch.cat([torch.ones(rows, 2, dtype=torch.long), valid.long()], dim=1)
    return RolloutBatch(
        ids,
        attention,
        ids[:, 2:],
        valid,
        torch.zeros_like(valid, dtype=torch.float32),
        2,
    )


def reference_for(model, rollout):
    with torch.no_grad():
        logits = model(rollout.input_ids, rollout.attention_mask).logits[:, 1:-1]
        return topk_reference_from_logits(
            logits, logits + torch.linspace(-0.2, 0.2, 11), rollout.valid_mask, top_k=5
        )


def config_for_control(enabled=None, ppo_size=4):
    config = _tiny_config(1, ppo_size)
    config["experiment"] = {"method": "cmt", "seed": 42}
    if enabled is not None:
        config["selector"]["cmt_random_weight_control"] = enabled
    return config


def test_disabled_default_and_false_keep_original_allocation_and_optimizer_bitwise(
    monkeypatch,
):
    import b200_experiment.trainer as trainer

    torch.manual_seed(123)
    initial = _TinyCausalLM()
    rollout = batch()
    reference = reference_for(initial, rollout)
    scores = torch.arange(12).reshape(4, 3).float()
    saved_scores, saved_ids = scores.clone(), rollout.input_ids.clone()
    base_raw, base_final, _, _ = cmt_allocation(
        scores[rollout.valid_mask], 0.5, mode="direct_bounded_gibbs", final_epsilon=0.02
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled CMT must never create a permutation")

    monkeypatch.setattr(trainer, "permute_final_cmt_weights", forbidden)
    outputs = []
    for enabled in [None, False]:
        model = copy.deepcopy(initial)
        result = _opd_train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            rollout,
            rollout.valid_mask.float(),
            reference,
            config_for_control(enabled),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
            gibbs_scores=scores,
            gibbs_epsilon=0.5,
            gibbs_mode="direct_bounded_gibbs",
        )
        assert torch.equal(
            result["allocated_position_weights"][rollout.valid_mask], base_final
        )
        assert torch.equal(
            result["allocated_raw_position_weights"][rollout.valid_mask], base_raw
        )
        assert result["cmt_random_weight_control"] is False
        assert result["cmt_permutation_seed"] is None
        outputs.append((_parameters(model), result))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1]["loss"] == outputs[1][1]["loss"]
    assert torch.equal(scores, saved_scores) and torch.equal(
        rollout.input_ids, saved_ids
    )


def test_enabled_control_matches_manual_final_weight_reassignment_without_changing_objective():
    torch.manual_seed(333)
    initial = _TinyCausalLM()
    rollout = batch()
    reference = reference_for(initial, rollout)
    scores = torch.arange(12).reshape(4, 3).float()
    raw, final, _, _ = cmt_allocation(
        scores[rollout.valid_mask], 0.5, mode="direct_bounded_gibbs", final_epsilon=0.02
    )
    _, shuffled, _ = permute_final_cmt_weights(
        raw, final, experiment_seed=42, optimizer_step=101
    )
    manual_weights = torch.zeros_like(scores)
    manual_weights[rollout.valid_mask] = shuffled
    models, results = [], []
    for controlled in [True, False]:
        model = copy.deepcopy(initial)
        kwargs = (
            {
                "gibbs_scores": scores,
                "gibbs_epsilon": 0.5,
                "gibbs_mode": "direct_bounded_gibbs",
            }
            if controlled
            else {}
        )
        result = _opd_train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            rollout,
            manual_weights,
            reference,
            config_for_control(controlled),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
            optimizer_step_start=100,
            **kwargs,
        )
        models.append(_parameters(model))
        results.append(result)
    assert torch.equal(models[0], models[1])
    assert results[0]["loss"] == results[1]["loss"]
    assert results[0]["optimizer_step"] == 101
    assert torch.equal(results[0]["allocated_position_weights"], manual_weights)
    assert results[0]["cmt_permutation_seed"] == permutation_seed(42, 101)


def _distributed_control_worker(rank, world_size, rendezvous, destination, uneven):
    import b200_experiment.trainer as trainer

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(321)
        model = _TinyCausalLM()
        wrapped = DistributedDataParallel(model)
        counts = (
            ((5, 4) if world_size == 2 else (3, 3, 2, 2))
            if uneven
            else (8,) * world_size
        )
        rollout = batch(rank, rows=counts[rank])
        reference = reference_for(model, rollout)
        scores = (
            torch.arange(counts[rank] * 3).reshape(counts[rank], 3).float() + rank * 24
        )
        original_scores, original_tokens = scores.clone(), rollout.input_ids.clone()
        originals = []
        allocate = trainer.cmt_allocation

        def capture(*args, **kwargs):
            result = allocate(*args, **kwargs)
            originals.append((result[0].clone(), result[1].clone(), dict(result[3])))
            return result

        trainer.cmt_allocation = capture
        context = DistributedContext(rank, rank, world_size, torch.device("cpu"))
        rng_state = torch.get_rng_state().clone()
        result = _opd_train_step(
            wrapped,
            torch.optim.SGD(model.parameters(), lr=0.001),
            rollout,
            rollout.valid_mask.float(),
            reference,
            config_for_control(
                True, (5 if world_size == 2 else 6) if uneven else world_size * 2
            ),
            torch.device("cpu"),
            context,
            trajectory_group_ids=[0] * counts[rank] if uneven else [0, 1, 2, 3] * 2,
            gibbs_scores=scores,
            gibbs_epsilon=0.5,
            gibbs_mode="direct_bounded_gibbs",
            optimizer_step_start=30,
        )
        assert torch.equal(rng_state, torch.get_rng_state())
        updates = 2 if uneven else 4
        assert result["optimizer_steps"] == result["gibbs_allocations"] == updates
        assert torch.equal(scores, original_scores) and torch.equal(
            rollout.input_ids, original_tokens
        )
        assert not result["allocated_position_weights"][~rollout.valid_mask].any()
        fingerprints = []
        ragged_groups = 0
        for group, (raw, final, allocation_metrics) in enumerate(originals):
            active = rollout.valid_mask & result["allocated_group_indices"].eq(group)
            actual, _, _, lengths = context.all_gather_variable_1d(
                result["allocated_position_weights"][active]
            )
            actual_raw, _, _, _ = context.all_gather_variable_1d(
                result["allocated_raw_position_weights"][active]
            )
            ragged_groups += len(set(lengths)) > 1
            metric = result["minibatches"][group]
            seed = permutation_seed(42, 31 + group)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            permutation = torch.randperm(len(final), generator=generator)
            assert torch.equal(actual, final[permutation])
            assert torch.equal(actual_raw, raw[permutation])
            assert moments(actual) == moments(final)
            assert abs(metric["global_weight_mass"] - len(actual)) < 1e-4
            assert metric["group_valid_token_count"] == len(actual)
            assert (
                metric["allocation_kl_final"]
                == allocation_metrics["allocation_kl_final"]
            )
            assert metric["cmt_permutation_seed"] == seed
            fingerprints.append(metric["cmt_permutation_sha256"])
        assert ragged_groups > 0
        assert len(set(fingerprints)) == updates
        torch.save(
            {"parameters": _parameters(model), "fingerprints": fingerprints},
            Path(destination) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("uneven", [False, True])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_global_permutation_before_scatter_and_distributed_partitioning(
    tmp_path, world_size, uneven
):
    mp.spawn(
        _distributed_control_worker,
        args=(world_size, str(tmp_path / "rendezvous"), str(tmp_path), uneven),
        nprocs=world_size,
        join=True,
    )
    rows = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]
    assert all(
        torch.equal(rows[0]["parameters"], row["parameters"]) for row in rows[1:]
    )
    assert all(rows[0]["fingerprints"] == row["fingerprints"] for row in rows[1:])


def test_resume_preserves_control_and_missing_flag_means_original(tmp_path):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/qwen3_b200_cmt.yaml"
    )
    checkpoint = tmp_path / "checkpoint-000150"
    checkpoint.mkdir()
    original = copy.deepcopy(config)
    original["selector"].pop("cmt_random_weight_control")
    save_config(original, tmp_path / "resolved_config.yaml")
    assert not validate_resume_config(checkpoint, config)["mismatches"]
    config["selector"]["cmt_random_weight_control"] = True
    with pytest.raises(ValueError, match="cmt_random_weight_control"):
        validate_resume_config(checkpoint, config)
    save_config(config, tmp_path / "resolved_config.yaml")
    assert not validate_resume_config(checkpoint, config)["mismatches"]


def test_flag_boolean_and_default():
    assert random_weight_control_enabled({}) is False
    assert (
        random_weight_control_enabled({"selector": {"cmt_random_weight_control": True}})
        is True
    )
    with pytest.raises(ValueError, match="boolean"):
        random_weight_control_enabled(
            {"selector": {"cmt_random_weight_control": "false"}}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optional")
def test_gpu_weights_use_same_cpu_permutation_without_touching_cuda_rng():
    raw = torch.arange(128, device="cuda", dtype=torch.float32)
    final = torch.linspace(0.5, 1.5, 128, device="cuda")
    before = torch.cuda.get_rng_state().clone()
    cpu = permute_final_cmt_weights(
        raw.cpu(), final.cpu(), experiment_seed=42, optimizer_step=27
    )
    gpu = permute_final_cmt_weights(raw, final, experiment_seed=42, optimizer_step=27)
    assert gpu[0].device.type == gpu[1].device.type == "cuda"
    assert torch.equal(cpu[0], gpu[0].cpu()) and torch.equal(cpu[1], gpu[1].cpu())
    assert cpu[2]["cmt_permutation_sha256"] == gpu[2]["cmt_permutation_sha256"]
    assert torch.equal(before, torch.cuda.get_rng_state())


def capture_launcher(tmp_path, *, fail_training=False):
    root = Path(__file__).resolve().parents[1]
    capture = tmp_path / "commands.jsonl"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        f"real={sys.executable!r}\n"
        "if sys.argv[1] == '-': os.execv(real,[real,*sys.argv[1:]])\n"
        "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if os.environ.get('FAIL_TRAIN') == '1' and 'train' in sys.argv: sys.exit(11)\n"
    )
    executable.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": str(executable),
        "CAPTURE": str(capture),
        "CUDA_VISIBLE_DEVICES": "0,1",
        "RUN_PREFIX": "unit_controlled_cmt",
        "STORAGE_ROOT": "/workspace/storage-shared",
        "FAIL_TRAIN": "1" if fail_training else "0",
    }
    completed = subprocess.run(
        [
            "bash",
            str(root / "scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh"),
            "both",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed, [json.loads(line) for line in capture.read_text().splitlines()]


def test_real_launcher_both_arms_match_except_control_and_output_train_then_eval(
    tmp_path,
):
    from b200_experiment.cli import _configured, build_parser

    completed, commands = capture_launcher(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert len(commands) == 4
    assert "train" in commands[0] and "multiseed_evaluation" in commands[1][1]
    assert "train" in commands[2] and "multiseed_evaluation" in commands[3][1]
    configs = [
        _configured(build_parser().parse_args(command[command.index("train") :]))
        for command in commands[::2]
    ]
    for index, config in enumerate(configs):
        assert (
            config["experiment"]["method"] == "cmt"
            and config["experiment"]["seed"] == 42
        )
        assert config["selector"]["cmt_random_weight_control"] is bool(index)
        assert (
            config["models"]["student_path"]
            == "/workspace/storage-shared/models/Qwen3-1.7B"
        )
        assert (
            config["models"]["teacher_path"]
            == "/workspace/storage-shared/models/Qwen3-4B"
        )
        assert (
            config["data"]["path"]
            == "/workspace/storage-shared/nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet"
        )
        assert (
            config["training"]["epochs"] == 1
            and config["training"]["max_steps"] is None
        )
        assert config["training"]["resume_from_checkpoint"] is None
        assert not config["training_evaluation"]["enabled"]
        assert not config["training_evaluation"]["eval_at_start"]
        assert not config["training_evaluation"]["eval_at_end"]
        assert config["training_evaluation"]["num_responses"] == 8
        config["selector"].pop("cmt_random_weight_control")
    canonical = [
        json.dumps(config, sort_keys=True).replace(
            config["experiment"]["output_dir"], "SAME_OUTPUT"
        )
        for config in configs
    ]
    assert canonical[0] == canonical[1]
    for command in commands[1::2]:
        assert "--fresh-seeds" in command
        assert command[command.index("--checkpoints") + 1] == "final"
        assert command[command.index("--seeds") + 1 : command.index("--seeds") + 4] == [
            "42",
            "43",
            "44",
        ]


def test_failed_training_does_not_start_final_evaluation(tmp_path):
    completed, commands = capture_launcher(tmp_path, fail_training=True)
    assert completed.returncode == 11
    assert len(commands) == 1 and "train" in commands[0]


def test_existing_cli_enables_boolean_control_and_tensorboard_reports_it():
    from b200_experiment.cli import _configured, build_parser
    from b200_experiment.tensorboard_logging import production_tensorboard_metrics
    from test_tensorboard_logging import _cmt_metrics

    root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(root / "configs/qwen3_b200_cmt.yaml"),
            "--set",
            "selector.cmt_random_weight_control=true",
        ]
    )
    assert _configured(args)["selector"]["cmt_random_weight_control"] is True
    metrics = _cmt_metrics()
    metrics.update(
        cmt_random_weight_control=True,
        cmt_weight_mean=1.0,
        cmt_weight_clip_min=0.5,
        cmt_weight_clip_max=2.0,
        cmt_effective_sample_size=9.0,
        cmt_permutation_moved_fraction=0.9,
    )
    selected = production_tensorboard_metrics(metrics, "cmt")
    assert selected["cmt/random_control/random_weight_control"] == 1
    assert selected["cmt/random_control/weight_mean"] == 1
    assert selected["cmt/random_control/weight_clip_min"] == 0.5
    assert selected["cmt/random_control/weight_clip_max"] == 2
    assert selected["cmt/random_control/effective_sample_size"] == 9
