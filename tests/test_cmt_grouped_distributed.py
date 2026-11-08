"""Regression for ALFWorld CMT: 16 tasks, n=4, PPO=8 on 2/4 ranks."""

import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from b200_experiment.alfworld_rollout import collate_episodes
from b200_experiment.distributed import (
    DistributedContext,
    grouped_ppo_minibatch_partition,
)
from b200_experiment.opd_core import topk_reference_from_logits
from b200_experiment.scoring import rollout_forward_inputs, select_rollout_logits
from b200_experiment.trainer import _opd_train_step
from test_distributed import _TinyCausalLM, _tiny_config, _parameters


@pytest.mark.parametrize("world_size", [2, 4])
def test_small_grouped_ppo_reaches_every_rank_without_dropping_rows(world_size):
    groups = tuple((0, 1, 2, 3) * (16 // world_size) for _ in range(world_size))
    seen = []
    for index in range(8):
        for rank in range(world_size):
            rows, global_count = grouped_ppo_minibatch_partition(groups, rank, 8, index)
            assert global_count == 8
            assert len(rows) == 8 // world_size
            assert all(groups[rank][row] == index // 2 for row in rows)
            seen.extend((rank, row) for row in rows)
    expected = [(r, i) for r, rows in enumerate(groups) for i in range(len(rows))]
    assert sorted(seen) == expected
    assert len(seen) == len(set(seen)) == 64


def test_uneven_grouped_data_tail_is_a_complete_permutation():
    groups = ((0, 1, 2, 3) * 2, (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
    seen = []
    for index in range(math.ceil(20 / 8)):
        for rank in range(4):
            rows, count = grouped_ppo_minibatch_partition(groups, rank, 8, index)
            assert rows  # this tail can give every rank a real episode
            assert count == (8 if index < 2 else 4)
            seen.extend((rank, row) for row in rows)
    assert sorted(seen) == [
        (r, i) for r, rows in enumerate(groups) for i in range(len(rows))
    ]


def _grouped_cmt_worker(rank, world_size, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(123)
        model = _TinyCausalLM()
        wrapped = DistributedDataParallel(model)
        local_count = 64 // world_size
        batch = collate_episodes(
            [
                {
                    "input_ids": [1 + (rank * local_count + i) % 7, 2, 3, 7, 8, 4],
                    "actor_positions": [2, 5],
                    "trace": {},
                }
                for i in range(local_count)
            ],
            0,
            "cpu",
        )
        with torch.no_grad():
            kwargs, selection = rollout_forward_inputs(
                model, batch, torch.arange(local_count), 2
            )
            logits = select_rollout_logits(model(**kwargs).logits, selection)
            reference = topk_reference_from_logits(
                logits,
                logits + torch.linspace(-0.2, 0.2, 11),
                batch.valid_mask,
                top_k=5,
            )
        context = DistributedContext(rank, rank, world_size, torch.device("cpu"))
        scores = (
            torch.arange(local_count * 2, dtype=torch.float32).reshape(local_count, 2)
            + rank * local_count * 2
        )
        metrics = _opd_train_step(
            wrapped,
            torch.optim.SGD(model.parameters(), lr=0.001),
            batch,
            batch.valid_mask.float(),
            reference,
            _tiny_config(1, 8),
            torch.device("cpu"),
            context,
            trajectory_group_ids=[0, 1, 2, 3] * (local_count // 4),
            gibbs_scores=scores,
            gibbs_epsilon=0.02,
            gibbs_mode="direct_bounded_gibbs",
        )
        assert metrics["optimizer_steps"] == metrics["gibbs_allocations"] == 8
        for index, entry in enumerate(metrics["minibatches"]):
            assert entry["ppo_minibatch_trajectory_count"] == 8
            assert entry["local_ppo_minibatch_trajectory_count"] == 8 // world_size
            assert entry["response_index_composition"] == {str(index // 2): 8}
            assert abs(entry["global_weight_mass"] - 16) < 1e-4
        torch.save(_parameters(model), Path(destination) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
def test_real_distributed_cmt_multiturn_optimizer_with_small_ppo(tmp_path, world_size):
    mp.spawn(
        _grouped_cmt_worker,
        args=(world_size, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    first = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    for rank in range(1, world_size):
        torch.testing.assert_close(
            first, torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        )
