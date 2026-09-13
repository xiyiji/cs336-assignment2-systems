"""Extra distributed tests: all DDP variants, FSDP at world size 4 with
uneven shapes, and sharded-optimizer param groups. Everything runs on CPU
with gloo, like the official tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cs336_systems.ddp import DDP_VARIANTS
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

from .common import FIXTURES_PATH, ToyModel, ToyModelWithTiedWeights, _cleanup_process_group, _setup_process_group


# --------------------------------------------------------------------------- #
# DDP variants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["naive", "flat", "bucketed", "overlap"])
@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_ddp_variants_match_non_parallel(variant, model_class):
    world_size = 2
    mp.spawn(_test_ddp_variant, args=(world_size, variant, model_class), nprocs=world_size, join=True)


def _test_ddp_variant(rank, world_size, variant, model_class):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()
    torch.manual_seed(rank)
    non_parallel = model_class().to(device)
    kwargs = {"bucket_size_mb": 0.001} if variant == "bucketed" else {}  # tiny buckets → many buckets
    ddp_model = DDP_VARIANTS[variant](deepcopy(non_parallel), **kwargs)

    # rank 0's weights were broadcast to everyone
    for p in ddp_model.module.parameters():
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)
        for g in gathered:
            assert torch.equal(g, gathered[0])
    if rank != 0:
        non_parallel.load_state_dict(ddp_model.module.state_dict())

    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")
    local_bs = all_x.size(0) // world_size
    loss_fn = nn.MSELoss()
    opt_ddp = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    opt_ref = torch.optim.SGD(non_parallel.parameters(), lr=0.1)

    for i in range(4):
        opt_ddp.zero_grad()
        opt_ref.zero_grad()
        loss_fn(non_parallel(all_x), all_y).backward()
        opt_ref.step()

        off = rank * local_bs
        loss_fn(ddp_model(all_x[off : off + local_bs]), all_y[off : off + local_bs]).backward()
        ddp_model.finish_gradient_synchronization()
        opt_ddp.step()

        for p_ref, p_ddp in zip(non_parallel.parameters(), ddp_model.parameters()):
            assert torch.allclose(p_ref, p_ddp, atol=1e-6), f"{variant}: mismatch at step {i}"
        torch.manual_seed(42 + i)
        perm = torch.randperm(all_x.size(0))
        all_x, all_y = all_x[perm], all_y[perm]
    _cleanup_process_group()


# --------------------------------------------------------------------------- #
# FSDP: world size 4, uneven parameter sizes, torch.nn layers, repeated steps
# --------------------------------------------------------------------------- #
class UnevenModel(nn.Module):
    """Parameter counts that are *not* multiples of the world size."""

    def __init__(self):
        super().__init__()
        from cs336_basics.model import Embedding, Linear, RMSNorm

        self.emb = Embedding(37, 30)
        self.norm = RMSNorm(30)
        self.lin1 = Linear(30, 45)
        self.lin2 = nn.Linear(45, 22, bias=True)  # torch.nn layer with bias (bias stays replicated)
        self.head = Linear(22, 37)

    def forward(self, x):
        x = self.norm(self.emb(x))
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return self.head(x)


@pytest.mark.parametrize("world_size", [2, 4])
def test_fsdp_uneven_shapes(world_size):
    mp.spawn(_test_fsdp_uneven, args=(world_size,), nprocs=world_size, join=True)


def _test_fsdp_uneven(rank, world_size):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()
    torch.manual_seed(0)
    base = UnevenModel().to(device)
    ref = deepcopy(base)
    fsdp = FSDP(deepcopy(base))

    assert set(fsdp.sharded_parameter_names()) == {"emb.weight", "lin1.weight", "lin2.weight", "head.weight"}
    n_local = sum(p.numel() for p in fsdp.parameters())
    n_full = sum(p.numel() for p in ref.parameters())
    assert n_local < n_full, "parameters were not sharded"

    opt_f = torch.optim.AdamW(fsdp.parameters(), lr=1e-2)
    opt_r = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    torch.manual_seed(7)
    batch = 4 * world_size
    ids = torch.randint(0, 37, (batch, 6))
    for step in range(4):
        opt_f.zero_grad(set_to_none=True)
        opt_r.zero_grad(set_to_none=True)
        ref(ids).float().pow(2).mean().backward()
        opt_r.step()
        local = ids[rank * 4 : (rank + 1) * 4]
        # the global mean over `batch` rows equals the average of per-rank means
        fsdp(local).float().pow(2).mean().backward()
        fsdp.finish_gradient_synchronization()
        opt_f.step()
        full = fsdp.gather_full_params()
        for name, p in ref.named_parameters():
            assert full[name].shape == p.shape
            assert torch.allclose(p, full[name], atol=1e-5, rtol=1e-4), f"{name} differs at step {step}"
    _cleanup_process_group()


# --------------------------------------------------------------------------- #
# Sharded optimizer with parameter groups and add_param_group
# --------------------------------------------------------------------------- #
def test_sharded_optimizer_param_groups():
    world_size = 2
    mp.spawn(_test_sharded_groups, args=(world_size,), nprocs=world_size, join=True)


def _test_sharded_groups(rank, world_size):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(3)
    ref_model = ToyModel().to(device)
    sh_model = deepcopy(ref_model)

    def groups(m):
        return [
            {"params": [m.fc1.weight], "lr": 0.05},
            {"params": [m.fc2.fc.weight, m.fc2.fc.bias], "weight_decay": 0.0},
        ]

    ref_opt = torch.optim.AdamW(groups(ref_model), lr=0.1, weight_decay=0.1)
    sh_opt = ShardedOptimizer(groups(sh_model), torch.optim.AdamW, lr=0.1, weight_decay=0.1)
    ref_opt.add_param_group({"params": [ref_model.fc3.weight]})
    sh_opt.add_param_group({"params": [sh_model.fc3.weight]})
    assert len(sh_opt.param_groups) == 3
    owners = {sh_opt.owner_of(p) for g in sh_opt.param_groups for p in g["params"]}
    assert owners == {0, 1}, "parameters should be spread over both ranks"

    for _ in range(5):
        x = torch.rand(16, 10)
        y = torch.rand(16, 10)
        for model, opt in ((ref_model, ref_opt), (sh_model, sh_opt)):
            opt.zero_grad()
            ((model(x) - y) ** 2).sum().backward()
            opt.step()
    for p_ref, p_sh in zip(ref_model.parameters(), sh_model.parameters()):
        assert torch.allclose(p_ref, p_sh, atol=1e-7)
    # State is sharded: each rank holds a strict subset of the AdamW state.
    local_states = sum(len(sh_opt.local_optimizer.state) for _ in [0])
    total = torch.tensor([local_states])
    dist.all_reduce(total)
    assert local_states < total.item()
    _cleanup_process_group()


# --------------------------------------------------------------------------- #
# FSDP + fused (chunked) LM-head loss: the head weight is sharded like any layer
# --------------------------------------------------------------------------- #
def test_fsdp_with_fused_lm_head_loss():
    mp.spawn(_test_fsdp_fused_head, args=(2,), nprocs=2, join=True)


def _test_fsdp_fused_head(rank, world_size):
    from cs336_basics.nn_utils import cross_entropy
    from cs336_systems.fused_ce import FusedLMHeadLoss

    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()
    torch.manual_seed(0)
    base = UnevenModel().to(device)
    ref = deepcopy(base)
    fused = deepcopy(base)
    fused.head = FusedLMHeadLoss(fused.head.weight, chunk_size=5)
    fsdp = FSDP(fused)
    assert "head.weight" in fsdp.sharded_parameter_names()

    opt_f = torch.optim.SGD(fsdp.parameters(), lr=0.1)
    opt_r = torch.optim.SGD(ref.parameters(), lr=0.1)
    torch.manual_seed(1)
    ids = torch.randint(0, 37, (8, 6))
    tgt = torch.randint(0, 37, (8, 6))
    for step in range(3):
        opt_f.zero_grad(set_to_none=True)
        opt_r.zero_grad(set_to_none=True)
        cross_entropy(ref(ids), tgt).backward()
        opt_r.step()
        sl = slice(rank * 4, (rank + 1) * 4)
        m = fsdp.module
        h = m.lin2(torch.relu(m.lin1(m.norm(m.emb(ids[sl])))))
        m.head(h, tgt[sl]).backward()
        fsdp.finish_gradient_synchronization()
        opt_f.step()
        full = fsdp.gather_full_params()
        for name, p in ref.named_parameters():
            assert torch.allclose(p, full[name], atol=1e-5, rtol=1e-4), f"{name} differs at step {step}"
    _cleanup_process_group()
