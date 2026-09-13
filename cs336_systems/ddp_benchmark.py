"""Benchmark distributed training strategies on the basics Transformer
(handout ``naive_ddp_benchmarking``, ``minimal_ddp_flat_benchmarking``,
``ddp_overlap_individual_parameters_benchmarking``, ``optimizer_state_sharding_accounting``,
``fsdp_accounting``).

    # 1 node x 2 GPUs, xl model, every DDP flavour
    uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy naive flat overlap bucketed
    # optimizer-state sharding memory accounting
    uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy overlap --sharded-optimizer
    # FSDP
    uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy fsdp --dtype bf16
    # CPU smoke run
    uv run python -m cs336_systems.ddp_benchmark --size tiny --world-size 2 --backend gloo --steps 3

Per step we report the total time, the time spent inside gradient
communication (``finish_gradient_synchronization`` for the after-backward
variants; for the overlapped variants only the *exposed* wait time), and the
memory (CUDA: allocated bytes at three points; CPU: analytic bytes for
parameters, gradients and optimizer state).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import timeit
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.config import MODEL_SIZES, get_model_kwargs
from cs336_systems.ddp import DDP_VARIANTS
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _mem(device) -> float:
    return torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")


def _peak(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")


def _worker(rank, args, port, shared):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    backend = args.backend
    if backend == "nccl":
        local = rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=args.world_size)
    torch.manual_seed(0)

    model = basics_model.BasicsTransformerLM(**get_model_kwargs(args.size, args.context_length)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.strategy == "fsdp":
        wrapped = FSDP(model, compute_dtype=None if args.dtype == "fp32" else DTYPES[args.dtype])
        autocast = nullcontext()
    else:
        wrapped = DDP_VARIANTS[args.strategy](model, **({"bucket_size_mb": args.bucket_size_mb} if args.strategy == "bucketed" else {}))
        autocast = torch.autocast(device.type, dtype=DTYPES[args.dtype]) if args.dtype != "fp32" else nullcontext()
    if args.compile:
        wrapped.module = torch.compile(wrapped.module)

    mem_after_init = _mem(device)
    if args.sharded_optimizer:
        optimizer = ShardedOptimizer(wrapped.parameters(), AdamW, lr=1e-3)
    else:
        optimizer = AdamW(wrapped.parameters(), lr=1e-3)

    local_bs = args.batch_size // args.world_size
    x = torch.randint(0, 10_000, (local_bs, args.context_length), device=device)
    y = torch.randint(0, 10_000, (local_bs, args.context_length), device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step_times, comm_times, mem_before_opt, mem_after_opt = [], [], [], []
    for step in range(args.warmup + args.steps):
        sync()
        t0 = timeit.default_timer()
        with autocast:
            loss = cross_entropy(wrapped(x), y)
        loss.backward()
        sync()
        t1 = timeit.default_timer()
        wrapped.finish_gradient_synchronization()
        sync()
        t2 = timeit.default_timer()
        if step >= args.warmup:
            mem_before_opt.append(_peak(device))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()
        t3 = timeit.default_timer()
        if step >= args.warmup:
            step_times.append(t3 - t0)
            comm_times.append(t2 - t1)
            mem_after_opt.append(_peak(device))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    gathered = [None] * args.world_size
    local = {
        "step_ms": 1000 * statistics.fmean(step_times),
        "step_std_ms": 1000 * statistics.pstdev(step_times),
        "comm_ms": 1000 * statistics.fmean(comm_times),
        "mem_after_init_mib": mem_after_init,
        "peak_before_opt_mib": statistics.fmean(mem_before_opt),
        "peak_after_opt_mib": statistics.fmean(mem_after_opt),
        "param_bytes_local": sum(p.numel() * p.element_size() for p in wrapped.parameters()),
        "opt_state_bytes_local": (
            optimizer.local_state_bytes()
            if isinstance(optimizer, ShardedOptimizer)
            else sum(v.numel() * v.element_size() for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v))
        ),
    }
    dist.all_gather_object(gathered, local)
    if rank == 0:
        agg = {k: statistics.fmean(g[k] for g in gathered) for k in local}
        agg.update(
            size=args.size, n_params=n_params, strategy=args.strategy, world_size=args.world_size, backend=backend,
            dtype=args.dtype, sharded_optimizer=args.sharded_optimizer, context_length=args.context_length,
            batch_size=args.batch_size, compiled=args.compile,
        )
        shared.update(agg)
    dist.barrier()
    dist.destroy_process_group()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="xl", choices=sorted(MODEL_SIZES))
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4, help="global batch size (split across ranks)")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", default="nccl" if torch.cuda.is_available() else "gloo", choices=["gloo", "nccl"])
    p.add_argument("--strategy", nargs="+", default=["overlap"], choices=[*DDP_VARIANTS, "fsdp"])
    p.add_argument("--bucket-size-mb", type=float, default=25.0)
    p.add_argument("--sharded-optimizer", action="store_true")
    p.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--port", type=int, default=29533)
    p.add_argument("--json", default=None, help="append one JSON line per strategy")
    args = p.parse_args(argv)

    results = []
    for strategy in list(args.strategy):
        run_args = argparse.Namespace(**vars(args))
        run_args.strategy = strategy
        manager = mp.Manager()
        shared = manager.dict()
        mp.spawn(_worker, args=(run_args, args.port, shared), nprocs=args.world_size, join=True)
        r = dict(shared)
        results.append(r)
        print(
            f"{strategy:9s} ws={r['world_size']} {r['size']} {r['dtype']}{' +sharded-opt' if r['sharded_optimizer'] else ''}: "
            f"step={r['step_ms']:.1f}±{r['step_std_ms']:.1f} ms, exposed comm={r['comm_ms']:.1f} ms "
            f"({100 * r['comm_ms'] / r['step_ms']:.1f}%), local params={r['param_bytes_local'] / 2**20:.0f} MiB, "
            f"local opt state={r['opt_state_bytes_local'] / 2**20:.0f} MiB, "
            f"peak before/after opt={r['peak_before_opt_mib']:.0f}/{r['peak_after_opt_mib']:.0f} MiB",
            flush=True,
        )
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "a") as f:
                f.write(json.dumps(r) + "\n")
    return results


if __name__ == "__main__":
    main()
