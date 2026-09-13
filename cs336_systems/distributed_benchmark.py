"""All-reduce benchmark, single node multi-process (handout §5.1, ``distributed_communication_single_node``).

    uv run python -m cs336_systems.distributed_benchmark --backend gloo --world-sizes 2 4 \
        --sizes-mb 1 10 100 1000 --csv results/allreduce_cpu.csv
    uv run python -m cs336_systems.distributed_benchmark --backend nccl --world-sizes 2 4 6 \
        --sizes-mb 1 10 100 1000 --csv results/allreduce_gpu.csv

Each configuration spawns ``world_size`` processes, runs ``--warmup`` un-timed
all-reduces, times ``--iters`` more (with ``torch.cuda.synchronize()`` for NCCL)
and gathers every rank's timings with ``all_gather_object`` so the reported
number is the mean over ranks.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world_size, backend, sizes_mb, warmup, iters, port, return_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    results = {}
    for size_mb in sizes_mb:
        numel = int(size_mb * 2**20 / 4)
        x = torch.randn(numel, device=device, dtype=torch.float32)
        for _ in range(warmup):
            dist.all_reduce(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            dist.barrier()
            t0 = timeit.default_timer()
            dist.all_reduce(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(timeit.default_timer() - t0)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, times)
        all_times = [t for ts in gathered for t in ts]
        results[size_mb] = (1000 * statistics.fmean(all_times), 1000 * statistics.pstdev(all_times))
        del x
    if rank == 0:
        return_dict.update(results)
    dist.barrier()
    dist.destroy_process_group()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="nccl" if torch.cuda.is_available() else "gloo", choices=["gloo", "nccl"])
    p.add_argument("--world-sizes", nargs="+", type=int, default=[2, 4, 6])
    p.add_argument("--sizes-mb", nargs="+", type=float, default=[1, 10, 100, 1000])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--port", type=int, default=29517)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    rows = []
    for ws in args.world_sizes:
        manager = mp.Manager()
        shared = manager.dict()
        mp.spawn(_worker, args=(ws, args.backend, args.sizes_mb, args.warmup, args.iters, args.port, shared), nprocs=ws, join=True)
        for size_mb in args.sizes_mb:
            mean_ms, std_ms = shared[size_mb]
            # Ring all-reduce moves 2(N-1)/N * S bytes per rank: "bus bandwidth" as NCCL reports it.
            algo_bw = (2 * (ws - 1) / ws) * size_mb * 2**20 / (mean_ms / 1000) / 1e9
            rows.append({"backend": args.backend, "world_size": ws, "size_mb": size_mb, "mean_ms": mean_ms, "std_ms": std_ms, "bus_bw_GBps": algo_bw})
            print(f"{args.backend} world={ws} size={size_mb:7.1f}MB  {mean_ms:9.3f} ± {std_ms:6.3f} ms  ({algo_bw:.2f} GB/s bus bw)", flush=True)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
