"""Attention benchmarking (handout §4.1.1 ``pytorch_attention``, §4.2 ``torch_compile``,
and ``flash_benchmarking``).

    uv run python -m cs336_systems.attention_benchmark --impl naive compiled flash_pytorch \
        --d 16 32 64 128 --seq 256 1024 4096 8192 16384 --csv results/attention.csv
    uv run python -m cs336_systems.attention_benchmark --impl naive flash_triton --causal \
        --batch 1 --dtype bf16 fp32 --seq 128 256 ... 65536 --d 16 32 64 128 --bench do_bench

Implementations: ``naive`` (materialises the score matrix), ``compiled``
(``torch.compile`` of the naive one), ``flash_pytorch`` (tiled autograd.Function),
``flash_triton`` (Triton kernels, Linux+CUDA). For each config we time forward,
backward, and forward+backward, and record memory held right before backward.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import timeit
from collections.abc import Callable

import torch
import torch._functorch.config as _functorch_config

from cs336_systems.flash_attention import flash_attention_pytorch, naive_attention

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# The backward-only timing calls backward(retain_graph=True) on one graph; compiled
# backward functions reject that unless donated buffers are disabled.
_functorch_config.donated_buffer = False


def get_impl(name: str) -> Callable:
    if name == "naive":
        return naive_attention
    if name == "compiled":
        return torch.compile(naive_attention)
    if name == "flash_pytorch":
        return flash_attention_pytorch
    if name == "flash_triton":
        from cs336_systems.flash_triton import flash_attention_triton

        return flash_attention_triton
    if name == "sdpa":  # PyTorch's fused kernels, as an extra reference point
        return lambda q, k, v, is_causal=False: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    raise ValueError(name)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn: Callable, iters: int, device, method: str) -> float:
    """Milliseconds per call."""
    if method == "do_bench":
        import triton.testing

        return float(triton.testing.do_bench(fn, rep=max(iters, 10)))
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = timeit.default_timer()
        fn()
        _sync(device)
        times.append(timeit.default_timer() - t0)
    return 1000 * statistics.fmean(times)


def bench_config(impl, batch, seq, d, dtype, causal, device, warmup, iters, method) -> dict:
    q, k, v = (torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True) for _ in range(3))
    do = torch.randn(batch, seq, d, device=device, dtype=dtype)
    row = {"batch": batch, "seq": seq, "d": d, "dtype": str(dtype).split(".")[-1], "causal": causal}
    try:
        for _ in range(warmup):
            impl(q, k, v, causal).backward(do)
            q.grad = k.grad = v.grad = None
        _sync(device)

        def fwd():
            with torch.no_grad():
                impl(q, k, v, causal)

        row["fwd_ms"] = _time(fwd, iters, device, method)

        out = impl(q, k, v, causal)
        _sync(device)
        row["mem_before_bwd_mib"] = torch.cuda.memory_allocated() / 2**20 if device.type == "cuda" else float("nan")

        def bwd():
            out.backward(do, retain_graph=True)
            q.grad = k.grad = v.grad = None

        row["bwd_ms"] = _time(bwd, iters, device, method)
        out = None  # free the graph before timing fwd+bwd

        def fwd_bwd():
            impl(q, k, v, causal).backward(do)
            q.grad = k.grad = v.grad = None

        row["fwd_bwd_ms"] = _time(fwd_bwd, iters, device, method)
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row.update(fwd_ms=float("nan"), bwd_ms=float("nan"), fwd_bwd_ms=float("nan"), mem_before_bwd_mib=float("nan"), status="OOM")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", nargs="+", default=["naive", "compiled", "flash_pytorch"])
    p.add_argument("--d", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--seq", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dtype", nargs="+", default=["fp32"], choices=sorted(DTYPES))
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--bench", default="timer", choices=["timer", "do_bench"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)
    device = torch.device(args.device)

    rows = []
    for name in args.impl:
        impl = get_impl(name)
        for dtype_name, d, seq in itertools.product(args.dtype, args.d, args.seq):
            row = bench_config(impl, args.batch, seq, d, DTYPES[dtype_name], args.causal, device, args.warmup, args.iters, args.bench)
            row = {"impl": name, **row}
            rows.append(row)
            print(
                f"{name:14s} {dtype_name} d={d:4d} seq={seq:6d} "
                f"fwd={row['fwd_ms']:9.3f}ms bwd={row['bwd_ms']:9.3f}ms fwd+bwd={row['fwd_bwd_ms']:9.3f}ms "
                f"mem_before_bwd={row['mem_before_bwd_mib']:9.1f}MiB {row['status']}",
                flush=True,
            )
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
