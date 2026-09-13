"""Autograd residual accounting and checkpointing experiments (handout §3).

    uv run python scripts/autograd_residuals.py rmsnorm            # §3.1 hook demo (+ torch.compile fusion)
    uv run python scripts/autograd_residuals.py block --size xl --context-length 2048
    uv run python scripts/autograd_residuals.py sweep --size xl --context-length 2048 --num-layers 32

``block`` measures the bytes saved for backward by one TransformerBlock (eager and,
when ``--compile`` is given, fused with ``torch.compile``).
``sweep`` measures the saved bytes of the full stack of blocks under single-level
checkpointing for every segment size, plus recursive checkpointing, and prints the
analytic model ``ceil(N/s)*C + s*R`` next to the measurement.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from cs336_basics.model import RMSNorm, RotaryEmbedding, TransformerBlock
from cs336_systems.checkpointing import (
    measure_saved_bytes,
    optimal_segment_size,
    peak_activation_bytes,
    recursive_checkpoint,
    segmented_checkpoint,
)
from cs336_systems.config import MODEL_SIZES

MiB = 2**20


def demo_rmsnorm(compile_: bool) -> None:
    x = torch.randn((4, 512, 2560), requires_grad=True)
    ln = RMSNorm(x.shape[-1])
    if compile_:
        ln = torch.compile(ln)

    def pack_hook(t):
        print(f"Saving residual: shape={tuple(t.shape)}, dtype={t.dtype}, grad_fn={t.grad_fn}")
        return t

    def unpack_hook(t):
        print(f"Loading residual: shape={tuple(t.shape)}, dtype={t.dtype}, grad_fn={t.grad_fn}")
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()


def make_block(size: str, context_length: int, device) -> TransformerBlock:
    spec = MODEL_SIZES[size]
    rope = RotaryEmbedding(context_length=context_length, dim=spec.d_model // spec.num_heads)
    return TransformerBlock(d_model=spec.d_model, d_ff=spec.d_ff, num_heads=spec.num_heads, positional_encoder=rope).to(device)


def block_residuals(args) -> None:
    spec = MODEL_SIZES[args.size]
    block = make_block(args.size, args.context_length, args.device)
    if args.compile:
        block = torch.compile(block, fullgraph=True)
    x = torch.randn((args.batch_size, args.context_length, spec.d_model), device=args.device, requires_grad=True)
    total, records = measure_saved_bytes(lambda: block(x))
    print(f"{args.size} block, batch={args.batch_size}, ctx={args.context_length}, compiled={args.compile}")
    for shape, dtype in records:
        print(f"  saved {shape} {dtype}  {math.prod(shape) * torch.tensor([], dtype=dtype).element_size() / MiB:.2f} MiB")
    print(f"Total size of saved tensors in single TransformerBlock: {total / MiB:.2f} MiB")
    residual = args.batch_size * args.context_length * spec.d_model * 4 / MiB
    print(f"One residual-stream activation tensor: {residual:.2f} MiB (batch*ctx*d_model*4 bytes)")


def sweep(args) -> None:
    spec = MODEL_SIZES[args.size]
    n = args.num_layers or spec.num_layers
    block = make_block(args.size, args.context_length, args.device)
    blocks = [block] * n  # weights are shared but residual memory is per call, which is what we count
    x = torch.randn((args.batch_size, args.context_length, spec.d_model), device=args.device, requires_grad=True)

    per_block, _ = measure_saved_bytes(lambda: block(x))
    ckpt_bytes = x.numel() * x.element_size()
    print(f"{args.size}: N={n} blocks, residuals per block R={per_block / MiB:.1f} MiB, checkpoint input C={ckpt_bytes / MiB:.1f} MiB")
    print(f"{'segment':>8} {'saved_fwd_MiB':>14} {'model_peak_MiB':>15}")
    best = optimal_segment_size(n, per_block, ckpt_bytes)
    for s in sorted({1, 2, 3, 4, 6, 8, 12, 16, n, best, max(1, best - 1), min(n, best + 1)}):
        if s > n:
            continue
        saved, _ = measure_saved_bytes(lambda: segmented_checkpoint(blocks, x, s))
        print(f"{s:>8} {saved / MiB:>14.1f} {peak_activation_bytes(n, s, per_block, ckpt_bytes) / MiB:>15.1f}{'  <- analytic optimum' if s == best else ''}")
    saved, _ = measure_saved_bytes(lambda: recursive_checkpoint(blocks, x))
    print(f"{'recursive':>8} {saved / MiB:>14.1f} {'~' + str(round((math.ceil(math.log2(n)) * ckpt_bytes + per_block) / MiB, 1)):>15}")
    none, _ = measure_saved_bytes(lambda: nn.Sequential(*blocks)(x))
    print(f"{'none':>8} {none / MiB:>14.1f} {none / MiB:>15.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["rmsnorm", "block", "sweep"])
    p.add_argument("--size", default="xl", choices=sorted(MODEL_SIZES))
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.experiment == "rmsnorm":
        demo_rmsnorm(args.compile)
    elif args.experiment == "block":
        block_residuals(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
