"""End-to-end benchmarking of the basics Transformer (handout §2.1.3 - §2.1.6, §3, §4.2).

Examples
--------
    uv run python -m cs336_systems.benchmark --size small --mode train
    uv run python -m cs336_systems.benchmark --size xl --context-length 2048 --mode forward \
        --dtype bf16 --memory-snapshot xl_fwd.pickle
    uv run python -m cs336_systems.benchmark --size medium --compile --json results/e2e.jsonl
    uv run nsys profile --trace=cuda,nvtx -o small python -m cs336_systems.benchmark --size small --nvtx

``--mode`` selects forward-only, forward+backward, or a full AdamW training step.
Timings use ``timeit.default_timer`` around each step with ``torch.cuda.synchronize()``
(so the GPU has actually finished), after ``--warmup`` un-timed steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.config import BATCH_SIZE, DEFAULT_CONTEXT_LENGTH, MODEL_SIZES, VOCAB_SIZE, get_model_kwargs

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
# NVTX-annotated attention (handout §2.1.4)
# --------------------------------------------------------------------------- #
def install_annotated_attention() -> None:
    """Swap ``cs336_basics.model.scaled_dot_product_attention`` for an NVTX-annotated copy."""
    import torch.cuda.nvtx as nvtx
    from einops import einsum

    from cs336_basics.nn_utils import softmax

    @nvtx.range("scaled dot product attention")
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        with nvtx.range("computing attention scores"):
            scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
        with nvtx.range("computing softmax"):
            weights = softmax(scores, dim=-1)
        with nvtx.range("final matmul"):
            return einsum(weights, V, "... query key, ... key d_v ->  ... query d_v")

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def _nvtx_range(name: str):
    if torch.cuda.is_available():
        import torch.cuda.nvtx as nvtx

        return nvtx.range(name)
    return nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- #
@dataclass
class StepTiming:
    total: float
    forward: float
    backward: float
    optimizer: float


@dataclass
class BenchmarkResult:
    size: str
    context_length: int
    batch_size: int
    mode: str
    dtype: str
    compiled: bool
    checkpointing: str
    warmup: int
    steps: int
    device: str
    n_params: int
    mean_ms: float
    std_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    peak_memory_mib: float | None
    per_step_ms: list[float]


def build_model(args) -> torch.nn.Module:
    kwargs = get_model_kwargs(args.size, args.context_length, vocab_size=args.vocab_size)
    model = basics_model.BasicsTransformerLM(**kwargs)
    if args.checkpointing != "none":
        from cs336_systems.checkpointing import CheckpointedTransformerLM

        model = CheckpointedTransformerLM(model, strategy=args.checkpointing, segment_size=args.segment_size)
    return model


def run_step(
    model, x, y, mode: str, optimizer, autocast_ctx, device: torch.device, timings: StepTiming | None = None
) -> None:
    """One step. Synchronises between phases only when ``timings`` is requested."""
    t0 = timeit.default_timer()
    with _nvtx_range("forward"), autocast_ctx:
        logits = model(x)
        loss = cross_entropy(logits, y)
    if timings is not None:
        synchronize(device)
    t1 = timeit.default_timer()
    if mode != "forward":
        with _nvtx_range("backward"):
            loss.backward()
        if timings is not None:
            synchronize(device)
    t2 = timeit.default_timer()
    if mode == "train":
        with _nvtx_range("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if timings is not None:
            synchronize(device)
    elif mode == "forward_backward":
        model.zero_grad(set_to_none=True)
    t3 = timeit.default_timer()
    if timings is not None:
        timings.forward, timings.backward, timings.optimizer, timings.total = t1 - t0, t2 - t1, t3 - t2, t3 - t0


def benchmark(args) -> BenchmarkResult:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.compile:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=1e-3) if args.mode == "train" else None
    dtype = DTYPES[args.dtype]
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype) if dtype is not torch.float32 else nullcontext()
    )

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    model.train()
    with _nvtx_range("warmup"):
        for _ in range(args.warmup):
            run_step(model, x, y, args.mode, optimizer, autocast_ctx, device)
        synchronize(device)

    if args.memory_snapshot and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    per_step, fwd, bwd, opt = [], [], [], []
    for i in range(args.steps):
        t = StepTiming(0, 0, 0, 0)
        with _nvtx_range(f"step_{i}"):
            run_step(model, x, y, args.mode, optimizer, autocast_ctx, device, timings=t)
        per_step.append(t.total)
        fwd.append(t.forward)
        bwd.append(t.backward)
        opt.append(t.optimizer)

    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        if args.memory_snapshot:
            torch.cuda.memory._dump_snapshot(args.memory_snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"memory snapshot written to {args.memory_snapshot}")

    ms = [1000 * s for s in per_step]
    return BenchmarkResult(
        size=args.size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode=args.mode,
        dtype=args.dtype,
        compiled=args.compile,
        checkpointing=args.checkpointing if args.checkpointing == "none" else f"{args.checkpointing}:{args.segment_size}",
        warmup=args.warmup,
        steps=args.steps,
        device=f"{device.type}:{torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or platform.machine()}",
        n_params=n_params,
        mean_ms=statistics.fmean(ms),
        std_ms=statistics.pstdev(ms) if len(ms) > 1 else 0.0,
        forward_ms=1000 * statistics.fmean(fwd),
        backward_ms=1000 * statistics.fmean(bwd),
        optimizer_ms=1000 * statistics.fmean(opt),
        peak_memory_mib=peak,
        per_step_ms=ms,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="small", choices=sorted(MODEL_SIZES))
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--mode", default="train", choices=["forward", "forward_backward", "train"])
    p.add_argument("--dtype", default="fp32", choices=sorted(DTYPES), help="autocast dtype (fp32 = no autocast)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--compile", action="store_true", help="torch.compile the whole model")
    p.add_argument("--checkpointing", default="none", choices=["none", "segments", "recursive"])
    p.add_argument("--segment-size", type=int, default=None, help="blocks per checkpoint segment")
    p.add_argument("--nvtx", action="store_true", help="annotate attention with NVTX ranges")
    p.add_argument("--memory-snapshot", default=None, help="path for torch.cuda.memory._dump_snapshot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="append the result as one JSON line to this file")
    return p.parse_args(argv)


def main(argv=None) -> BenchmarkResult:
    args = parse_args(argv)
    if args.nvtx:
        install_annotated_attention()
    result = benchmark(args)
    print(
        f"[{result.size} ctx={result.context_length} bs={result.batch_size} {result.mode} {result.dtype}"
        f"{' compiled' if result.compiled else ''}{' ckpt=' + result.checkpointing if result.checkpointing != 'none' else ''}] "
        f"params={result.n_params / 1e6:.1f}M  step={result.mean_ms:.2f}±{result.std_ms:.2f} ms  "
        f"(fwd {result.forward_ms:.2f} / bwd {result.backward_ms:.2f} / opt {result.optimizer_ms:.2f})"
        + (f"  peak={result.peak_memory_mib:.0f} MiB" if result.peak_memory_mib is not None else "")
    )
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
    return result


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
