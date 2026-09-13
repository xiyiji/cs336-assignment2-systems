"""Leaderboard training-step benchmark (handout §9).

The reference test times one full training step (forward, loss, backward,
AdamW) of an ~8B model at batch 2 x 32768 tokens on two B200s with
``triton.testing.do_bench``. This script reproduces that harness and stacks the
optimisations implemented in this repo:

* FlashAttention-2 Triton kernels (forward + two-pass backward) in place of the
  basics attention (``--attention triton``);
* bf16 compute with fp32 master weights through our FSDP over all GPUs
  (``--strategy fsdp``), or plain DDP for comparison;
* chunked fused LM-head + cross-entropy that never materialises the
  ``(batch, seq, vocab)`` logits (``--chunked-ce``);
* activation checkpointing (``--checkpointing segments --segment-size k``);
* ``torch.compile`` of the transformer blocks (``--compile``).

    uv run python -m cs336_systems.leaderboard --world-size 2                 # full config
    uv run python -m cs336_systems.leaderboard --size tiny --world-size 1 --device cpu --attention pytorch --rep 3 --warmup 1
"""

from __future__ import annotations

import argparse
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
from cs336_systems.checkpointing import CheckpointedTransformerLM, recursive_checkpoint, segmented_checkpoint
from cs336_systems.config import MODEL_SIZES, LeaderboardConfig, get_model_kwargs
from cs336_systems.ddp import DDPOverlapIndividual
from cs336_systems.flash_patch import patch_basics_attention
from cs336_systems.fsdp import FSDP
from cs336_systems.fused_ce import FusedLMHeadLoss


def build_model(args):
    if args.size == "leaderboard":
        cfg = LeaderboardConfig()
        kwargs = dict(
            vocab_size=cfg.vocab_size, context_length=cfg.ctx_len, d_model=cfg.d_model, num_layers=cfg.num_layers,
            num_heads=cfg.num_heads, d_ff=cfg.d_ff, rope_theta=10_000.0,
        )
        vocab, ctx = cfg.vocab_size, cfg.ctx_len
    else:
        kwargs = get_model_kwargs(args.size, args.context_length)
        vocab, ctx = kwargs["vocab_size"], args.context_length
    model = basics_model.BasicsTransformerLM(**kwargs)
    return model, vocab, ctx


def train_step_fn(model, optimizer, x, y, use_chunked_ce, autocast_ctx, sync_fn):
    inner = model.module if hasattr(model, "module") else model
    wrapper = inner if isinstance(inner, CheckpointedTransformerLM) else None
    inner = wrapper.model if wrapper is not None else inner

    def run_blocks(h):
        blocks = list(inner.layers)
        if wrapper is not None and wrapper.strategy == "segments":
            return segmented_checkpoint(blocks, h, wrapper.segment_size)
        if wrapper is not None and wrapper.strategy == "recursive":
            return recursive_checkpoint(blocks, h)
        for b in blocks:
            h = b(h)
        return h

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            h = inner.ln_final(run_blocks(inner.token_embeddings(x)))
            if use_chunked_ce:
                loss = inner.lm_head(h, y)  # FusedLMHeadLoss: never materialises the logits
            else:
                loss = cross_entropy(inner.lm_head(h), y)
        loss.backward()
        sync_fn()
        optimizer.step()

    return train_step


def _worker(rank, args, port, shared):
    if args.world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        backend = "nccl" if args.device == "cuda" else "gloo"
        if args.device == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=args.world_size)
    device = torch.device(args.device, rank) if args.device == "cuda" else torch.device(args.device)
    torch.manual_seed(0)
    patch_basics_attention(args.attention)

    model, vocab, ctx = build_model(args)
    model = model.to(device)
    if args.chunked_ce:
        model.lm_head = FusedLMHeadLoss(model.lm_head.weight, chunk_size=4096)
    if args.checkpointing != "none":
        model = CheckpointedTransformerLM(model, strategy=args.checkpointing, segment_size=args.segment_size)
    if args.compile:
        inner = model.model if isinstance(model, CheckpointedTransformerLM) else model
        inner.layers = torch.nn.ModuleList([torch.compile(b) for b in inner.layers])

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.world_size > 1 and args.strategy == "fsdp":
        # compute_dtype makes the *communicated* weights bf16; autocast additionally keeps
        # activations bf16 where fp32 buffers (RoPE) would otherwise promote them.
        model = FSDP(model, compute_dtype=dtype if dtype is not torch.float32 else None)
        autocast_ctx = torch.autocast(device.type, dtype=dtype) if dtype is not torch.float32 else nullcontext()
    elif args.world_size > 1:
        model = DDPOverlapIndividual(model)
        autocast_ctx = torch.autocast(device.type, dtype=dtype) if dtype is not torch.float32 else nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type, dtype=dtype) if dtype is not torch.float32 else nullcontext()
    sync_fn = model.finish_gradient_synchronization if hasattr(model, "finish_gradient_synchronization") else (lambda: None)

    optimizer = AdamW(model.parameters())
    local_bs = max(1, args.batch_size // args.world_size)
    x = torch.randint(0, vocab, (local_bs, ctx), device=device)
    y = torch.randint(0, vocab, (local_bs, ctx), device=device)
    step = train_step_fn(model, optimizer, x, y, args.chunked_ce, autocast_ctx, sync_fn)

    if device.type == "cuda":
        import triton.testing

        ms = float(triton.testing.do_bench(step, rep=args.rep, warmup=args.warmup))
    else:
        for _ in range(args.warmup):
            step()
        times = []
        for _ in range(args.rep):
            t0 = timeit.default_timer()
            step()
            times.append(timeit.default_timer() - t0)
        ms = 1000 * statistics.fmean(times)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    if args.world_size > 1:
        gathered = [None] * args.world_size
        dist.all_gather_object(gathered, (ms, peak))
        ms = max(m for m, _ in gathered)
        peak = max(p for _, p in gathered)
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        shared["ms"] = ms
        shared["peak_gib"] = peak


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="leaderboard", choices=["leaderboard", *sorted(MODEL_SIZES)])
    p.add_argument("--context-length", type=int, default=512, help="only for --size != leaderboard")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count() or 1)
    p.add_argument("--strategy", default="fsdp", choices=["fsdp", "ddp"])
    p.add_argument("--attention", default="triton" if torch.cuda.is_available() else "pytorch", choices=["triton", "pytorch", "original"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--chunked-ce", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpointing", default="segments", choices=["none", "segments", "recursive"])
    p.add_argument("--segment-size", type=int, default=2)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--rep", type=int, default=30_000, help="ms of timed reps for do_bench (count on CPU)")
    p.add_argument("--warmup", type=int, default=10_000, help="ms of warmup for do_bench (count on CPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=29561)
    args = p.parse_args(argv)

    manager = mp.Manager()
    shared = manager.dict()
    if args.world_size > 1:
        mp.spawn(_worker, args=(args, args.port, shared), nprocs=args.world_size, join=True)
    else:
        _worker(0, args, args.port, shared)
    print(
        f"leaderboard[{args.size}] ws={args.world_size} {args.strategy} attn={args.attention} {args.dtype} "
        f"chunked_ce={args.chunked_ce} ckpt={args.checkpointing}:{args.segment_size} compile={args.compile}: "
        f"{shared['ms']:.1f} ms / training step, peak {shared['peak_gib']:.1f} GiB"
    )
    return shared["ms"]


if __name__ == "__main__":
    main()
