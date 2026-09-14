# PLAN — CS336 Assignment 2 (Systems), done end to end

Date: 2026-09-13. Target: complete every problem in the Spring 2026 handout
(v26.1.3, `cs336_assignment2_systems.pdf`) and beat the reference repo
[YifanLi3/cs336-assignment2-systems](https://github.com/YifanLi3/cs336-assignment2-systems).

## 1. What the reference repo actually contains

| Area | Reference (YifanLi3) | Status |
|---|---|---|
| Benchmarking script (fwd / bwd / optimizer, warmup, timing stats) | `benchmark.py` | done |
| Mixed precision (autocast flag) | flag in `benchmark.py` | done |
| Memory profiling (`_record_memory_history`) | flag in `benchmark.py` | done |
| Mixed-precision accumulation experiment | `mixed_precision_accumulation.py` | done |
| Attention benchmarking | `benchmark_attention.py` is a 10-line stub that does not run | **missing** |
| torch.compile comparison | – | **missing** |
| FlashAttention-2 (PyTorch autograd.Function) | – | **missing** |
| FlashAttention-2 (Triton forward kernel) | – | **missing** |
| FlashAttention-2 backward (compiled / Triton) | – | **missing** |
| Distributed all-reduce benchmark | – | **missing** |
| Naive / flat / overlapped DDP | – | **missing** |
| Optimizer state sharding | – | **missing** |
| FSDP (new in 2026 handout) | – | **missing** |
| `tests/adapters.py` | all `NotImplementedError` → 0 tests pass | **missing** |
| Written answers (writeup) | – | **missing** |
| Analysis section (ring all-reduce, DP / FSDP / TP / 2D calcs) | – | **missing** |
| CI | – | **missing** |

It is also built on the Spring 2025 scaffold (v1.0.5); the 2026 scaffold adds
FSDP, removes bucketed DDP, and changes the model table. We start from the
2026 scaffold and keep the 2025 bucketed-DDP problem as a bonus.

## 2. What we will deliver (and how it is better)

1. **Every adapter implemented, every CPU-runnable test green**
   (`test_attention.py::*pytorch`, `test_ddp.py`, `test_sharded_optimizer.py`,
   `test_fsdp.py`), plus our own extra tests (causal PyTorch flash, flat / naive /
   bucketed DDP variants, FSDP at world size 4, checkpointing equivalence).
2. **FlashAttention-2 in three tiers**: tiled PyTorch reference, Triton forward
   kernel (handout signature + causal early-exit), and the *optional* two-pass
   Triton backward (Algorithm 2) with a compiled-PyTorch fallback.
   Triton kernels are validated in CI under `TRITON_INTERPRET=1` because this
   laptop (Apple M2) has no CUDA.
3. **Four DDP flavours** behind one interface: naive per-parameter all-reduce,
   flat single all-reduce, overlapped per-parameter async all-reduce (the graded
   one), and bucketed overlapped (2025 bonus).
4. **Sharded optimizer** (ZeRO-1-style, round-robin by parameter size, works
   with any `torch.optim.Optimizer`, supports `add_param_group`).
5. **FSDP** with flat 1-D sharding (any shape, any world size), forward/backward
   all-gather via `.data` swapping, prefetch two layers ahead, async
   reduce-scatter of gradients, replicated-parameter all-reduce, optional
   `compute_dtype` mixed precision with fp32 master weights.
6. **One benchmarking CLI** (`cs336_systems.benchmark`) covering model size,
   context length, fwd / fwd+bwd / full step, autocast, torch.compile,
   activation checkpointing, NVTX ranges, memory snapshots, JSON output.
7. **Benchmark scripts for every measured problem** (attention sweep,
   flash vs. PyTorch via `triton.testing.do_bench`, all-reduce sweep, DDP
   variants, sharded-optimizer memory, FSDP), each writing CSV/Markdown tables.
8. **Writeup** (`writeup/writeup.md`) answering every written question with
   derivations; CPU-measured numbers where possible, exact GPU commands
   otherwise (no fabricated GPU numbers).
9. **CI** on GitHub Actions: ruff + CPU tests + Triton interpreter tests.

## 3. Execution order

1. Scaffold: config, package layout, `pyproject` (triton on Linux only).
2. FlashAttention PyTorch → tests → Triton fwd/bwd kernels.
3. DDP (naive → flat → overlap → bucketed) → tests.
4. Sharded optimizer → tests.
5. FSDP → tests (fp32 + fp16).
6. Benchmark CLI + experiment scripts; run what runs on CPU.
7. Writeup + README + CI; push to `github.com/xiyiji/cs336-assignment2-systems`.

## 4. Constraints

- No GPU locally: Triton and CUDA-only numbers are produced by scripts that are
  ready to run on a B200/H100 box; CI validates kernel logic with the Triton
  interpreter.
- Tests bind `MASTER_PORT=12390`, so distributed tests run serially.

## 5. Status (2026-09-14)

Everything in §2 is implemented and pushed; CI is green (lint, CPU tests, Triton interpreter tests);
the writeup and result tables are published at https://xiyiji.github.io/cs336-assignment2-systems/.
Only the GPU-only measurements (`scripts/run_gpu_suite.sh`) remain to be run on a multi-GPU box.
