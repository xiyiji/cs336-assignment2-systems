# CS336 Assignment 2 (Systems) — complete implementation

[![ci](https://github.com/xiyiji/cs336-assignment2-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/xiyiji/cs336-assignment2-systems/actions/workflows/ci.yml)

Everything in the Stanford CS336 Spring 2026 *Systems and Parallelism*
assignment ([handout](./cs336_assignment2_systems.pdf), scaffold
[stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) v26.1.4),
implemented from scratch: benchmarking/profiling harness, activation
checkpointing, FlashAttention-2 (PyTorch + Triton, forward **and** the optional
two-pass Triton backward), four DDP flavours, optimizer-state sharding, FSDP with
mixed precision, the parallelism-analysis derivations, and a leaderboard
training-step harness.

* [`PLAN.md`](./PLAN.md) — what the reference repo we started from was missing and how this one is organised.
* [`writeup/writeup.md`](./writeup/writeup.md) — answers to every written question, with measured numbers.
* [`results/cpu/`](./results/cpu) — raw outputs of every experiment that runs without a GPU (Apple M2, 8 cores).

## Status

| Part | Handout problems | Code | Tests | Numbers |
|---|---|---|---|---|
| Benchmarking harness | `benchmarking_script`, `nsys_profile`, `mixed_precision_*`, `memory_profiling` | `cs336_systems/benchmark.py`, `mixed_precision.py` | CLI smoke | CPU ✔, GPU: `scripts/run_gpu_suite.sh` |
| Activation checkpointing | `gradient_checkpointing` | `cs336_systems/checkpointing.py`, `scripts/autograd_residuals.py` | `tests/test_extra_model.py` | CPU ✔ |
| Attention benchmarking + compile | `pytorch_attention`, `torch_compile` | `cs336_systems/attention_benchmark.py` | – | CPU ✔ |
| FlashAttention-2 PyTorch | `flash_forward (a)`, `flash_backward` | `cs336_systems/flash_attention.py` | official ✔ + extra | – |
| FlashAttention-2 Triton fwd + bwd | `flash_forward (b,c)`, optional §4.2.3 | `cs336_systems/flash_triton.py` | official (CUDA) + interpreter in CI | GPU script |
| All-reduce benchmark | `distributed_communication_single_node` | `cs336_systems/distributed_benchmark.py` | – | CPU/gloo ✔ |
| DDP naive / flat / overlap / bucketed | `naive_ddp*`, `minimal_ddp_flat*`, `ddp_overlap_*` | `cs336_systems/ddp.py`, `ddp_benchmark.py` | official ✔ + extra | CPU ✔ |
| Optimizer state sharding | `optimizer_state_sharding*` | `cs336_systems/sharded_optimizer.py` | official ✔ + extra | CPU ✔ |
| FSDP | `fsdp`, `fsdp_accounting` | `cs336_systems/fsdp.py` | official ✔ (fp32, fp16) + extra (ws 4, uneven shapes) | CPU ✔ |
| Parallelism analysis | `alternate_ring_all_reduce`, `*_calcs` | – | – | derivations in writeup |
| Leaderboard | `leaderboard` | `cs336_systems/leaderboard.py`, `fused_ce.py`, `flash_patch.py` | extra | needs 2 GPUs |

CPU-only numbers (this laptop has no CUDA) are marked as such in the writeup;
every GPU-only measurement has a ready-to-run command and nothing is fabricated.

## Layout

```
cs336_systems/
  config.py             Table 1 model sizes, leaderboard config
  benchmark.py          end-to-end fwd / fwd+bwd / train benchmark (autocast, compile, checkpointing, NVTX, memory snapshots)
  mixed_precision.py    accumulation experiment, autocast dtype inspection
  checkpointing.py      saved-tensor accounting, segmented + recursive checkpointing wrapper
  flash_attention.py    FlashAttentionPytorch (tiled fwd, recomputation bwd, torch.compile)
  flash_triton.py       flash_fwd_kernel, flash_bwd_dkdv_kernel, flash_bwd_dq_kernel, FlashAttentionTriton
  flash_patch.py        swap the basics model's attention for flash
  attention_benchmark.py naive / compiled / flash / sdpa sweeps (timer or triton.testing.do_bench)
  ddp.py                NaiveDDP, FlatDDP, DDPOverlapIndividual (graded), DDPBucketed
  sharded_optimizer.py  ShardedOptimizer (ZeRO-1 style, any torch optimizer, param groups)
  fsdp.py               FSDP: flat 1-D shards, prefetch, async reduce-scatter, compute_dtype
  distributed_benchmark.py  all-reduce sweep (gloo / nccl)
  ddp_benchmark.py      training-step benchmark for every strategy (+ sharded optimizer, + FSDP)
  fused_ce.py           chunked LM-head + cross-entropy that never materialises the logits
  leaderboard.py        §9 harness: FSDP + bf16 + Triton flash + fused CE + checkpointing
scripts/
  run_cpu_suite.sh      every CPU-runnable experiment → results/cpu
  run_gpu_suite.sh      every GPU experiment (nsys, memory snapshots, do_bench, NCCL) → results/gpu
  autograd_residuals.py §3 residual accounting and checkpointing sweeps
  make_tables.py        results → Markdown tables
tests/                  staff tests (unchanged) + test_extra_*.py
writeup/writeup.md      the written deliverables
```

## Running

```bash
uv sync                      # Python 3.12/3.13, torch 2.11
make test                    # staff tests + extra tests (CPU, gloo); ~1.5 min
make test-triton-cpu         # Linux: Triton kernels through the interpreter
make cpu-suite               # ~1 h of CPU benchmarks → results/cpu
make gpu-suite               # on a ≥2-GPU Linux box → results/gpu
uv run python -m cs336_systems.benchmark --help
```

## Design notes

* **FlashAttention-2.** The PyTorch version follows Algorithm 1 literally
  (online softmax over `(B_q, B_k)` tiles, `L = m + log l` saved for backward).
  The Triton forward uses the handout's block-pointer signature with a single
  key-tile loop, causal early-exit at the diagonal, fp32 accumulators and
  `tl.dot(..., acc=acc)`. The backward is the FA2 two-pass scheme of Algorithm 2:
  one kernel owns a key tile and accumulates `dK, dV` over query tiles, a second
  owns a query tile and accumulates `dQ`, so no atomics are needed; `D = rowsum(O∘dO)`
  is computed once in PyTorch. A `torch.compile`d recomputation backward is the
  fallback (`FlashAttentionTriton.backward_impl = "compiled"`).
* **DDP.** All variants broadcast rank 0's weights at construction and average
  gradients. The overlapped version registers `post_accumulate_grad_hook`s that
  launch `all_reduce(async_op=True)` per parameter; `finish_gradient_synchronization`
  waits. Bucketed DDP fills buckets in reverse parameter order and reduces each as
  one flat tensor.
* **Sharded optimizer.** `ShardedOptimizer(params, cls, **kw)` subclasses
  `torch.optim.Optimizer`; `add_param_group` assigns each parameter to the rank
  with the least elements so far (deterministic, no communication) and forwards
  the local subset to the wrapped optimizer; `step` runs the local optimizer then
  broadcasts every parameter from its owner.
* **FSDP.** Linear/Embedding weights are flattened, zero-padded, and split into
  equal 1-D shards that replace `param.data` in place, so names, `requires_grad`,
  and any optimizer keep working. Forward/backward hooks all-gather the shard
  (cast to `compute_dtype` *before* the collective), point `param.data` at the
  full tensor for the duration of the layer, prefetch two layers ahead in both
  directions, and reduce-scatter the fp32 gradient asynchronously as soon as it
  is accumulated. Norm weights stay replicated with async all-reduce.
