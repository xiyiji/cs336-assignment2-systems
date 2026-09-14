<h1 align="center">CS336 Assignment 2 · Systems</h1>

<p align="center">
  <b>FlashAttention-2 in PyTorch and Triton · DDP · optimizer-state sharding · FSDP · profiling harness</b><br>
  Stanford CS336 (Spring 2026) <i>Systems and Parallelism</i> assignment, implemented from scratch, end to end.
</p>

<p align="center">
  <a href="https://github.com/xiyiji/cs336-assignment2-systems/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/xiyiji/cs336-assignment2-systems/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue">
  <img alt="torch" src="https://img.shields.io/badge/torch-2.11-ee4c2c">
  <img alt="triton" src="https://img.shields.io/badge/triton-kernels%20%2B%20interpreter%20CI-76b900">
  <a href="https://xiyiji.github.io/cs336-assignment2-systems/"><img alt="docs" src="https://img.shields.io/badge/docs-writeup%20%2B%20results-8a2be2"></a>
</p>

<p align="center">
  <a href="https://xiyiji.github.io/cs336-assignment2-systems/writeup.html">📄 Writeup</a> ·
  <a href="https://xiyiji.github.io/cs336-assignment2-systems/tables_cpu.html">📊 Result tables</a> ·
  <a href="./PLAN.md">🗺 Plan</a> ·
  <a href="./cs336_assignment2_systems.pdf">📘 Handout</a>
</p>

---

## What is in here

| Handout part | Problems | Implementation | Verified by |
|---|---|---|---|
| **Benchmarking & profiling** | `benchmarking_script`, `nsys_profile`, `mixed_precision_*`, `memory_profiling` | [`benchmark.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/benchmark.py) — fwd / fwd+bwd / train, autocast, `torch.compile`, checkpointing, NVTX ranges, memory snapshots, JSON output; [`mixed_precision.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/mixed_precision.py) | CPU runs in [`results/cpu`](https://github.com/xiyiji/cs336-assignment2-systems/tree/main/results/cpu) |
| **Activation checkpointing** | `gradient_checkpointing` | [`checkpointing.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/checkpointing.py) — residual accounting, segmented and recursive (`O(log N)`) checkpointing | [`test_extra_model.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/tests/test_extra_model.py), saved-bytes sweep |
| **FlashAttention-2, PyTorch** | `flash_forward (a)`, `flash_backward` | [`flash_attention.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/flash_attention.py) — tiled online-softmax forward, recomputation backward (`torch.compile`) | staff tests + causal / bf16 / rectangular extras |
| **FlashAttention-2, Triton** | `flash_forward (b,c)`, optional §4.2.3 | [`flash_triton.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/flash_triton.py) — forward kernel with causal early-exit **and** the two-pass tiled backward (`dK,dV` kernel + `dQ` kernel, no atomics) | staff tests on CUDA; **Triton interpreter in CI** on CPU |
| **Attention benchmarks** | `pytorch_attention`, `torch_compile`, `flash_benchmarking` | [`attention_benchmark.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/attention_benchmark.py) — naive / compiled / flash / SDPA, timer or `triton.testing.do_bench` | CPU sweep |
| **Distributed comms** | `distributed_communication_single_node` | [`distributed_benchmark.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/distributed_benchmark.py) — gloo / NCCL all-reduce sweep | CPU sweep |
| **DDP** | `naive_ddp`, `minimal_ddp_flat_*`, `ddp_overlap_individual_parameters*` | [`ddp.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/ddp.py) — `NaiveDDP`, `FlatDDP`, `DDPOverlapIndividual`, `DDPBucketed` | staff test + all four variants |
| **Optimizer state sharding** | `optimizer_state_sharding*` | [`sharded_optimizer.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/sharded_optimizer.py) — ZeRO-1-style, any `torch.optim.Optimizer`, param groups | staff test + param-group test |
| **FSDP** | `fsdp`, `fsdp_accounting` | [`fsdp.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/fsdp.py) — flat 1-D shards, prefetch two layers ahead, async reduce-scatter, `compute_dtype` mixed precision | staff tests (fp32/fp16) + world-size-4 / uneven-shape / fused-head tests |
| **Parallelism analysis** | `alternate_ring_all_reduce`, `data_parallel_calcs`, `fsdp_calcs`, `tp_calcs`, `fsdp_tp_calcs` | derivations in the [writeup](writeup.html) | – |
| **Leaderboard** | `leaderboard` | [`leaderboard.py`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/leaderboard.py) — FSDP + bf16 + Triton flash + [fused chunked LM-head loss](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/cs336_systems/fused_ce.py) + checkpointing, `do_bench` harness | end-to-end CPU smoke; needs 2 GPUs to time |

Everything runnable on a CPU has been run (Apple M2, 8 cores); every
GPU-only measurement (Nsight, `memory_viz`, NCCL, B200 leaderboard) has its
exact command in [`scripts/run_gpu_suite.sh`](https://github.com/xiyiji/cs336-assignment2-systems/blob/main/scripts/run_gpu_suite.sh) and is
marked *not measured* in the writeup — no number is invented.

## Results at a glance (CPU, gloo, 2 processes, `small` = 128 M params)

| Data-parallel strategy | step | exposed comm | local params | local AdamW state |
|---|---|---|---|---|
| naive DDP (sync all-reduce per parameter) | 7.3 s | 0.63 s | 491 MiB | 981 MiB |
| flat DDP (one all-reduce) | 5.6 s | 1.53 s | 491 MiB | 981 MiB |
| **overlapped DDP** (async per-parameter, hooks) | **4.0 s** | **0.04 s** | 491 MiB | 981 MiB |
| bucketed DDP (25 MB buckets) | 5.8 s | 0.49 s | 491 MiB | 981 MiB |
| overlapped DDP + sharded optimizer | 3.3 s | 0.04 s | 491 MiB | **491 MiB** |
| **FSDP** | 3.3 s | – | **245 MiB** | **491 MiB** |

| End-to-end (`small`, batch 4, 5 warm-up / 10 timed) | ctx 128 | ctx 256 | ctx 512 |
|---|---|---|---|
| forward | 0.36 s | 0.50 s | 2.95 s |
| forward + backward | 1.03 s | 1.89 s | 6.60 s |
| full training step (+ AdamW) | 1.76 s | 2.65 s | 8.92 s |

More: attention sweeps, all-reduce bandwidth, residual accounting
(an eager `xl` block saves 9.4 GiB for backward at ctx 2048; checkpointing
every block is optimal because the checkpoint is 70× smaller than a block's
residuals), warm-up ablation — all in the
[writeup](https://xiyiji.github.io/cs336-assignment2-systems/writeup.html).

## Quick start

```bash
uv sync                          # Python 3.12/3.13, torch 2.11 (Linux: uv sync --extra triton)
make test                        # staff tests + extra tests on CPU with gloo (~1.5 min)
make test-triton-cpu             # Linux: Triton kernels through the interpreter
make cpu-suite                   # every CPU benchmark  -> results/cpu
make gpu-suite                   # every GPU experiment -> results/gpu   (>= 2 GPUs)

uv run python -m cs336_systems.benchmark --size small --mode train --dtype bf16 --compile
uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy naive flat overlap bucketed fsdp
uv run python -m cs336_systems.leaderboard --world-size 2
```

## Design notes

**FlashAttention-2.** The PyTorch version follows Algorithm 1 literally
(online softmax over `(B_q, B_k)` tiles, `L = m + log l` saved for backward).
The Triton forward uses the handout's block-pointer signature with a single
key-tile loop, causal early exit at the diagonal, fp32 accumulators and
`tl.dot(..., acc=acc)`. The backward is the FA2 two-pass scheme of Algorithm 2:
one kernel owns a key tile and accumulates `dK, dV` over query tiles, a second
owns a query tile and accumulates `dQ`, so no atomics are needed; `D = rowsum(O∘dO)`
is computed once in PyTorch. A `torch.compile`d recomputation backward is the
fallback (`FlashAttentionTriton.backward_impl = "compiled"`). The kernels are
validated on every push through the Triton interpreter (`TRITON_INTERPRET=1`).

**DDP.** All variants broadcast rank 0's weights at construction and average
gradients. The overlapped version registers `post_accumulate_grad_hook`s that
launch `all_reduce(async_op=True)` per parameter; `finish_gradient_synchronization`
waits. Bucketed DDP fills buckets in reverse parameter order and reduces each
as one flat tensor as soon as its gradients are ready.

**Sharded optimizer.** `ShardedOptimizer(params, cls, **kw)` subclasses
`torch.optim.Optimizer`; `add_param_group` assigns each parameter to the rank
with the least elements so far (deterministic, no communication) and forwards
the local subset to the wrapped optimizer; `step` runs the local optimizer then
broadcasts every parameter from its owner.

**FSDP.** Linear/Embedding weights are flattened, zero-padded, and split into
equal 1-D shards that replace `param.data` in place, so names, `requires_grad`
and any optimizer keep working. Forward/backward hooks all-gather the shard
(cast to `compute_dtype` *before* the collective), point `param.data` at the
full tensor for the duration of the layer, prefetch two layers ahead in both
directions, and reduce-scatter the fp32 gradient asynchronously as soon as it
is accumulated. Norm weights stay replicated with async all-reduce. Any module
with `fsdp_shard_weight = True` (e.g. the fused LM-head loss) is sharded too.

## Layout

```
cs336_systems/
  config.py               Table 1 model sizes, leaderboard config
  benchmark.py            end-to-end benchmark CLI
  mixed_precision.py      accumulation experiment, autocast dtype inspection
  checkpointing.py        residual accounting, segmented / recursive checkpointing
  flash_attention.py      FlashAttentionPytorch
  flash_triton.py         flash_fwd_kernel, flash_bwd_dkdv_kernel, flash_bwd_dq_kernel, FlashAttentionTriton
  flash_patch.py          swap the basics model's attention for flash
  attention_benchmark.py  attention sweeps
  ddp.py                  NaiveDDP, FlatDDP, DDPOverlapIndividual, DDPBucketed
  sharded_optimizer.py    ShardedOptimizer
  fsdp.py                 FSDP
  distributed_benchmark.py, ddp_benchmark.py, leaderboard.py, fused_ce.py
scripts/                  run_cpu_suite.sh, run_gpu_suite.sh, autograd_residuals.py, make_tables.py, build_docs.py
tests/                    staff tests (unchanged) + test_extra_{flash,distributed,model}.py
writeup/                  writeup.md, tables_cpu.md
results/cpu/              raw outputs of every CPU experiment
docs/                     GitHub Pages site (generated by scripts/build_docs.py)
```

## Acknowledgements

Scaffold, tests and handout are from
[stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
(MIT). All solution code in `cs336_systems/`, the extra tests, scripts and
the writeup are original.
