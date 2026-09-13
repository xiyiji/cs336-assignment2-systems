# CS336 Assignment 2 (Systems) — Writeup

Repository: <https://github.com/xiyiji/cs336-assignment2-systems>.
Handout version 26.1.3 (Spring 2026).

**Hardware note.** All measurements in this document were taken on an Apple M2
laptop (8 cores, 24 GB, no CUDA) with PyTorch 2.11 (fp32 unless stated, gloo
backend for anything distributed). Every problem that *requires* a GPU
(Nsight Systems, `memory_viz`, NCCL, Triton timing, the 2×B200 leaderboard) has
its exact command in [`scripts/run_gpu_suite.sh`](../scripts/run_gpu_suite.sh)
and is answered here with the analysis / expected behaviour, clearly labelled
**[GPU – not measured]**. No GPU number in this document is invented.
Raw outputs for everything that was measured are in
[`results/cpu/`](../results/cpu) and rendered in
[`tables_cpu.md`](./tables_cpu.md).

---

## 2 Profiling and Benchmarking

### Problem (benchmarking_script)

**(a)** [`cs336_systems/benchmark.py`](../cs336_systems/benchmark.py). It builds
`BasicsTransformerLM` from Table 1 (`--size`), draws a random batch, runs
`--warmup` un-timed steps and then `--steps` timed steps in one of three modes
(`--mode forward | forward_backward | train`; `train` includes the assignment-1
AdamW step). Each phase is timed with `timeit.default_timer()` and
`torch.cuda.synchronize()` is called after every phase. Extra switches used
later: `--dtype bf16` (autocast), `--compile`, `--checkpointing`, `--nvtx`,
`--memory-snapshot`, `--json`.

**(b)** CPU, `small` model, batch 4, 5 warm-up / 10 timed steps
(full table in `tables_cpu.md`, "End-to-end benchmark"):

<!-- E2E_TABLE -->

**(c)** Warm-up ablation (`small`, ctx 256, forward+backward, 10 timed steps):

<!-- WARMUP_TABLE -->

Without warm-up the first timed step is much slower and the standard deviation
explodes: the first iteration pays for lazy one-time work — on a GPU that is
CUDA context creation, cuBLAS/cuDNN handle creation and heuristic selection,
kernel module loading and the caching allocator growing its pool; on CPU it is
page-faulting the freshly allocated weight/activation buffers and spinning up
the OpenMP thread pool. With only 1–2 warm-up steps the numbers can still
differ because some one-time costs happen on the *second* step (e.g. the
optimizer's first `step()` allocates its state, the allocator keeps growing
until the steady-state working set has been touched once, and GPU clocks take
several hundred ms to boost), so the handout's 5 warm-up steps is a safer
default.

### Problem (nsys_profile) **[GPU – not measured]**

Commands: `scripts/run_gpu_suite.sh` (section "§2.1.4"): two sizes × three
context lengths, `--nvtx` swaps in the NVTX-annotated attention from
`cs336_systems.benchmark.install_annotated_attention`, `--capture-range=nvtx
-p step_0` skips the warm-up range.

What the profile answers, and what to expect:

* (a) The forward-pass NVTX range should match the `timeit` number within a few
  percent; the profiler adds a small constant overhead per kernel launch.
* (b) The kernel with the largest cumulative time in the forward pass is the
  GEMM (cuBLAS `sm90_xmma_gemm_*`/`cutlass` on Hopper/Blackwell, `ampere_sgemm`
  on older parts). It is launched 6 times per block (q/k/v/out projections and
  the three SwiGLU matmuls) plus 2 batched GEMMs for `QKᵀ` and `PV`, plus the
  LM head: `8·L + 1` per forward (97 for `small`). In forward+backward the same
  GEMM family still dominates (backward has ~2× the matmul FLOPs), joined by
  the transposed variants.
* (c) Non-matmul kernels with non-trivial time: `softmax` (materialised
  `seq×seq` scores), elementwise kernels for RMSNorm (`pow`, `mean`, `rsqrt`,
  `mul`), SiLU/gating multiplies, RoPE's `cos/sin` multiplies and `cat`,
  `where` for the causal mask, and `copy_`/`contiguous` from `rearrange`.
* (d) With the full training step, the matmul fraction drops: AdamW adds a
  dozen elementwise/`foreach` kernels per parameter tensor (and our assignment-1
  AdamW is not fused), and cross-entropy adds `log_softmax`, `gather`, `mean`.
* (e) Inside attention at ctx 512 the softmax kernel takes a time of the same
  order as the two matmuls even though its FLOPs are ~`d/2 ≈ 32×` smaller —
  softmax is memory-bound (reads and writes the whole `seq×seq` matrix twice),
  the matmuls are compute-bound and reach a much higher fraction of peak.

### Problem (mixed_precision_accumulation)

```
fp32 accumulator, fp32 addend                      10.000134
fp16 accumulator, fp16 addend                      9.953125
fp32 accumulator, fp16 addend (implicit upcast)    10.002136
fp32 accumulator, fp16 addend cast to fp32         10.002136
exact                                              10.000000
```

fp32+fp32 is off by 1.3e-4 only because 0.01 is not exactly representable.
fp16+fp16 is off by 0.5%: once the sum exceeds 8 the fp16 ulp is 2⁻⁷ ≈ 0.0078,
so adding 0.01 rounds to a multiple of it every time and error accumulates
systematically. With an fp32 accumulator the error is 2.1e-3 regardless of
whether the cast is implicit or explicit — that residual is purely the fp16
representation error of the addend (0.01 → 0.010002136, ×1000). Lesson: the
*accumulator* must be high precision; low-precision inputs are fine.

### Problem (benchmarking_mixed_precision)

**(a)** Under `torch.autocast(dtype=float16)` on CUDA:

| component | dtype | why |
|---|---|---|
| model parameters | float32 | autocast never changes parameters; it casts *copies* on the way into eligible ops |
| output of `fc1` | float16 | `linear`/`matmul` are on autocast's lower-precision list |
| output of `ln` (LayerNorm) | float32 | `layer_norm` is on the fp32 list (reductions need range/precision) |
| logits (`fc2`) | float16 | fp32 input is cast down for the matmul |
| loss | float32 | `cross_entropy`/`log_softmax` run in fp32 |
| gradients of the parameters | float32 | grads match parameter dtype; intermediate activation grads are fp16 |

(Measured on this machine's *CPU* autocast, `layer_norm` is *not* on the CPU
fp32 list, so the LayerNorm output came out fp16/bf16 —
`results/cpu/toy_autocast_fp16.txt` — a reminder that the op lists differ per
device.)

**(b)** The sensitive parts are the reductions: the mean and especially the
variance (a sum of squares that overflows fp16's 65504 range for activations of
magnitude ≳ 16, and suffers cancellation) and `eps = 1e-5`, which is below
fp16's smallest normal (6.1e-5). BF16 has fp32's exponent range, so overflow
and `eps` are no longer a problem and one *could* leave LayerNorm in bf16; but
bf16 has only 8 mantissa bits (vs 11), so the centred values `x − μ` and the
normalised outputs lose precision, which is why PyTorch keeps `layer_norm` in
fp32 under bf16 autocast too.

**(c)** fp32 vs bf16 autocast on this CPU (`small` ctx 256, `medium` ctx 128, fwd+bwd):

<!-- BF16_TABLE -->

**[GPU – not measured]** On a B200 bf16 autocast should give 3–4× on the
matmul-dominated sizes, with the gain growing with model size because the
matmul share of the step grows and the (fp32) elementwise/normalisation share
shrinks; the smallest model gains least because launch overhead and
memory-bound kernels dominate.

### Problem (memory_profiling)

**(a)** `--memory-snapshot path.pickle` records
`torch.cuda.memory._record_memory_history` after warm-up and dumps the snapshot
after the timed steps. **[GPU – not measured]** Expected timelines: the forward
pass is a monotonic ramp (activations accumulate layer by layer) followed by a
cliff when the outputs are freed; a training step shows the same ramp, then a
staircase *down* during backward as saved activations are released while
gradient tensors (same size as the weights) are allocated, then a plateau
during the optimizer step (AdamW's `m`, `v` are allocated at the first step and
persist).

**(b)** Analytic expectation for `xl` (2560/10240/32 layers, 3.41 B parameters,
vocab 10 000, batch 4), fp32:

| | ctx 128 | ctx 2048 |
|---|---|---|
| parameters | 13.6 GB | 13.6 GB |
| forward activations (≈ 32 × per-block residuals) | ≈ 7 GB | ≈ 114 GB (3.65 GB/block, §3) |
| + gradients + AdamW state for a full step | +40.9 GB | +40.9 GB |

So forward-only peaks around 21 GB (128) / 130 GB (2048), and a full step
around 62 GB / 170 GB — the 2048 training step only fits on a 192 GB B200.
Measure with the commands in the GPU suite and read `peak_memory_mib` from
`results/gpu/memory.jsonl`.

**(c)** **[GPU – not measured]** Mixed precision halves the *activation*
memory that is saved in bf16 (matmul outputs) but leaves parameters,
gradients, optimizer state and the fp32 outputs of RMSNorm/softmax/loss
untouched, so for the forward pass at long context the peak drops by roughly
30–40 %, while for the full training step at ctx 128 (dominated by the 54.6 GB
of fp32 parameter/gradient/optimizer state) it barely moves.

**(d)** One residual-stream activation tensor is `batch × ctx × d_model × 4 B`:
`4 × 512 × 2560 × 4 = 20 971 520 B = 20 MiB` at the default context 512
(5 MiB at 128, 80 MiB at 2048).

**(e)** **[GPU – not measured]** At the 10 % detail level the largest
allocations are the attention score/probability matrices of one layer,
`batch × heads × ctx × ctx × 4 B = 4·32·2048²·4 = 2 GiB` each at ctx 2048 (the
`einsum` for `QKᵀ`, the `where`-masked copy, and the softmax output), followed
by the SwiGLU intermediates `4·2048·10240·4 = 320 MiB`. The stack traces point
at `scaled_dot_product_attention` and `SwiGLU.forward`.

**(f)** **[GPU – not measured]** `nsys profile --cuda-memory-usage=true`
with the PyTorch NVTX labels. From the residual accounting in §3 (measured on
CPU, which is dtype/shape-exact): a single `xl` block at ctx 2048 saves
3651 MiB, dominated by (i) the three `seq×seq` attention tensors (3 × 2 GiB
… of which one is freed after softmax, see the table in
`results/cpu/residuals_xl_block.txt`), (ii) SwiGLU's `w1(x)`, `w3(x)`, `silu`
and the gated product (4 × 320 MiB), (iii) the two RMSNorm inputs and the
attention input/output (80 MiB each). During backward the block's gradients
are `4·d² + 3·d·d_ff + 2·d` fp32 values = 400 MiB per block, i.e. roughly
one ninth of the activations it frees, matching the gradient-tensor size one
expects from the parameter count (104.9 M params × 4 B).

---

## 3 Single-GPU Memory

Measured with [`scripts/autograd_residuals.py`](../scripts/autograd_residuals.py)
using `saved_tensors_hooks` (device-independent):

<!-- RESIDUALS_BLOCK -->

### Problem (gradient_checkpointing)

**(a)** Ignoring compute, nest the checkpoints recursively (binary tree): wrap
the first half of the blocks in one `checkpoint`, and inside it wrap *its* first
half, and so on. At any point in backward only one checkpointed input per
recursion level is alive plus one fully materialised leaf block, so peak
activation memory is `O(log N)` block-residual units (vs `O(N)` without and
`O(√N)` for the classic single-level `√N` scheme); every block is recomputed
once per level it sits in, so compute is `O(N log N)` forward-equivalents. Code
sketch (the real one is `cs336_systems.checkpointing.recursive_checkpoint`):

```python
def run(blocks, x):
    if len(blocks) == 1:
        return blocks[0](x)
    mid = len(blocks) // 2
    x = checkpoint(lambda h: run(blocks[:mid], h), x, use_reentrant=False)  # first half: recomputed later
    return run(blocks[mid:], x)                                             # second half: kept
```

**(b)** With one level of checkpointing and segments of `s` blocks, peak
activation memory is `⌈N/s⌉·C + s·R`, where `C` is the checkpointed input
(one residual-stream tensor, 80 MiB at ctx 2048) and `R` the residuals of one
block (3651 MiB measured). Minimising over `s` gives `s* = √(N·C/R) =
√(32·80/3651) ≈ 0.84`, i.e. **checkpoint every single block** (`s = 1`):
peak ≈ `32·80 MiB + 3651 MiB ≈ 6.2 GiB` of activations versus
`16·80 + 2·3651 ≈ 8.4 GiB` for `s = 2`, and `114 GiB` with no checkpointing.
Because `C ≪ R` the checkpoints themselves are nearly free, so the smallest
allowed segment wins and the "next smaller" size only exists by nesting
(forbidden here). Measured saved-bytes sweep on a `small`-sized stack
(12 blocks, ctx 512) confirming the `⌈N/s⌉·C + s·R` model:

<!-- CKPT_SWEEP -->

**[GPU – not measured]** The GPU suite runs `--checkpointing segments
--segment-size {1,2,3,…,32}` on `xl`/2048 and records `peak_memory_mib`.

---

## 4 GPU Kernels

### Problem (pytorch_attention)

Script: [`cs336_systems/attention_benchmark.py`](../cs336_systems/attention_benchmark.py)
(batch 8, no heads, 100 forward and 100 backward passes after warm-up, memory
read right before backward). CPU results (`iters=5`, seq ≤ 4096; larger
sequences were not run on the laptop):

<!-- ATTN_TABLE -->

Memory accounting for the smallest configuration that typically OOMs on an
80 GB GPU (d = 16, seq = 16384): the inputs are tiny (`3 × 8 × 16384 × 16 × 4 B
= 25 MiB`) but attention materialises `S = QKᵀ` of `8 × 16384² × 4 B = 8 GiB`,
the masked copy (another 8 GiB), and the softmax output `P` (8 GiB) which is
*saved for backward* together with `S`; backward then allocates `dP` and `dS`
of the same size, so ≈ 40 GiB of `seq²` tensors are live at the start of
backward and the run OOMs at `seq = 16384` for every `d` (and at 8192 for large
`d` once the per-iteration copies are counted). The memory saved for backward
scales as `O(batch · seq²)` and is independent of `d`, whereas useful work
scales as `O(seq² · d)` — so small `d` is the worst case. FlashAttention removes
it: save only `O` and the `seq`-sized logsumexp `L`, recompute `P` tile by tile
in backward (§4.2).

### Problem (torch_compile)

**(a)** `--impl compiled` in the same script (`torch.compile(naive_attention)`);
CPU numbers in the table above. Inductor fuses the scale, mask and softmax into
one kernel and removes the extra `where` copy, which mostly helps the
memory-bound softmax part; the matmuls are unchanged.

**(b)** Whole-model compile (`--compile`), `small` ctx 128 fwd+bwd on CPU:

<!-- COMPILE_TABLE -->

**[GPU – not measured]** Expect 1.2–1.5× on forward (RMSNorm/SiLU/RoPE fusion,
fewer launches) and a smaller relative gain on the full step because the
optimizer is not compiled.

### Problem (flash_forward)

**(a)** [`cs336_systems/flash_attention.py::FlashAttentionPytorch`](../cs336_systems/flash_attention.py):
Algorithm 1 in plain PyTorch — `(B_q, B_k) = (64, 64)` tiles, running max `m`,
running denominator `l`, un-normalised accumulator `O`, all in fp32, causal
tiles above the diagonal skipped. Saves `L = m + log l` (shape `(..., n_q)`),
`Q, K, V, O`. Passes `test_flash_forward_pass_pytorch` and our causal / batch-dim
/ rectangular extras.

**(b)** [`cs336_systems/flash_triton.py::flash_fwd_kernel`](../cs336_systems/flash_triton.py)
uses the handout's signature and block pointers, grid `(T_q, batch)`, one loop
over key tiles, fp32 on-chip buffers, `tl.dot(P.to(V.dtype), V, acc=O)`.

**(c)** `is_causal: tl.constexpr` builds `q_idx[:, None] >= k_idx[None, :]`
and adds `-1e6` to masked scores; in addition the loop stops at the diagonal
tile (`n_k_tiles = cdiv(min((i+1)·B_q, N_k), B_k)`), so fully-masked tiles are
never loaded. The flag defaults to `False`; `ctx.is_causal` is saved.

The Triton tests need CUDA; on this laptop they are skipped, but the kernels
are exercised in CI through the Triton interpreter (`TRITON_INTERPRET=1`,
`tests/test_extra_flash.py::test_triton_flash_forward_backward`).

### Problem (flash_backward)

`flash_backward` in `flash_attention.py` implements Eq. 13–19 with the `D =
rowsum(O ∘ dO)` trick and is wrapped in `torch.compile(dynamic=True)` on CUDA
(`CS336_COMPILE_FLASH_BACKWARD=1` forces it on CPU; CI runs that path).
`test_flash_backward_pytorch` passes. The Triton class can use the same
compiled backward (`FlashAttentionTriton.backward_impl = "compiled"`) or the
optional tiled Triton backward below (default).

**Optional §4.2.3.** `flash_bwd_dkdv_kernel` (grid over key tiles, inner loop
over query tiles, accumulates `dK, dV`) and `flash_bwd_dq_kernel` (grid over
query tiles, inner loop over key tiles, accumulates `dQ`) implement Algorithm 2
exactly; `P` is recomputed twice, no atomics. Both skip tiles above the
diagonal when causal.

### Problem (flash_benchmarking) **[GPU – not measured]**

`uv run python -m cs336_systems.attention_benchmark --impl naive flash_triton
--causal --batch 1 --bench do_bench --dtype bf16 fp32 --d 16 32 64 128 --seq
128 … 65536 --csv results/gpu/flash_benchmark.csv` produces the required table
(forward, backward, forward+backward for both implementations). Tile sizes are
class attributes (`FlashAttentionTriton.Q_TILE_SIZE / K_TILE_SIZE`) so they can
be swept per input size; the PyTorch baseline will OOM well before 65536.

---

## 5 Distributed Data Parallel Training

### Problem (distributed_communication_single_node)

[`cs336_systems/distributed_benchmark.py`](../cs336_systems/distributed_benchmark.py):
`mp.spawn` × world size, 5 warm-up all-reduces, 10 timed, timings gathered with
`all_gather_object` and averaged over ranks (`--backend nccl` on GPUs). CPU /
gloo, 2 / 4 / 6 processes, 1 MB – 1 GB:

<!-- ALLREDUCE_TABLE -->

<!-- ALLREDUCE_COMMENT -->

### Problem (naive_ddp)

[`cs336_systems/ddp.py::NaiveDDP`](../cs336_systems/ddp.py) broadcasts rank 0's
parameters and buffers at construction and, in `finish_gradient_synchronization`,
issues one synchronous `all_reduce` per parameter and divides by the world
size. The graded adapter uses `DDPOverlapIndividual` (below); all four variants
pass the staff DDP test harness (`tests/test_extra_distributed.py`).

### Problem (naive_ddp_benchmarking), (minimal_ddp_flat_benchmarking), (ddp_overlap_individual_parameters_benchmarking)

[`cs336_systems/ddp_benchmark.py`](../cs336_systems/ddp_benchmark.py) trains the
model with the chosen wrapper on a global batch split across ranks and reports
the mean step time and the time spent in `finish_gradient_synchronization`
(for the overlapped variants that is the *exposed* communication only).
Setup here: CPU, gloo, 2 processes, `small` model, ctx 128, global batch 4,
2 warm-up + 5 timed steps (the handout's setting is 1 node × 2 GPUs, `xl` —
`scripts/run_gpu_suite.sh` §5):

<!-- DDP_TABLE -->

<!-- DDP_COMMENT -->

`FlatDDP` concatenates all gradients with `torch._utils._flatten_dense_tensors`,
issues a single all-reduce and copies back; `DDPOverlapIndividual` registers a
`post_accumulate_grad_hook` on every parameter that pre-divides the gradient and
launches `all_reduce(async_op=True)`, and `finish_gradient_synchronization`
waits on the handles; `DDPBucketed` (Spring-2025 bonus) groups parameters in
reverse registration order into ≤ `bucket_size_mb` buckets and reduces each
bucket as one flat tensor as soon as all of its gradients are ready.

**(b)** **[GPU – not measured]** `nsys profile --trace=cuda,nvtx,nccl` on
`--strategy naive` vs `overlap` (commands in the GPU suite). In the naive trace
the NCCL `AllReduce` kernels form a solid band *after* the last backward kernel;
in the overlapped trace they are interleaved with the backward GEMMs on a
second stream and only a short tail remains after backward.

---

## 6 Optimizer State Sharding

### Problem (optimizer_state_sharding)

[`cs336_systems/sharded_optimizer.py::ShardedOptimizer`](../cs336_systems/sharded_optimizer.py)
subclasses `torch.optim.Optimizer`. `add_param_group` (called by the base
constructor and later by the user) records the full group, assigns every new
parameter to the rank with the smallest element count so far (deterministic,
no communication), and forwards the local subset to the wrapped optimizer
(created lazily so a rank that owns nothing in the first group is fine).
`step` runs the local optimizer and then broadcasts every parameter from its
owner (asynchronously, waited at the end); `zero_grad` clears *all* gradients
since backward produced them for every parameter. Passes the staff test and
our param-group / `add_param_group` test.

### Problem (optimizer_state_sharding_accounting)

**(a)** Memory. For `xl` (3.41 B parameters, fp32) per rank:

| stage | no sharding | sharded over 2 |
|---|---|---|
| after model init | params 13.6 GB | 13.6 GB |
| before optimizer step | + grads 13.6 GB (+ AdamW state 27.3 GB from step 2 on) = 54.6 GB | 13.6 + 13.6 + 13.6 = 40.9 GB |
| after optimizer step | 54.6 GB | 40.9 GB |

Sharding only removes the optimizer state of the other rank's parameters
(`2·Ψ·4 B·(1 − 1/N)` = 13.6 GB here); parameters and gradients stay fully
replicated. Our CPU run reports the analytic byte counts
(`param_bytes_local`, `opt_state_bytes_local`) and they match: with sharding
the local AdamW state halves while parameter bytes are unchanged:

<!-- SHARDED_TABLE -->

**(b)** Speed: the per-rank optimizer step is ~2× cheaper (half the
parameters), but each step now ends with a broadcast of *all* parameters
(`Ψ·4 B` per step, 13.6 GB for `xl`) that is not overlapped with anything, so on
CPU/gloo the step got slower overall (see table). On NCCL with fast NVLink the
extra broadcast is cheaper than the saved optimizer work for large models.

**(c)** ZeRO stage 1 (`P_os`) also partitions optimizer state 1/N per rank, but
it *reduce-scatters* the gradients (each rank receives the averaged gradient
only for its partition, communication `Ψ`) and then *all-gathers* the updated
parameters (`Ψ`), for a total of `2Ψ` per step — the same as plain DDP's
all-reduce — and it partitions a flat contiguous buffer so the shards are
exactly equal. Our implementation keeps DDP's full all-reduce (`2Ψ`, and every
rank holds the full averaged gradient, so gradient memory is not reduced) and
adds a parameter broadcast (`Ψ`), i.e. `3Ψ` of communication, and shards at
tensor granularity (slightly unbalanced).

---

## 7 Fully-Sharded Data Parallel

### Problem (fsdp)

[`cs336_systems/fsdp.py::FSDP`](../cs336_systems/fsdp.py). Every `Linear` /
`Embedding` weight is flattened, zero-padded to a multiple of the world size
and split into equal 1-D shards; `param.data` is replaced by the local shard in
place, so parameter names, `requires_grad` and any optimizer (SGD, AdamW from
assignment 1) work unchanged, and `fsdp_gather_full_params` all-gathers the
shards back. Norm weights are not sharded (their gradients are averaged with an
async all-reduce). Forward pre-hooks all-gather the shard — cast to
`compute_dtype` *before* the collective, so communication is bf16/fp16 while
master weights stay fp32 — and point `param.data` at the full tensor for the
duration of the layer; post-hooks restore the shard. The first forward records
layer order; afterwards the gather for layer `k` is launched when layer `k−2`
finishes (the first two at the start of forward), as the handout prescribes,
and the same two-ahead prefetch runs backwards in the backward pass (a hook on
each layer's output re-gathers just before that layer's backward node). When
the full-size gradient has been accumulated a `post_accumulate_grad_hook`
casts it to fp32, launches an asynchronous `reduce_scatter_tensor` into a
shard-sized buffer and frees the full gradient; `finish_gradient_synchronization`
waits and installs the shard gradients. Passes `tests/test_fsdp.py` (fp32 and
fp16 with the 1e-4 tolerance) and our extra tests (world size 4, parameter
counts not divisible by the world size, `torch.nn` layers with biases, a
sharded fused LM-head loss).

### Problem (fsdp_accounting)

**(a)** With DP + sharded optimizer we were at `(4 + 4 + 8/N)·Ψ` bytes per rank
= 40.9 GB for `xl` on 2 GPUs; FSDP shards parameters and gradients as well,
`(4 + 4 + 8)·Ψ/N = 16Ψ/N` = 27.3 GB per rank, so the expected saving from the
peak is `(8 − 8/N)·Ψ`… concretely 13.6 GB versus sharded-optimizer DDP and
27.3 GB versus plain DDP, ignoring the transient all-gather buffers (at most
two full layers' weights in `compute_dtype` — 2 × 100 MiB for the largest `xl`
matrix in fp32, half that in bf16). Our CPU run shows the per-rank parameter
*and* optimizer-state bytes halving (table above).

**(b)** **[GPU – not measured]** `nsys profile --trace=cuda,nvtx,nccl` on
`--strategy fsdp` (GPU suite §7). With the two-layer prefetch the
`AllGather` kernels for layer `k` sit between the GEMMs of layers `k−2` and
`k−1` on the communication stream; on NVLink the 100 MiB gather (≈ 0.4 ms at
~250 GB/s) is far shorter than a layer's forward at ctx 512 (~3 ms for `xl`),
so it should finish in time and no gap appears before layer `k`'s first GEMM.

---

## 8 Analyzing Parallelism Strategies

### Problem (alternate_ring_all_reduce)

`(N − 1)·S / W`. Every step each device sends a *full* tensor `x^(j)` of `S`
bytes (not an `S/N` chunk) and there are `N − 1` steps, so each device's egress
carries `(N − 1)·S` bytes at rate `W` — `N`× the ring reduce-scatter +
all-gather cost of `2(N−1)/N · S/W` for large `N`.

### Problem (data_parallel_calcs)

**(a)** Backward FLOPs = `12·B·D·D_FF / N_DP`. The six backward matmuls
(`dz`, `dx₁W₁ᵀ`, `dx₂W₂ᵀ`, `dW₃`, `dW₂`, `dW₁`) are each `2·(B/N_DP)·D·D_FF`
FLOPs (twice the forward's three).

**(b)** Communication time = `12·D·D_FF·(N_DP − 1) / (N_DP·W)`. The three
weight gradients hold `3·D·D_FF` fp16 values = `6·D·D_FF` bytes, and a ring
all-reduce of `S` bytes costs `2·(N−1)/N · S/W`.

**(c)** Compute time is `12·B·D·D_FF / (N_DP·C)`; communication exceeds it when
`(N_DP − 1)/W > B/C`, i.e. we stay compute-bound while
`N_DP ≤ 1 + B·W / C` (≈ `B·W/C`). Only the per-device batch matters — the model
size cancels.

### Problem (fsdp_calcs)

**(a)** Same FLOPs as DP: backward `12·B·D·D_FF / N_FSDP`, forward
`6·B·D·D_FF / N_FSDP` — sharding weights changes communication, not arithmetic.

**(b)** Forward: three all-gathers of `2·D·D_FF` bytes each, `(N−1)/N · S/W`
apiece → `6·D·D_FF·(N − 1)/(N·W)`. Backward: the same three all-gathers again
plus three reduce-scatters of the same size → `12·D·D_FF·(N − 1)/(N·W)`.

**(c)** Backward: `12·D·D_FF·(N−1)/(N·W) ≤ 12·B·D·D_FF/(N·C)` ⇔
`N_FSDP ≤ 1 + B·W/C`. Forward: `6·D·D_FF·(N−1)/(N·W) ≤ 6·B·D·D_FF/(N·C)` ⇔
`N_FSDP ≤ 1 + B·W/C`. Identical bound to DP: FSDP's extra volume (`3Ψ` vs `2Ψ`
per backward, plus `Ψ` in forward) is matched by the extra compute it is
overlapped against.

### Problem (tp_calcs)

**(a)** `dy` is replicated on all TP ranks (it is the gradient of the
all-reduced `y`). With `W₁⁽ⁱ⁾, W₂⁽ⁱ⁾ ∈ ℝ^{D×D_FF/N}` and `W₃⁽ⁱ⁾ ∈ ℝ^{D_FF/N×D}`:

```
dz⁽ⁱ⁾  = dy · W₃⁽ⁱ⁾ᵀ                       (B, D_FF/N)
dW₃⁽ⁱ⁾ = z⁽ⁱ⁾ᵀ · dy                        (D_FF/N, D)
dx₂⁽ⁱ⁾ = dz⁽ⁱ⁾ ∗ f(x₁⁽ⁱ⁾)                   (B, D_FF/N)
dx₁⁽ⁱ⁾ = dz⁽ⁱ⁾ ∗ f′(x₁⁽ⁱ⁾) ∗ x₂⁽ⁱ⁾          (B, D_FF/N)
dW₁⁽ⁱ⁾ = xᵀ · dx₁⁽ⁱ⁾,   dW₂⁽ⁱ⁾ = xᵀ · dx₂⁽ⁱ⁾   (D, D_FF/N)
dx̃⁽ⁱ⁾  = dx₁⁽ⁱ⁾ · W₁⁽ⁱ⁾ᵀ + dx₂⁽ⁱ⁾ · W₂⁽ⁱ⁾ᵀ    (B, D)   partial sum over the D_FF shard
dx     = all-reduce({dx̃⁽ⁱ⁾}ᵢ)
```

No communication is needed for the weight gradients (each device owns its
shard); the only collective is the all-reduce of `dx` (the backward of the
column-parallel input broadcast), mirroring the forward's all-reduce of `y`.

**(b)** Forward `6·B·D·D_FF / N_TP` (three matmuls with the `D_FF` dimension
sharded), backward `12·B·D·D_FF / N_TP`.

**(c)** Forward: one all-reduce of `y` (`2·B·D` bytes) →
`4·B·D·(N_TP − 1)/(N_TP·W)`. Backward: one all-reduce of `dx` of the same size
→ `4·B·D·(N_TP − 1)/(N_TP·W)`.

**(d)** Backward: `4·B·D·(N−1)/(N·W) ≤ 12·B·D·D_FF/(N·C)` ⇔
`N_TP ≤ 1 + 3·D_FF·W / C`. Forward: `4·B·D·(N−1)/(N·W) ≤ 6·B·D·D_FF/(N·C)` ⇔
`N_TP ≤ 1 + 3·D_FF·W / (2C)` (the forward is the tighter one). TP's limit
depends on the model width `D_FF`, not on the batch — the complement of DP.

### Problem (fsdp_tp_calcs)

**(a)** `6·B·D·D_FF / (N_FSDP·N_TP)`: every device computes its `1/N_TP` slice of
the hidden dimension on its `1/N_FSDP` slice of the batch.

**(b)** FSDP axis: three all-gathers of the TP-sharded weights, each
`2·D·D_FF/N_TP` bytes over `N_FSDP` devices →
`6·D·D_FF·(N_FSDP − 1)/(N_FSDP·N_TP·W)`. TP axis: one all-reduce of
`y⁽ⁱ,ʲ⁾` of `2·B·D/N_FSDP` bytes over `N_TP` devices →
`4·B·D·(N_TP − 1)/(N_TP·N_FSDP·W)`. Overlapped:
`T_comm = max( 6·D·D_FF·(N_FSDP−1)/(N_FSDP·N_TP·W),  4·B·D·(N_TP−1)/(N_TP·N_FSDP·W) )`.

**(c)** Compute is `6·B·D·D_FF/(N_FSDP·N_TP·C)`. Requiring each overlapped term
to stay below it gives two *independent* constraints:
FSDP term ⇒ `N_FSDP − 1 ≤ B·W/C`; TP term ⇒ `N_TP − 1 ≤ 3·D_FF·W/(2C)`.
They can be saturated simultaneously, so the largest compute-bound device
count is
`N = N_FSDP·N_TP ≤ (1 + B·W/C)·(1 + 3·D_FF·W/(2C)) ≈ (3/2)·B·D_FF·W²/C²`.

**(d)** Without overlap the two terms add. Using `(N−1)/N ≈ 1` and
multiplying the constraint `6·D·D_FF/(N_TP·W) + 4·B·D/(N_FSDP·W) ≤
6·B·D·D_FF/(N_FSDP·N_TP·C)` by `N_FSDP·N_TP·W`:
`6·D·D_FF·N_FSDP + 4·B·D·N_TP ≤ 6·B·D·D_FF·W/C =: K`.
Maximising the product `N_FSDP·N_TP` under a linear budget puts half the budget
on each term (AM–GM): `N_FSDP = K/(12·D·D_FF) = B·W/(2C)`,
`N_TP = K/(8·B·D) = 3·D_FF·W/(4C)`, hence
`N ≤ K²/(4·6·D·D_FF·4·B·D) = (3/8)·B·D_FF·W²/C²` — exactly ¼ of the overlapped
bound in (c).

---

## 9 Leaderboard **[GPU – not measured]**

[`cs336_systems/leaderboard.py`](../cs336_systems/leaderboard.py) reproduces the
handout's `do_bench` harness for the 8 B config (batch 2 × 32 768, bf16,
causal) on all visible GPUs and stacks:

1. our Triton FlashAttention-2 (forward + two-pass backward, causal early exit)
   swapped into the basics model (`cs336_systems.flash_patch`), run in the
   autocast dtype;
2. FSDP over the GPUs with `compute_dtype=bf16` (bf16 communication and
   compute, fp32 master weights and AdamW) — 8 B parameters × 16 B = 128 GB of
   training state split across two 192 GB B200s;
3. a chunked fused LM-head + cross-entropy (`cs336_systems.fused_ce`, sharded
   by FSDP like any layer) so the `2 × 32768 × 151936` logits (19 GiB in bf16,
   38 GiB as fp32 loss inputs) are never materialised;
4. per-segment activation checkpointing (`--segment-size 2` by default) and
   optional `torch.compile` of the blocks.

Everything above is tested for correctness on CPU (`tests/test_extra_model.py`,
`tests/test_extra_distributed.py::test_fsdp_with_fused_lm_head_loss`) and a
tiny end-to-end run of the harness is part of the CPU suite; the actual timing
requires two B200s (`uv run python -m cs336_systems.leaderboard --world-size 2`).
