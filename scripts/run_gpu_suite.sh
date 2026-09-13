#!/usr/bin/env bash
# Every GPU-only experiment from the handout, in order. Meant for a Linux box with
# >= 2 NVIDIA GPUs (the handout assumes B200s), CUDA, nsys on PATH, and `uv`.
# Usage: scripts/run_gpu_suite.sh [results_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/gpu}
mkdir -p "$OUT"
NGPU=$(nvidia-smi -L | wc -l)
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/suite.log"; }

log "== tests (incl. Triton) =="
uv run pytest tests -q --junitxml="$OUT/test_results.xml" | tee -a "$OUT/suite.log"

log "== §2.1.3 benchmarking_script: all sizes, fwd / fwd+bwd / train, 5 warmup, 10 steps =="
rm -f "$OUT/e2e.jsonl"
for size in small medium large xl; do
  for mode in forward forward_backward train; do
    uv run python -m cs336_systems.benchmark --size $size --mode $mode --warmup 5 --steps 10 --json "$OUT/e2e.jsonl" || true
  done
done
for w in 0 1 2; do
  uv run python -m cs336_systems.benchmark --size small --mode forward_backward --warmup $w --steps 10 --json "$OUT/e2e_warmup.jsonl"
done

log "== §2.1.4 nsys_profile: two sizes x three context lengths, forward / fwd+bwd / train =="
for size in small large; do
  for ctx in 256 512 1024; do
    for mode in forward forward_backward train; do
      uv run nsys profile --trace=cuda,cudnn,cublas,nvtx --pytorch=functions-trace,autograd-shapes-nvtx \
        --capture-range=nvtx --capture-range-end=stop -p "step_0" \
        -o "$OUT/nsys_${size}_${ctx}_${mode}" --force-overwrite=true \
        python -m cs336_systems.benchmark --size $size --context-length $ctx --mode $mode --nvtx --warmup 5 --steps 3 || true
      uv run nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT/nsys_${size}_${ctx}_${mode}" "$OUT/nsys_${size}_${ctx}_${mode}.nsys-rep" || true
    done
  done
done

log "== §2.1.5 benchmarking_mixed_precision (c): fp32 vs bf16 for every size =="
for size in small medium large xl; do
  for dt in fp32 bf16; do
    uv run python -m cs336_systems.benchmark --size $size --mode forward_backward --dtype $dt --json "$OUT/e2e_bf16.jsonl" || true
  done
done

log "== §2.1.6 memory_profiling: xl at ctx 128 / 2048, forward and train, fp32 and bf16 =="
for ctx in 128 2048; do
  for mode in forward train; do
    for dt in fp32 bf16; do
      uv run python -m cs336_systems.benchmark --size xl --context-length $ctx --mode $mode --dtype $dt --warmup 2 --steps 1 \
        --memory-snapshot "$OUT/mem_xl_${ctx}_${mode}_${dt}.pickle" --json "$OUT/memory.jsonl" || true
    done
  done
done
# (f) Nsight memory view: add --cuda-memory-usage=true
uv run nsys profile --trace=cuda,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cuda-memory-usage=true \
  -o "$OUT/nsys_mem_xl_train" --force-overwrite=true python -m cs336_systems.benchmark --size xl --context-length 512 --mode train --warmup 1 --steps 1 || true

log "== §3.2 gradient_checkpointing (b): xl, ctx 2048, single-level segment sizes =="
for s in 1 2 3 4 5 6 8 11 16 32; do
  uv run python -m cs336_systems.benchmark --size xl --context-length 2048 --mode train --warmup 1 --steps 2 \
    --checkpointing segments --segment-size $s --json "$OUT/checkpointing.jsonl" || true
done
uv run python -m cs336_systems.benchmark --size xl --context-length 2048 --mode train --warmup 1 --steps 2 --checkpointing recursive --json "$OUT/checkpointing.jsonl" || true
uv run python scripts/autograd_residuals.py block --size xl --context-length 2048 | tee "$OUT/residuals_xl_block.txt"
uv run python scripts/autograd_residuals.py block --size xl --context-length 2048 --compile | tee "$OUT/residuals_xl_block_compiled.txt"

log "== §4.1.1 / §4.2 attention benchmark: naive vs compiled vs flash (batch 8, no heads) =="
uv run python -m cs336_systems.attention_benchmark --impl naive compiled flash_pytorch flash_triton sdpa \
  --d 16 32 64 128 --seq 256 1024 4096 8192 16384 --batch 8 --iters 100 --csv "$OUT/attention.csv"

log "== §4.2 torch_compile (b): compiled whole model =="
for size in small medium large xl; do
  for mode in forward forward_backward train; do
    uv run python -m cs336_systems.benchmark --size $size --mode $mode --compile --json "$OUT/e2e_compile.jsonl" || true
  done
done

log "== §4.2.2 flash_benchmarking: triton.testing.do_bench, batch 1, causal, bf16 + fp32 =="
uv run python -m cs336_systems.attention_benchmark --impl naive flash_triton --causal --batch 1 --bench do_bench \
  --dtype bf16 fp32 --d 16 32 64 128 --seq 128 256 512 1024 2048 4096 8192 16384 32768 65536 --csv "$OUT/flash_benchmark.csv"

log "== §5.1 distributed_communication_single_node =="
WS=(2); [ "$NGPU" -ge 4 ] && WS+=(4); [ "$NGPU" -ge 6 ] && WS+=(6)
uv run python -m cs336_systems.distributed_benchmark --backend nccl --world-sizes "${WS[@]}" --sizes-mb 1 10 100 1000 --csv "$OUT/allreduce_nccl.csv"
uv run python -m cs336_systems.distributed_benchmark --backend gloo --world-sizes "${WS[@]}" --sizes-mb 1 10 100 1000 --csv "$OUT/allreduce_gloo.csv"

log "== §5.2-§5.3 DDP benchmarks: xl, 1 node x 2 GPUs =="
rm -f "$OUT/ddp.jsonl"
uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy naive flat overlap bucketed --json "$OUT/ddp.jsonl"
for impl in naive overlap; do
  uv run nsys profile --trace=cuda,nvtx,nccl -o "$OUT/nsys_ddp_${impl}" --force-overwrite=true \
    python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy $impl --warmup 1 --steps 2 || true
done

log "== §6 optimizer_state_sharding_accounting =="
uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy overlap --json "$OUT/sharded_opt.jsonl"
uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy overlap --sharded-optimizer --json "$OUT/sharded_opt.jsonl"

log "== §7 fsdp_accounting =="
uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy fsdp --json "$OUT/fsdp.jsonl"
uv run python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy fsdp --dtype bf16 --json "$OUT/fsdp.jsonl"
uv run nsys profile --trace=cuda,nvtx,nccl -o "$OUT/nsys_fsdp" --force-overwrite=true \
  python -m cs336_systems.ddp_benchmark --size xl --world-size 2 --strategy fsdp --warmup 1 --steps 2 || true

log "== §9 leaderboard (2 GPUs) =="
uv run python -m cs336_systems.leaderboard --world-size 2 --rep 30000 --warmup 10000 | tee "$OUT/leaderboard.txt" || true

log "== done: run scripts/make_tables.py $OUT to render the writeup tables =="
