#!/usr/bin/env bash
# Every experiment that can run without a GPU, sequentially (timings are CPU-bound,
# so nothing else should be running). Output goes to results/cpu/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/cpu
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/suite.log"; }

log "== 1. end-to-end benchmark (§2.1.3): small model, warmup ablation, bf16, compile =="
rm -f "$OUT/e2e.jsonl"
for ctx in 128 256 512; do
  for mode in forward forward_backward train; do
    uv run python -m cs336_systems.benchmark --size small --context-length $ctx --mode $mode \
      --warmup 5 --steps 10 --json "$OUT/e2e.jsonl" | tee -a "$OUT/suite.log"
  done
done
for w in 0 1 2; do
  uv run python -m cs336_systems.benchmark --size small --context-length 256 --mode forward_backward \
    --warmup $w --steps 10 --json "$OUT/e2e_warmup.jsonl" | tee -a "$OUT/suite.log"
done
uv run python -m cs336_systems.benchmark --size small --context-length 256 --mode forward_backward \
  --warmup 5 --steps 10 --dtype bf16 --json "$OUT/e2e_bf16.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.benchmark --size medium --context-length 128 --mode forward_backward \
  --warmup 2 --steps 5 --json "$OUT/e2e.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.benchmark --size medium --context-length 128 --mode forward_backward \
  --warmup 2 --steps 5 --dtype bf16 --json "$OUT/e2e_bf16.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.benchmark --size small --context-length 128 --mode forward_backward \
  --warmup 5 --steps 10 --json "$OUT/e2e_compile.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.benchmark --size small --context-length 128 --mode forward_backward \
  --warmup 5 --steps 10 --compile --json "$OUT/e2e_compile.jsonl" | tee -a "$OUT/suite.log"

log "== 2. mixed precision (§2.1.5) =="
uv run python -m cs336_systems.mixed_precision accumulation | tee "$OUT/mixed_precision_accumulation.txt"
uv run python -m cs336_systems.mixed_precision toy --dtype fp16 | tee "$OUT/toy_autocast_fp16.txt"
uv run python -m cs336_systems.mixed_precision toy --dtype bf16 | tee "$OUT/toy_autocast_bf16.txt"

log "== 3. autograd residuals + checkpointing (§3) =="
uv run python scripts/autograd_residuals.py rmsnorm 2>&1 | tee "$OUT/residuals_rmsnorm.txt"
uv run python scripts/autograd_residuals.py block --size xl --context-length 2048 2>&1 | tee "$OUT/residuals_xl_block.txt"
uv run python scripts/autograd_residuals.py sweep --size small --context-length 512 --num-layers 12 2>&1 | tee "$OUT/checkpoint_sweep_small.txt"

log "== 4. attention benchmark (§4.1.1, §4.2) =="
uv run python -m cs336_systems.attention_benchmark --impl naive compiled flash_pytorch \
  --d 16 32 64 128 --seq 256 1024 4096 --batch 8 --warmup 2 --iters 5 --csv "$OUT/attention.csv" | tee -a "$OUT/suite.log"

log "== 5. all-reduce benchmark (§5.1) =="
uv run python -m cs336_systems.distributed_benchmark --backend gloo --world-sizes 2 4 6 \
  --sizes-mb 1 10 100 1000 --warmup 5 --iters 10 --csv "$OUT/allreduce_gloo.csv" | tee -a "$OUT/suite.log"

log "== 6. DDP / sharded optimizer / FSDP training benchmark (§5.2-§7) =="
rm -f "$OUT/ddp.jsonl"
uv run python -m cs336_systems.ddp_benchmark --size small --context-length 128 --world-size 2 --backend gloo \
  --warmup 2 --steps 5 --strategy naive flat overlap bucketed fsdp --json "$OUT/ddp.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.ddp_benchmark --size small --context-length 128 --world-size 2 --backend gloo \
  --warmup 2 --steps 5 --strategy overlap --sharded-optimizer --json "$OUT/ddp.jsonl" | tee -a "$OUT/suite.log"

log "== done =="
