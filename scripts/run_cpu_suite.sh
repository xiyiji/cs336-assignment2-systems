#!/usr/bin/env bash
# Every experiment that can run without a GPU, sequentially (timings are CPU-bound,
# so nothing else should be running). Output goes to results/cpu/.
#   PHASES="1 4 5" scripts/run_cpu_suite.sh     # run a subset of the numbered phases
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/cpu
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PHASES=${PHASES:-"1 2 3 4 5 6"}
want() { [[ " $PHASES " == *" $1 "* ]]; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/suite.log"; }

if want 1; then
log "== 1. end-to-end benchmark (§2.1.3): small model, warmup ablation, bf16, compile =="
rm -f "$OUT/e2e.jsonl" "$OUT/e2e_warmup.jsonl"
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
uv run python -m cs336_systems.benchmark --size medium --context-length 128 --mode forward_backward \
  --warmup 2 --steps 5 --json "$OUT/e2e.jsonl" | tee -a "$OUT/suite.log"
fi

if want 2; then
log "== 2. mixed precision (§2.1.5). NOTE: bf16 autocast on this CPU is single-threaded, so few steps =="
rm -f "$OUT/e2e_bf16.jsonl"
uv run python -m cs336_systems.benchmark --size small --context-length 128 --mode forward_backward \
  --warmup 2 --steps 3 --json "$OUT/e2e_bf16.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.benchmark --size small --context-length 128 --mode forward_backward \
  --warmup 1 --steps 2 --dtype bf16 --json "$OUT/e2e_bf16.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.mixed_precision accumulation | tee "$OUT/mixed_precision_accumulation.txt"
uv run python -m cs336_systems.mixed_precision toy --dtype fp16 | tee "$OUT/toy_autocast_fp16.txt"
uv run python -m cs336_systems.mixed_precision toy --dtype bf16 | tee "$OUT/toy_autocast_bf16.txt"
fi

if want 3; then
log "== 3. autograd residuals + checkpointing (§3) + torch.compile of the whole model (§4.2) =="
uv run python scripts/autograd_residuals.py rmsnorm 2>&1 | tee "$OUT/residuals_rmsnorm.txt"
uv run python scripts/autograd_residuals.py block --size xl --context-length 2048 2>&1 | tee "$OUT/residuals_xl_block.txt"
uv run python scripts/autograd_residuals.py sweep --size small --context-length 512 --num-layers 12 2>&1 | tee "$OUT/checkpoint_sweep_small.txt"
rm -f "$OUT/e2e_compile.jsonl"
uv run python -m cs336_systems.benchmark --size small --context-length 128 --mode forward_backward \
  --warmup 3 --steps 5 --json "$OUT/e2e_compile.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.benchmark --size small --context-length 128 --mode forward_backward \
  --warmup 3 --steps 5 --compile --json "$OUT/e2e_compile.jsonl" | tee -a "$OUT/suite.log"
fi

if want 4; then
log "== 4. attention benchmark (§4.1.1, §4.2) =="
uv run python -m cs336_systems.attention_benchmark --impl naive compiled flash_pytorch \
  --d 16 32 64 128 --seq 256 1024 4096 --batch 8 --warmup 2 --iters 5 --csv "$OUT/attention.csv" | tee -a "$OUT/suite.log"
fi

if want 5; then
log "== 5. all-reduce benchmark (§5.1) =="
uv run python -m cs336_systems.distributed_benchmark --backend gloo --world-sizes 2 4 6 \
  --sizes-mb 1 10 100 1000 --warmup 5 --iters 10 --csv "$OUT/allreduce_gloo.csv" | tee -a "$OUT/suite.log"
fi

if want 6; then
log "== 6. DDP / sharded optimizer / FSDP training benchmark (§5.2-§7) =="
rm -f "$OUT/ddp.jsonl"
uv run python -m cs336_systems.ddp_benchmark --size small --context-length 128 --world-size 2 --backend gloo \
  --warmup 2 --steps 5 --strategy naive flat overlap bucketed fsdp --json "$OUT/ddp.jsonl" | tee -a "$OUT/suite.log"
uv run python -m cs336_systems.ddp_benchmark --size small --context-length 128 --world-size 2 --backend gloo \
  --warmup 2 --steps 5 --strategy overlap --sharded-optimizer --json "$OUT/ddp.jsonl" | tee -a "$OUT/suite.log"
fi

log "== done =="
