"""Render benchmark outputs (JSONL / CSV under results/<dir>/) as Markdown tables.

    uv run python scripts/make_tables.py results/cpu > writeup/tables_cpu.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def md_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """columns: (key, header, format)."""
    if not rows:
        return "_no data_\n"
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _, fmt in columns:
            v = r.get(key, "")
            try:
                cells.append(format(float(v), fmt) if fmt else str(v))
            except (TypeError, ValueError):
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(results_dir: str) -> None:
    d = Path(results_dir)
    out = []

    e2e = read_jsonl(d / "e2e.jsonl")
    if e2e:
        out.append("## End-to-end benchmark (5 warmup / 10 timed unless noted)\n")
        out.append(
            md_table(
                e2e,
                [("size", "size", ""), ("context_length", "ctx", ""), ("mode", "mode", ""), ("dtype", "dtype", ""),
                 ("n_params", "params", ".3e"), ("forward_ms", "fwd ms", ".1f"), ("backward_ms", "bwd ms", ".1f"),
                 ("optimizer_ms", "opt ms", ".1f"), ("mean_ms", "step ms", ".1f"), ("std_ms", "± ms", ".1f"),
                 ("warmup", "warmup", ""), ("steps", "steps", "")],
            )
        )
    for name, title in [("e2e_warmup.jsonl", "Warm-up ablation"), ("e2e_bf16.jsonl", "fp32 vs bf16 autocast"),
                        ("e2e_compile.jsonl", "eager vs torch.compile"), ("memory.jsonl", "Memory profiling runs"),
                        ("checkpointing.jsonl", "Activation checkpointing sweep")]:
        rows = read_jsonl(d / name)
        if rows:
            out.append(f"## {title}\n")
            out.append(
                md_table(
                    rows,
                    [("size", "size", ""), ("context_length", "ctx", ""), ("mode", "mode", ""), ("dtype", "dtype", ""),
                     ("compiled", "compiled", ""), ("checkpointing", "ckpt", ""), ("warmup", "warmup", ""),
                     ("mean_ms", "step ms", ".1f"), ("std_ms", "± ms", ".1f"), ("forward_ms", "fwd ms", ".1f"),
                     ("backward_ms", "bwd ms", ".1f"), ("per_step_ms", "per-step ms", ""), ("peak_memory_mib", "peak MiB", ".0f")],
                )
            )

    for name, title in [("attention.csv", "Attention benchmark"), ("flash_benchmark.csv", "FlashAttention-2 vs PyTorch (do_bench)")]:
        rows = read_csv(d / name)
        if rows:
            out.append(f"## {title}\n")
            out.append(
                md_table(
                    rows,
                    [("impl", "impl", ""), ("dtype", "dtype", ""), ("d", "d", ""), ("seq", "seq", ""), ("fwd_ms", "fwd ms", ".3f"),
                     ("bwd_ms", "bwd ms", ".3f"), ("fwd_bwd_ms", "fwd+bwd ms", ".3f"), ("mem_before_bwd_mib", "mem before bwd MiB", ".0f"),
                     ("status", "status", "")],
                )
            )

    for name in ["allreduce_gloo.csv", "allreduce_nccl.csv"]:
        rows = read_csv(d / name)
        if rows:
            out.append(f"## All-reduce benchmark ({rows[0]['backend']})\n")
            out.append(md_table(rows, [("world_size", "procs", ""), ("size_mb", "MB", ".0f"), ("mean_ms", "ms", ".3f"), ("std_ms", "± ms", ".3f"), ("bus_bw_GBps", "bus GB/s", ".2f")]))

    for name in ["ddp.jsonl", "sharded_opt.jsonl", "fsdp.jsonl"]:
        rows = read_jsonl(d / name)
        if rows:
            out.append(f"## Distributed training ({name})\n")
            out.append(
                md_table(
                    rows,
                    [("strategy", "strategy", ""), ("sharded_optimizer", "sharded opt", ""), ("size", "size", ""), ("dtype", "dtype", ""),
                     ("world_size", "procs", ""), ("step_ms", "step ms", ".1f"), ("step_std_ms", "± ms", ".1f"), ("comm_ms", "exposed comm ms", ".1f"),
                     ("param_bytes_local", "local params B", ".3e"), ("opt_state_bytes_local", "local opt state B", ".3e"),
                     ("mem_after_init_mib", "mem after init MiB", ".0f"), ("peak_before_opt_mib", "peak before opt MiB", ".0f"),
                     ("peak_after_opt_mib", "peak after opt MiB", ".0f")],
                )
            )
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/cpu")
