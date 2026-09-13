"""Activation (gradient) checkpointing utilities (handout §3).

* :func:`measure_saved_bytes` counts autograd residual bytes with
  ``saved_tensors_hooks`` (the technique shown in §3.1).
* :class:`CheckpointedTransformerLM` re-runs a ``BasicsTransformerLM`` with
  the transformer blocks grouped into segments of ``segment_size`` blocks;
  each segment is wrapped in ``torch.utils.checkpoint.checkpoint``
  (``use_reentrant=False``). ``segment_size=None`` disables checkpointing.
* :func:`recursive_checkpoint` nests checkpoints binary-tree style, giving
  ``O(log N)`` peak activation memory for ``O(N log N)`` compute (§3.2 (a)).
* :func:`optimal_segment_size` gives the analytic optimum for a single level
  of checkpointing (§3.2 (b)).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def measure_saved_bytes(fn: Callable[[], Tensor], skip_parameters: bool = True) -> tuple[int, list[tuple]]:
    """Run ``fn`` under ``saved_tensors_hooks`` and return (total bytes, list of (shape, dtype))."""
    total = 0
    records: list[tuple] = []

    def pack(t: Tensor):
        nonlocal total
        if skip_parameters and isinstance(t, nn.Parameter):
            return t
        total += t.numel() * t.element_size()
        records.append((tuple(t.shape), t.dtype))
        return t

    def unpack(t: Tensor):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        fn()
    return total, records


def _run_blocks(blocks: Sequence[nn.Module], x: Tensor) -> Tensor:
    for block in blocks:
        x = block(x)
    return x


def recursive_checkpoint(blocks: Sequence[nn.Module], x: Tensor, leaf_size: int = 1) -> Tensor:
    """Binary-recursive checkpointing: the first half of the blocks is checkpointed as one
    unit and recursed into only during backward. Peak activation memory is O(log N)."""
    if len(blocks) <= leaf_size:
        return _run_blocks(blocks, x)
    mid = len(blocks) // 2

    def first_half(inp):
        return recursive_checkpoint(blocks[:mid], inp, leaf_size)

    x = checkpoint(first_half, x, use_reentrant=False)
    return recursive_checkpoint(blocks[mid:], x, leaf_size)


def segmented_checkpoint(blocks: Sequence[nn.Module], x: Tensor, segment_size: int) -> Tensor:
    """One level of checkpointing: every ``segment_size`` consecutive blocks form one checkpoint."""
    for start in range(0, len(blocks), segment_size):
        segment = blocks[start : start + segment_size]

        def run_segment(inp, segment=segment):
            return _run_blocks(segment, inp)

        x = checkpoint(run_segment, x, use_reentrant=False)
    return x


class CheckpointedTransformerLM(nn.Module):
    """Wraps a ``BasicsTransformerLM`` and checkpoints its blocks.

    ``strategy``: ``"none"`` | ``"segments"`` (needs ``segment_size``) | ``"recursive"``.
    """

    def __init__(self, model: nn.Module, strategy: str = "segments", segment_size: int | None = None):
        super().__init__()
        self.model = model
        self.strategy = strategy
        if strategy == "segments" and segment_size is None:
            segment_size = max(1, int(round(math.sqrt(len(model.layers)))))
        self.segment_size = segment_size

    def forward(self, x: Tensor) -> Tensor:
        m = self.model
        h = m.token_embeddings(x)
        blocks = list(m.layers)
        if self.strategy == "none" or not torch.is_grad_enabled():
            h = _run_blocks(blocks, h)
        elif self.strategy == "segments":
            h = segmented_checkpoint(blocks, h, self.segment_size)
        elif self.strategy == "recursive":
            h = recursive_checkpoint(blocks, h)
        else:
            raise ValueError(f"unknown checkpointing strategy {self.strategy!r}")
        return m.lm_head(m.ln_final(h))


def optimal_segment_size(num_blocks: int, block_residual_bytes: float, checkpoint_bytes: float) -> int:
    """Single-level checkpointing peak ≈ (N/s)·C + s·R.  Minimised at s = sqrt(N·C/R)."""
    s = math.sqrt(num_blocks * checkpoint_bytes / block_residual_bytes)
    candidates = sorted({max(1, math.floor(s)), max(1, math.ceil(s)), 1, num_blocks})
    return min(candidates, key=lambda k: math.ceil(num_blocks / k) * checkpoint_bytes + k * block_residual_bytes)


def peak_activation_bytes(num_blocks: int, segment_size: int, block_residual_bytes: float, checkpoint_bytes: float) -> float:
    """Model of peak activation memory for single-level checkpointing with segment size ``s``:
    all ``ceil(N/s)`` checkpoint inputs plus one materialised segment during backward."""
    return math.ceil(num_blocks / segment_size) * checkpoint_bytes + segment_size * block_residual_bytes
