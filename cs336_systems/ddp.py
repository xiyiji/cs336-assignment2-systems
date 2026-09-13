"""Distributed data parallel containers (handout §5.2-§5.3).

Four variants share one interface (``forward`` + ``finish_gradient_synchronization``):

* :class:`NaiveDDP` – after backward, one synchronous all-reduce per parameter.
* :class:`FlatDDP` – after backward, flatten every gradient into one tensor and
  issue a single all-reduce (§5.3.1).
* :class:`DDPOverlapIndividual` – asynchronous all-reduce of each parameter's
  gradient as soon as it is accumulated, overlapping with the rest of the
  backward pass (§5.3.2). This is the graded implementation.
* :class:`DDPBucketed` – like the above, but gradients are grouped into
  buckets of at most ``bucket_size_mb`` and each bucket is all-reduced as one
  flat tensor once all of its gradients are ready (Spring 2025 bonus).

All variants broadcast rank 0's parameters and buffers at construction so every
rank starts from identical weights, and average (not sum) gradients.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def _broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Make every rank's parameters and buffers identical to ``src``."""
    handles = []
    for tensor in list(module.parameters()) + list(module.buffers()):
        handles.append(dist.broadcast(tensor.data, src=src, async_op=True))
    for h in handles:
        h.wait()


def _grad_params(module: nn.Module) -> list[nn.Parameter]:
    """Unique parameters that require grad (tied weights appear once)."""
    return [p for p in module.parameters() if p.requires_grad]


class _DDPBase(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialised before wrapping a module")
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        _broadcast_module(module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class NaiveDDP(_DDPBase):
    """All-reduce each gradient individually after ``backward()`` (synchronous)."""

    def finish_gradient_synchronization(self) -> None:
        for p in _grad_params(self.module):
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)


class FlatDDP(_DDPBase):
    """Flatten all gradients and issue a single all-reduce after ``backward()``."""

    def finish_gradient_synchronization(self) -> None:
        params = [p for p in _grad_params(self.module) if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


class DDPOverlapIndividual(_DDPBase):
    """Overlap backward computation with per-parameter asynchronous all-reduces.

    A ``post_accumulate_grad_hook`` on every trainable parameter pre-divides the
    gradient by the world size and launches ``all_reduce(async_op=True)``;
    :meth:`finish_gradient_synchronization` waits on the handles.
    """

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self._handles: list[dist.Work] = []
        for p in _grad_params(module):
            p.register_post_accumulate_grad_hook(self._launch_all_reduce)

    def _launch_all_reduce(self, param: torch.Tensor) -> None:
        if param.grad is None:
            return
        param.grad.div_(self.world_size)
        self._handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))

    def finish_gradient_synchronization(self) -> None:
        for h in self._handles:
            h.wait()
        self._handles.clear()


class _Bucket:
    __slots__ = ("params", "ready", "flat", "handle")

    def __init__(self, params: list[nn.Parameter]):
        self.params = params
        self.ready = 0
        self.flat: torch.Tensor | None = None
        self.handle: dist.Work | None = None


class DDPBucketed(_DDPBase):
    """Overlapped DDP with gradient bucketing.

    Parameters are assigned to buckets in *reverse* registration order (the
    order in which gradients usually become available in backward) so the
    first bucket to fill is the one closest to the loss. When every gradient in
    a bucket has been accumulated the bucket is flattened and all-reduced
    asynchronously. ``bucket_size_mb=None`` means a single unbounded bucket.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float | None = 25.0):
        super().__init__(module)
        self.bucket_size_bytes = None if bucket_size_mb is None else int(bucket_size_mb * 1024 * 1024)
        self._buckets: list[_Bucket] = []
        self._param_to_bucket: dict[int, _Bucket] = {}
        self._build_buckets(reversed(_grad_params(module)))
        for bucket in self._buckets:
            for p in bucket.params:
                p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _build_buckets(self, params: Iterable[nn.Parameter]) -> None:
        current: list[nn.Parameter] = []
        current_bytes = 0
        for p in params:
            nbytes = p.numel() * p.element_size()
            if current and self.bucket_size_bytes is not None and current_bytes + nbytes > self.bucket_size_bytes:
                self._buckets.append(_Bucket(current))
                current, current_bytes = [], 0
            current.append(p)
            current_bytes += nbytes
        if current:
            self._buckets.append(_Bucket(current))
        for bucket in self._buckets:
            for p in bucket.params:
                self._param_to_bucket[id(p)] = bucket

    def _on_grad_ready(self, param: torch.Tensor) -> None:
        bucket = self._param_to_bucket[id(param)]
        bucket.ready += 1
        if bucket.ready == len(bucket.params):
            grads = [p.grad for p in bucket.params]
            bucket.flat = _flatten_dense_tensors(grads)
            bucket.flat.div_(self.world_size)
            bucket.handle = dist.all_reduce(bucket.flat, op=dist.ReduceOp.SUM, async_op=True)

    def finish_gradient_synchronization(self) -> None:
        for bucket in self._buckets:
            if bucket.handle is None:
                # Some parameter in this bucket received no gradient (e.g. unused
                # in this step); fall back to a synchronous reduce of what exists.
                for p in bucket.params:
                    if p.grad is not None:
                        p.grad.div_(self.world_size)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            else:
                bucket.handle.wait()
                grads = [p.grad for p in bucket.params]
                for g, synced in zip(grads, _unflatten_dense_tensors(bucket.flat, grads)):
                    g.copy_(synced)
            bucket.ready = 0
            bucket.flat = None
            bucket.handle = None

    def on_train_batch_start(self) -> None:
        """Reset bucket state (also done at the end of ``finish_gradient_synchronization``)."""
        for bucket in self._buckets:
            bucket.ready = 0
            bucket.flat = None
            bucket.handle = None


# Alias for the graded implementation.
DDP = DDPOverlapIndividual

DDP_VARIANTS = {
    "naive": NaiveDDP,
    "flat": FlatDDP,
    "overlap": DDPOverlapIndividual,
    "bucketed": DDPBucketed,
}
