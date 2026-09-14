"""Fully-sharded data parallel (handout §7).

Design
------
* Every ``Linear`` / ``Embedding`` weight (from ``cs336_basics`` or ``torch.nn``,
  or any module with a class attribute ``fsdp_shard_weight = True``) is flattened, zero-padded to a multiple of ``world_size`` and split into
  equal 1-D shards. The *same* ``nn.Parameter`` object keeps living in the
  module, with ``param.data`` replaced by the local shard, so ``named_parameters``
  keeps its names and any optimizer sees only the shard.
* Norm weights (anything that is not Linear/Embedding) stay replicated; their
  gradients are averaged with an asynchronous all-reduce.
* Forward: a pre-forward hook all-gathers the shard (in ``compute_dtype`` if
  given, so the communication itself is low precision) and temporarily points
  ``param.data`` at the full tensor; the post-forward hook restores the shard.
  Autograd saved the *Parameter* itself (not a copy), so the full weight is
  freed after the layer runs.
* Prefetch: the first forward records execution order; afterwards the gather
  for layer ``k`` is launched when layer ``k-2`` finishes its forward (and the
  first two gathers at the start of forward), exactly as the handout asks.
* Backward: a hook on each layer's output re-gathers the weight just before the
  layer's backward node runs (and prefetches the layer two positions earlier
  in forward order). When the full-size gradient has been accumulated, a
  ``post_accumulate_grad_hook`` casts it to fp32, launches an asynchronous
  ``reduce_scatter`` into a shard-sized buffer and frees the full gradient.
* :meth:`FSDP.finish_gradient_synchronization` waits for all communication and
  installs the fp32 shard gradients, so a plain optimizer (SGD, AdamW from
  assignment 1, ...) can step on the shards.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.distributed as dist
from torch import Tensor, nn

from cs336_systems.ddp import _broadcast_module


def _shardable_types() -> tuple[type, ...]:
    types: list[type] = [nn.Linear, nn.Embedding]
    try:
        from cs336_basics.model import Embedding, Linear

        types += [Linear, Embedding]
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


class _ShardUnit:
    """Book-keeping for one sharded parameter."""

    def __init__(self, name: str, module: nn.Module, param: nn.Parameter, world_size: int, rank: int):
        self.name = name
        self.module = module
        self.param = param
        self.full_shape = tuple(param.shape)
        self.numel = param.numel()
        self.padded_numel = math.ceil(self.numel / world_size) * world_size
        self.shard_numel = self.padded_numel // world_size
        flat = torch.zeros(self.padded_numel, dtype=param.dtype, device=param.device)
        flat[: self.numel].copy_(param.data.reshape(-1))
        self.shard: Tensor = flat[rank * self.shard_numel : (rank + 1) * self.shard_numel].clone()
        param.data = self.shard  # optimizer + named_parameters now see the shard
        self.order_index: int | None = None
        # Transient state
        self.gather_buf: Tensor | None = None
        self.gather_handle: dist.Work | None = None
        self.full: Tensor | None = None
        self.grad_buf: Tensor | None = None
        self.grad_handle: dist.Work | None = None
        self.output_hook_handles: list = []

    def compute_dtype_for(self, compute_dtype: torch.dtype | None) -> torch.dtype:
        return compute_dtype if compute_dtype is not None else self.param.dtype


def _default_sync_comm() -> bool:
    """gloo's libuv transport (macOS) corrupts its stream with many in-flight async
    collectives ("Unexpected opcode"); serialise communication there by default.
    Override with CS336_FSDP_SYNC_COMM=0/1."""
    env = os.environ.get("CS336_FSDP_SYNC_COMM")
    if env is not None:
        return env not in ("0", "false", "False", "")
    return sys.platform == "darwin" and dist.get_backend() == "gloo"


class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        prefetch_distance: int = 2,
        sync_comm: bool | None = None,
    ):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialised before wrapping a module")
        self.module = module
        self.compute_dtype = compute_dtype
        self.prefetch_distance = prefetch_distance
        self.sync_comm = _default_sync_comm() if sync_comm is None else sync_comm
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        _broadcast_module(module)

        self._units: list[_ShardUnit] = []
        self._unit_by_param: dict[int, _ShardUnit] = {}
        shardable = _shardable_types()
        for mod_name, mod in module.named_modules():
            if not (isinstance(mod, shardable) or getattr(mod, "fsdp_shard_weight", False)):
                continue
            weight = getattr(mod, "weight", None)
            if not isinstance(weight, nn.Parameter) or id(weight) in self._unit_by_param:
                continue
            name = f"{mod_name}.weight" if mod_name else "weight"
            unit = _ShardUnit(name, mod, weight, self.world_size, self.rank)
            self._units.append(unit)
            self._unit_by_param[id(weight)] = unit
            mod.register_forward_pre_hook(self._make_pre_forward(unit))
            mod.register_forward_hook(self._make_post_forward(unit))
            if weight.requires_grad:
                weight.register_post_accumulate_grad_hook(self._make_on_grad(unit))

        self._replicated: list[nn.Parameter] = [
            p for p in module.parameters() if id(p) not in self._unit_by_param and p.requires_grad
        ]
        self._replicated_handles: list[dist.Work] = []
        for p in self._replicated:
            p.register_post_accumulate_grad_hook(self._on_replicated_grad)

        self._forward_order: list[_ShardUnit] = []
        self._order_known = False

    # ------------------------------------------------------------------ #
    # Communication helpers
    # ------------------------------------------------------------------ #
    def _launch_gather(self, unit: _ShardUnit) -> None:
        if unit.full is not None or unit.gather_handle is not None:
            return
        dtype = unit.compute_dtype_for(self.compute_dtype)
        shard = unit.shard.to(dtype)  # cast *before* communicating (saves bandwidth)
        unit.gather_buf = torch.empty(unit.padded_numel, dtype=dtype, device=shard.device)
        unit.gather_handle = dist.all_gather_into_tensor(unit.gather_buf, shard, async_op=True)
        if self.sync_comm:
            self._wait_gather(unit)

    def _wait_gather(self, unit: _ShardUnit) -> Tensor:
        if unit.full is None:
            if unit.gather_handle is None:
                self._launch_gather(unit)
                if unit.full is not None:  # sync_comm: the launch already waited
                    return unit.full
            unit.gather_handle.wait()
            unit.gather_handle = None
            unit.full = unit.gather_buf[: unit.numel].view(unit.full_shape)
            unit.gather_buf = None
        return unit.full

    def _free_full(self, unit: _ShardUnit) -> None:
        unit.param.data = unit.shard
        unit.full = None

    def _prefetch(self, index: int) -> None:
        if self._order_known and 0 <= index < len(self._forward_order):
            self._launch_gather(self._forward_order[index])

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def _make_pre_forward(self, unit: _ShardUnit):
        def hook(mod, inputs):
            if not self._order_known:
                unit.order_index = len(self._forward_order)
                self._forward_order.append(unit)
            unit.param.data = self._wait_gather(unit)

        return hook

    def _make_post_forward(self, unit: _ShardUnit):
        def hook(mod, inputs, output):
            self._free_full(unit)
            if self._order_known and unit.order_index is not None:
                # "only start gathering after the layer two before the current one has completed"
                self._prefetch(unit.order_index + self.prefetch_distance)
            # Re-gather right before this layer's backward runs.
            outputs = output if isinstance(output, (tuple, list)) else (output,)
            for out in outputs:
                if torch.is_tensor(out) and out.requires_grad:
                    out.register_hook(self._make_pre_backward(unit))
                    break

        return hook

    def _make_pre_backward(self, unit: _ShardUnit):
        def hook(grad_output):
            unit.param.data = self._wait_gather(unit)
            if unit.order_index is not None:
                self._prefetch(unit.order_index - self.prefetch_distance)
            return None

        return hook

    def _make_on_grad(self, unit: _ShardUnit):
        def hook(param: Tensor):
            full_grad = param.grad
            param.grad = None
            self._free_full(unit)
            if full_grad is None:
                return
            flat = torch.zeros(unit.padded_numel, dtype=unit.shard.dtype, device=unit.shard.device)
            flat[: unit.numel].copy_(full_grad.reshape(-1))
            flat.div_(self.world_size)
            del full_grad
            unit.grad_buf = torch.empty(unit.shard_numel, dtype=unit.shard.dtype, device=unit.shard.device)
            unit.grad_handle = dist.reduce_scatter_tensor(unit.grad_buf, flat, op=dist.ReduceOp.SUM, async_op=True)
            if self.sync_comm:
                unit.grad_handle.wait()
                unit.grad_handle = None
                unit.param.grad = unit.grad_buf
                unit.grad_buf = None

        return hook

    def _on_replicated_grad(self, param: Tensor) -> None:
        if param.grad is None:
            return
        param.grad.div_(self.world_size)
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        if self.sync_comm:
            handle.wait()
        else:
            self._replicated_handles.append(handle)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(self, *inputs, **kwargs):
        if self._order_known:
            for i in range(min(self.prefetch_distance, len(self._forward_order))):
                self._launch_gather(self._forward_order[i])
        output = self.module(*inputs, **kwargs)
        if not self._order_known:
            self._order_known = True
        if torch.is_grad_enabled():
            # Backward will visit layers in reverse order: prefetch the last two.
            n = len(self._forward_order)
            for i in range(n - 1, max(n - 1 - self.prefetch_distance, -1), -1):
                self._launch_gather(self._forward_order[i])
        return output

    def finish_gradient_synchronization(self) -> None:
        for unit in self._units:
            if unit.grad_handle is not None:
                unit.grad_handle.wait()
                unit.grad_handle = None
                unit.param.grad = unit.grad_buf
                unit.grad_buf = None
            # Drop any prefetched-but-unused gathers (e.g. forward under no_grad).
            if unit.gather_handle is not None:
                unit.gather_handle.wait()
                unit.gather_handle = None
                unit.gather_buf = None
            unit.full = None
            unit.param.data = unit.shard
        for h in self._replicated_handles:
            h.wait()
        self._replicated_handles.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, Tensor]:
        """All-gather every shard; returns ``{name: full fp32 tensor}`` for all parameters."""
        result: dict[str, Tensor] = {}
        for name, p in self.module.named_parameters():
            unit = self._unit_by_param.get(id(p))
            if unit is None:
                result[name] = p.data
                continue
            buf = torch.empty(unit.padded_numel, dtype=unit.shard.dtype, device=unit.shard.device)
            dist.all_gather_into_tensor(buf, unit.shard)
            result[name] = buf[: unit.numel].view(unit.full_shape)
        return result

    def sharded_parameter_names(self) -> list[str]:
        return [u.name for u in self._units]

    def local_param_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.module.parameters())
