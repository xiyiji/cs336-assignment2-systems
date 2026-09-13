"""Optimizer state sharding (handout §6, a simplified ZeRO stage 1).

Each rank owns roughly ``1/world_size`` of the parameters and keeps optimizer
state only for those. After each ``step()`` every parameter is broadcast from
its owner so the model stays replicated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Wrap ``optimizer_cls`` so that each rank only updates its parameter shard.

    Parameters are assigned greedily to the rank with the smallest total
    element count so far (ties → lowest rank). The assignment is a pure
    function of the parameter order, so all ranks agree without communication.
    """

    def __init__(self, params: Iterable, optimizer_cls: type[Optimizer], **kwargs: Any):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialised")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self._rank_load = [0] * self.world_size
        self._owner: dict[int, int] = {}  # id(param) -> rank
        self._pending_local_groups: list[dict[str, Any]] = []
        self.local_optimizer: Optimizer | None = None
        # The super constructor calls add_param_group for every group.
        super().__init__(params, defaults=dict(kwargs))

    # ------------------------------------------------------------------ #
    def _assign(self, p: torch.Tensor) -> int:
        owner = self._owner.get(id(p))
        if owner is None:
            owner = min(range(self.world_size), key=lambda r: (self._rank_load[r], r))
            self._rank_load[owner] += p.numel()
            self._owner[id(p)] = owner
        return owner

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)  # keeps the full group in self.param_groups
        group = self.param_groups[-1]
        local_params = [p for p in group["params"] if self._assign(p) == self.rank]
        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        if not local_params:
            return  # this rank owns nothing in the group
        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
        else:
            self.local_optimizer.add_param_group(local_group)

    def owner_of(self, p: torch.Tensor) -> int:
        return self._owner[id(p)]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _synchronize_parameters(self) -> None:
        handles = []
        for group in self.param_groups:
            for p in group["params"]:
                handles.append(dist.broadcast(p.data, src=self._owner[id(p)], async_op=True))
        for h in handles:
            h.wait()

    def step(self, closure: Callable | None = None, **kwargs: Any):
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure, **kwargs) if closure is not None else self.local_optimizer.step(**kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._synchronize_parameters()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Gradients exist for every parameter (backward ran on the full model),
        # so clear all of them, not only the local shard.
        super().zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        """Local shard's state (plus ownership map). Not a full optimizer state."""
        local = self.local_optimizer.state_dict() if self.local_optimizer is not None else {}
        return {"rank": self.rank, "world_size": self.world_size, "local": local}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.local_optimizer is not None and state_dict.get("local"):
            self.local_optimizer.load_state_dict(state_dict["local"])

    def local_state_bytes(self) -> int:
        """Bytes of optimizer state held on this rank (for memory accounting)."""
        total = 0
        if self.local_optimizer is not None:
            for st in self.local_optimizer.state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        total += v.numel() * v.element_size()
        return total
