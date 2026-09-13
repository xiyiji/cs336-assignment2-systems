"""Swap the basics model's attention for FlashAttention-2.

``cs336_basics.model.CausalMultiHeadSelfAttention`` calls the module-level
``scaled_dot_product_attention(Q, K, V, mask)`` with ``Q, K, V`` of shape
``(batch, heads, seq, d_head)`` and a causal boolean mask. Our autograd
functions accept arbitrary leading dims, so the swap is a thin adapter that
maps ``mask is not None`` to ``is_causal=True`` (the basics model only ever
builds causal masks).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

import cs336_basics.model as basics_model
from cs336_systems.flash_attention import flash_attention_pytorch

_ORIGINAL = basics_model.scaled_dot_product_attention


def _make_adapter(impl):
    def scaled_dot_product_attention(Q, K, V, mask=None):
        # autograd.Functions are not autocast-aware: under autocast, run the kernel in the
        # autocast dtype (RoPE's fp32 buffers otherwise promote Q/K back to fp32).
        dev = Q.device.type
        if torch.is_autocast_enabled(dev):
            dt = torch.get_autocast_dtype(dev)
            Q, K, V = Q.to(dt), K.to(dt), V.to(dt)
        return impl(Q, K, V, mask is not None)

    scaled_dot_product_attention.__wrapped_flash_impl__ = impl
    return scaled_dot_product_attention


def patch_basics_attention(impl: str = "triton") -> None:
    """``impl``: ``"triton"`` (Linux+CUDA), ``"pytorch"`` (tiled autograd.Function) or ``"original"``."""
    if impl == "original":
        basics_model.scaled_dot_product_attention = _ORIGINAL
        return
    if impl == "triton":
        from cs336_systems.flash_triton import flash_attention_triton

        fn = flash_attention_triton
    elif impl == "pytorch":
        fn = flash_attention_pytorch
    else:
        raise ValueError(impl)
    basics_model.scaled_dot_product_attention = _make_adapter(fn)


@contextlib.contextmanager
def flash_attention(impl: str = "triton") -> Iterator[None]:
    previous = basics_model.scaled_dot_product_attention
    patch_basics_attention(impl)
    try:
        yield
    finally:
        basics_model.scaled_dot_product_attention = previous
