"""FlashAttention-2 implemented with plain PyTorch ops (handout §4.2.2).

Two pieces live here:

* :class:`FlashAttentionPytorch` – a ``torch.autograd.Function`` whose forward
  follows Algorithm 1 tile by tile (online softmax, never materialising the
  full ``N_q x N_k`` score matrix) and whose backward uses the recomputation
  formulation of Eq. 13-19 (optionally ``torch.compile``d).
* :func:`flash_backward` – the backward kernel shared with the Triton
  ``autograd.Function`` as its fallback.

The Triton implementation lives in :mod:`cs336_systems.flash_triton`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import torch
from torch import Tensor

# Value added to masked-out attention scores (handout §4.2.2 (c)).
MASK_VALUE = -1.0e6


def _flatten_batch(x: Tensor) -> Tensor:
    """(..., n, d) -> (B, n, d)."""
    return x.reshape(-1, x.shape[-2], x.shape[-1])


def _common_dtype(*tensors: Tensor) -> torch.dtype:
    """Under ``torch.autocast`` Q/K (after RoPE) may be fp32 while V is bf16; pick one dtype."""
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return dtype


def causal_mask(n_queries: int, n_keys: int, device=None) -> Tensor:
    q = torch.arange(n_queries, device=device)[:, None]
    k = torch.arange(n_keys, device=device)[None, :]
    return q >= k


# --------------------------------------------------------------------------- #
# Backward (Eq. 13-19)
# --------------------------------------------------------------------------- #
def flash_backward(
    Q: Tensor, K: Tensor, V: Tensor, O: Tensor, dO: Tensor, L: Tensor, is_causal: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute ``P`` from ``L`` and return ``(dQ, dK, dV)``.

    All tensors are ``(B, n, d)`` except ``L`` which is ``(B, n_q)``.
    ``D = rowsum(O * dO)`` replaces the softmax Jacobian (Eq. 17). Matmuls run in
    the input dtype (bf16 under autocast); ``P``, ``D`` and ``dS`` are kept in fp32
    and cast back right before each matmul, mirroring the Triton kernels.
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-1, -2)).float() * scale
    if is_causal:
        S = torch.where(causal_mask(Q.shape[-2], K.shape[-2], S.device), S, MASK_VALUE)
    P = torch.exp(S - L[..., None])
    dV = torch.matmul(P.to(dO.dtype).transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2)).float()
    D = (O.float() * dO.float()).sum(-1)
    dS = P * (dP - D[..., None])
    dQ = torch.matmul(dS.to(K.dtype), K) * scale
    dK = torch.matmul(dS.to(Q.dtype).transpose(-1, -2), Q) * scale
    return dQ, dK, dV


_compiled_backward: Callable | None = None


def _want_compile(device: torch.device) -> bool:
    flag = os.environ.get("CS336_COMPILE_FLASH_BACKWARD")
    if flag is not None:
        return flag not in ("0", "false", "False", "")
    return device.type == "cuda"


def get_flash_backward(device: torch.device) -> Callable:
    """Return the (possibly compiled) backward function for ``device``."""
    global _compiled_backward
    if not _want_compile(device):
        return flash_backward
    if _compiled_backward is None:
        _compiled_backward = torch.compile(flash_backward, dynamic=True)
    return _compiled_backward


# --------------------------------------------------------------------------- #
# Forward (Algorithm 1)
# --------------------------------------------------------------------------- #
def flash_forward_tiled(
    Q: Tensor, K: Tensor, V: Tensor, is_causal: bool, q_tile: int, k_tile: int
) -> tuple[Tensor, Tensor]:
    """Tiled forward pass in PyTorch. Returns ``O`` ``(B, n_q, d)`` and ``L`` ``(B, n_q)``."""
    B, n_q, d = Q.shape
    n_k = K.shape[-2]
    scale = 1.0 / math.sqrt(d)
    O = torch.empty_like(Q)
    L = torch.empty(B, n_q, device=Q.device, dtype=torch.float32)
    n_q_tiles = math.ceil(n_q / q_tile)
    n_k_tiles = math.ceil(n_k / k_tile)

    for i in range(n_q_tiles):
        q0, q1 = i * q_tile, min((i + 1) * q_tile, n_q)
        Qi = Q[:, q0:q1].float()
        Oi = torch.zeros(B, q1 - q0, d, device=Q.device, dtype=torch.float32)
        li = torch.zeros(B, q1 - q0, device=Q.device, dtype=torch.float32)
        mi = torch.full((B, q1 - q0), float("-inf"), device=Q.device, dtype=torch.float32)
        q_idx = torch.arange(q0, q1, device=Q.device)[:, None]
        # With causal masking, key tiles entirely above the diagonal are all-zero: skip them.
        last_k_tile = n_k_tiles if not is_causal else math.ceil(min(q1, n_k) / k_tile)
        for j in range(last_k_tile):
            k0, k1 = j * k_tile, min((j + 1) * k_tile, n_k)
            Kj = K[:, k0:k1].float()
            Vj = V[:, k0:k1].float()
            S = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale  # (B, Bq, Bk)
            if is_causal:
                k_idx = torch.arange(k0, k1, device=Q.device)[None, :]
                S = torch.where(q_idx >= k_idx, S, S + MASK_VALUE)
            m_new = torch.maximum(mi, S.amax(-1))
            P = torch.exp(S - m_new[..., None])
            alpha = torch.exp(mi - m_new)
            li = alpha * li + P.sum(-1)
            Oi = alpha[..., None] * Oi + torch.matmul(P.to(Vj.dtype), Vj)
            mi = m_new
        O[:, q0:q1] = (Oi / li[..., None]).to(O.dtype)
        L[:, q0:q1] = mi + torch.log(li)
    return O, L


class FlashAttentionPytorch(torch.autograd.Function):
    """FlashAttention-2 forward in tiled PyTorch; backward via recomputation.

    ``forward(ctx, Q, K, V, is_causal=False)``; inputs are ``(..., n, d)``.
    Saves ``L, Q, K, V, O`` for backward. ``L`` has shape ``(..., n_q)`` and is
    the only saved tensor of that shape (the tests rely on this).
    """

    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        batch_shape = Q.shape[:-2]
        ctx.input_dtypes = (Q.dtype, K.dtype, V.dtype)
        dtype = _common_dtype(Q, K, V)
        Q, K, V = (t.to(dtype) for t in (Q, K, V))
        Qf, Kf, Vf = (_flatten_batch(t) for t in (Q, K, V))
        q_tile = min(FlashAttentionPytorch.Q_TILE_SIZE, Qf.shape[-2])
        k_tile = min(FlashAttentionPytorch.K_TILE_SIZE, Kf.shape[-2])
        O, L = flash_forward_tiled(Qf, Kf, Vf, is_causal, q_tile, k_tile)
        O = O.reshape(Q.shape)
        L = L.reshape(*batch_shape, Q.shape[-2])
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        n_q = Q.shape[-2]
        bwd = get_flash_backward(Q.device)
        dQ, dK, dV = bwd(
            _flatten_batch(Q),
            _flatten_batch(K),
            _flatten_batch(V),
            _flatten_batch(O),
            _flatten_batch(dO.to(Q.dtype)),
            L.reshape(-1, n_q),
            ctx.is_causal,
        )
        dq_t, dk_t, dv_t = ctx.input_dtypes
        return dQ.reshape(Q.shape).to(dq_t), dK.reshape(K.shape).to(dk_t), dV.reshape(V.shape).to(dv_t), None


def flash_attention_pytorch(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
    return FlashAttentionPytorch.apply(Q, K, V, is_causal)


# --------------------------------------------------------------------------- #
# Plain (non-flash) attention used as the benchmarking baseline
# --------------------------------------------------------------------------- #
def naive_attention(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
    """Standard attention that materialises the full score matrix (handout Eq. 1)."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if is_causal:
        S = torch.where(causal_mask(Q.shape[-2], K.shape[-2], S.device), S, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)
