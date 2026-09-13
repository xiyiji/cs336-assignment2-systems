"""FlashAttention-2 with Triton kernels (handout §4.2.2 and optional §4.2.3).

* :func:`flash_fwd_kernel` follows Algorithm 1 exactly with the launch grid
  ``(T_q, batch)`` and a single loop over key tiles. With ``is_causal`` the
  loop stops at the diagonal tile (tiles above it are all zero).
* :func:`flash_bwd_dkdv_kernel` / :func:`flash_bwd_dq_kernel` implement the
  two-pass tiled backward of Algorithm 2 (no atomics: one pass owns a key
  tile and produces ``dK, dV``; the other owns a query tile and produces ``dQ``).
* :class:`FlashAttentionTriton` is the ``autograd.Function`` wrapping them.
  ``FlashAttentionTriton.backward_impl`` selects ``"triton"`` (default) or
  ``"compiled"`` (the PyTorch recomputation backward from
  :mod:`cs336_systems.flash_attention`, as required by ``flash_backward``).

Triton is only importable on Linux; importing this module elsewhere raises
``ImportError`` from :func:`_require_triton`.  Set ``TRITON_INTERPRET=1`` to
run the kernels on CPU through the Triton interpreter (used by CI).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cs336_systems.flash_attention import MASK_VALUE, _common_dtype, _flatten_batch, get_flash_backward

try:  # pragma: no cover - exercised only where Triton is installed
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise ImportError("triton is not installed (it is Linux-only); use FlashAttentionPytorch instead")


if HAS_TRITON:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):  # fmt: skip
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        if is_causal:
            # Only key tiles up to (and including) the diagonal contribute.
            q_end = tl.minimum((query_tile_index + 1) * Q_TILE_SIZE, N_KEYS)
            n_k_tiles = tl.cdiv(q_end, K_TILE_SIZE)
        else:
            n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(n_k_tiles):
            k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
            s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE, K_TILE), fp32
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            valid = k_idx[None, :] < N_KEYS
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            s = tl.where(valid, s, s + MASK_VALUE)
            m_new = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new[:, None])
            l = alpha * l + tl.sum(p, axis=1)
            o = o * alpha[:, None]
            o = tl.dot(p.to(v.dtype), v, acc=o)
            m = m_new
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        o = o / l[:, None]
        L_val = m + tl.log(l)
        tl.store(O_block_ptr, o.to(q.dtype), boundary_check=(0,))  # O has Q's dtype
        tl.store(L_block_ptr, L_val, boundary_check=(0,))  # L is fp32

    @triton.jit
    def flash_bwd_dkdv_kernel(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
        dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_dkb, stride_dkk, stride_dkd,
        stride_dvb, stride_dvk, stride_dvd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):  # fmt: skip
        """Algorithm 2, lines 6-19: one program per (key tile, batch) computes dK_j, dV_j."""
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        if is_causal:
            # Query tiles strictly above the diagonal never attend to this key tile.
            first_q_tile = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
        else:
            first_q_tile = 0
        q_start = first_q_tile * Q_TILE_SIZE

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(q_start, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob, shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
            offsets=(q_start, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,),
            offsets=(q_start,), block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db, shape=(N_QUERIES,), strides=(stride_dq,),
            offsets=(q_start,), block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb, shape=(N_KEYS, D), strides=(stride_dkk, stride_dkd),
            offsets=(key_tile_index * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb, shape=(N_KEYS, D), strides=(stride_dvk, stride_dvd),
            offsets=(key_tile_index * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )

        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        dk = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dv = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

        n_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
        for i in range(first_q_tile, n_q_tiles):
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
            do = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
            l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

            s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE, K_TILE)
            valid = (q_idx[:, None] < N_QUERIES) & (k_idx[None, :] < N_KEYS)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            p = tl.where(valid, tl.exp(s - l[:, None]), 0.0)
            dv = tl.dot(tl.trans(p).to(do.dtype), do, acc=dv)
            dp = tl.dot(do, tl.trans(v))  # (Q_TILE, K_TILE)
            ds = p * (dp - d[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk)

            Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
            dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
            L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
            D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

        dk = dk * scale
        tl.store(dK_block_ptr, dk.to(k.dtype), boundary_check=(0,))
        tl.store(dV_block_ptr, dv.to(v.dtype), boundary_check=(0,))

    @triton.jit
    def flash_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
        dQ_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_dqb, stride_dqq, stride_dqd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):  # fmt: skip
        """Algorithm 2, lines 20-32: one program per (query tile, batch) computes dQ_i."""
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob, shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
            offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db, shape=(N_QUERIES,), strides=(stride_dq,),
            offsets=(query_tile_index * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
            offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
            offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb, shape=(N_QUERIES, D), strides=(stride_dqq, stride_dqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        do = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
        l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        dq = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        if is_causal:
            q_end = tl.minimum((query_tile_index + 1) * Q_TILE_SIZE, N_KEYS)
            n_k_tiles = tl.cdiv(q_end, K_TILE_SIZE)
        else:
            n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(n_k_tiles):
            k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = tl.dot(q, tl.trans(k)) * scale
            valid = k_idx[None, :] < N_KEYS
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            p = tl.where(valid, tl.exp(s - l[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - d[:, None])
            dq = tl.dot(ds.to(k.dtype), k, acc=dq)
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        dq = dq * scale
        tl.store(dQ_block_ptr, dq.to(q.dtype), boundary_check=(0,))


def _pick_tile(n: int, preferred: int) -> int:
    """Largest power of two ≤ ``preferred`` that is ≤ ``n`` (never below 16)."""
    t = min(preferred, n)
    t = 1 << (t.bit_length() - 1)
    return max(t, 16)


def flash_forward_triton(
    Q: Tensor, K: Tensor, V: Tensor, is_causal: bool, q_tile: int, k_tile: int
) -> tuple[Tensor, Tensor]:
    _require_triton()
    B, n_q, d = Q.shape
    n_k = K.shape[-2]
    O = torch.empty_like(Q)
    L = torch.empty(B, n_q, device=Q.device, dtype=torch.float32)
    grid = (triton.cdiv(n_q, q_tile), B)
    flash_fwd_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        n_q, n_k,
        1.0 / math.sqrt(d),
        D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=is_causal,
    )  # fmt: skip
    return O, L


def flash_backward_triton(
    Q: Tensor, K: Tensor, V: Tensor, O: Tensor, dO: Tensor, L: Tensor, is_causal: bool, q_tile: int, k_tile: int
) -> tuple[Tensor, Tensor, Tensor]:
    _require_triton()
    B, n_q, d = Q.shape
    n_k = K.shape[-2]
    dO = dO.contiguous()
    Dvec = (O.float() * dO.float()).sum(-1)  # (B, n_q), line 2 of Algorithm 2
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    scale = 1.0 / math.sqrt(d)
    common = (
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        Dvec.stride(0), Dvec.stride(1),
    )  # fmt: skip
    flash_bwd_dkdv_kernel[(triton.cdiv(n_k, k_tile), B)](
        Q, K, V, dO, L, Dvec, dK, dV,
        *common,
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        n_q, n_k, scale,
        D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=is_causal,
    )  # fmt: skip
    flash_bwd_dq_kernel[(triton.cdiv(n_q, q_tile), B)](
        Q, K, V, dO, L, Dvec, dQ,
        *common,
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        n_q, n_k, scale,
        D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=is_causal,
    )  # fmt: skip
    return dQ, dK, dV


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 whose forward is the fused Triton kernel.

    Class attributes control tiling and which backward is used::

        FlashAttentionTriton.Q_TILE_SIZE = 128
        FlashAttentionTriton.backward_impl = "compiled"   # or "triton"
    """

    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64
    backward_impl = "triton"

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        _require_triton()
        batch_shape = Q.shape[:-2]
        ctx.input_dtypes = (Q.dtype, K.dtype, V.dtype)
        dtype = _common_dtype(Q, K, V)
        Q, K, V = (t.to(dtype) for t in (Q, K, V))
        Qf, Kf, Vf = (_flatten_batch(t).contiguous() for t in (Q, K, V))
        q_tile = _pick_tile(Qf.shape[-2], FlashAttentionTriton.Q_TILE_SIZE)
        k_tile = _pick_tile(Kf.shape[-2], FlashAttentionTriton.K_TILE_SIZE)
        O, L = flash_forward_triton(Qf, Kf, Vf, is_causal, q_tile, k_tile)
        O = O.reshape(Q.shape)
        L = L.reshape(*batch_shape, Q.shape[-2])
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.tiles = (q_tile, k_tile)
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        n_q = Q.shape[-2]
        args = (
            _flatten_batch(Q).contiguous(),
            _flatten_batch(K).contiguous(),
            _flatten_batch(V).contiguous(),
            _flatten_batch(O).contiguous(),
            _flatten_batch(dO.to(Q.dtype)).contiguous(),
            L.reshape(-1, n_q).contiguous(),
            ctx.is_causal,
        )
        if FlashAttentionTriton.backward_impl == "triton":
            dQ, dK, dV = flash_backward_triton(*args, *ctx.tiles)
        else:
            dQ, dK, dV = get_flash_backward(Q.device)(*args)
        dq_t, dk_t, dv_t = ctx.input_dtypes
        return dQ.reshape(Q.shape).to(dq_t), dK.reshape(K.shape).to(dk_t), dV.reshape(V.shape).to(dv_t), None


def flash_attention_triton(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
    return FlashAttentionTriton.apply(Q, K, V, is_causal)
