"""Extra FlashAttention tests beyond the official suite.

* causal masking in the PyTorch implementation (forward + backward)
* arbitrary leading batch dims and non-square (n_q != n_k) inputs
* the ``torch.compile``d backward path
* the Triton kernels under the Triton *interpreter* (``TRITON_INTERPRET=1``),
  which lets CI validate kernel logic on CPU-only runners.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

from cs336_systems import flash_attention as fa
from cs336_systems.flash_attention import FlashAttentionPytorch, naive_attention

from .test_attention import _attention_and_lse, _make_attn_inputs


def _ref_grads(q, k, v, do, is_causal):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    naive_attention(q, k, v, is_causal).backward(do)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("is_causal", [False, True])
def test_pytorch_flash_causal_forward_backward(is_causal):
    q, k, v, do = _make_attn_inputs()
    o = FlashAttentionPytorch.apply(q, k, v, is_causal)
    o_ref, l_ref = _attention_and_lse(q, k, v, is_causal)
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    l = [t for t in o.grad_fn.saved_tensors if t.shape == (q.shape[0], q.shape[1])][0]
    torch.testing.assert_close(l, l_ref, rtol=1e-2, atol=1e-2)

    o.backward(do)
    dq, dk, dv = _ref_grads(q, k, v, do, is_causal)
    torch.testing.assert_close(q.grad, dq, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad, dk, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v.grad, dv, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [((2, 3), 64, 128, 32), ((1,), 256, 128, 16), ((5,), 32, 32, 64)])
def test_pytorch_flash_batch_dims_and_rectangular(shape):
    batch, n_q, n_k, d = shape
    torch.manual_seed(1)
    q = torch.randn(*batch, n_q, d, requires_grad=True)
    k = torch.randn(*batch, n_k, d, requires_grad=True)
    v = torch.randn(*batch, n_k, d, requires_grad=True)
    do = torch.randn(*batch, n_q, d)
    o = FlashAttentionPytorch.apply(q, k, v, False)
    torch.testing.assert_close(o, naive_attention(q, k, v), rtol=1e-3, atol=1e-3)
    o.backward(do)
    dq, dk, dv = _ref_grads(q, k, v, do, False)
    torch.testing.assert_close(q.grad, dq, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k.grad, dk, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v.grad, dv, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("is_causal", [False, True])
def test_pytorch_flash_bf16_inputs(is_causal):
    """bf16 Q/K/V (as under autocast): matmuls in bf16, softmax statistics in fp32."""
    q, k, v, do = _make_attn_inputs()
    qb, kb, vb = (t.detach().bfloat16().requires_grad_(True) for t in (q, k, v))
    o = FlashAttentionPytorch.apply(qb, kb, vb, is_causal)
    assert o.dtype == torch.bfloat16
    o.backward(do.bfloat16())
    o_ref = naive_attention(q, k, v, is_causal)
    dq, dk, dv = _ref_grads(q, k, v, do, is_causal)
    tol = dict(rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(o.float(), o_ref, **tol)
    torch.testing.assert_close(qb.grad.float(), dq, **tol)
    torch.testing.assert_close(kb.grad.float(), dk, **tol)
    torch.testing.assert_close(vb.grad.float(), dv, **tol)


def test_pytorch_flash_small_tiles_match():
    """Tile size must not change the result (online softmax invariance)."""
    q, k, v, _ = _make_attn_inputs()
    old = (FlashAttentionPytorch.Q_TILE_SIZE, FlashAttentionPytorch.K_TILE_SIZE)
    try:
        FlashAttentionPytorch.Q_TILE_SIZE, FlashAttentionPytorch.K_TILE_SIZE = 16, 32
        o_small = FlashAttentionPytorch.apply(q, k, v, True)
        FlashAttentionPytorch.Q_TILE_SIZE, FlashAttentionPytorch.K_TILE_SIZE = 128, 128
        o_big = FlashAttentionPytorch.apply(q, k, v, True)
    finally:
        FlashAttentionPytorch.Q_TILE_SIZE, FlashAttentionPytorch.K_TILE_SIZE = old
    torch.testing.assert_close(o_small, o_big, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("CS336_TEST_COMPILED_BACKWARD") is None,
    reason="set CS336_TEST_COMPILED_BACKWARD=1 to exercise the torch.compile backward (slow first call)",
)
def test_compiled_backward_matches_eager(monkeypatch):
    monkeypatch.setenv("CS336_COMPILE_FLASH_BACKWARD", "1")
    fa._compiled_backward = None
    q, k, v, do = _make_attn_inputs()
    FlashAttentionPytorch.apply(q, k, v, True).backward(do)
    dq, dk, dv = _ref_grads(q, k, v, do, True)
    torch.testing.assert_close(q.grad, dq, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad, dk, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v.grad, dv, rtol=1e-2, atol=1e-2)
    assert fa._compiled_backward is not None


# --------------------------------------------------------------------------- #
# Triton kernels through the interpreter (CPU) or on a real GPU
# --------------------------------------------------------------------------- #
def _triton_available() -> bool:
    if torch.cuda.is_available():
        return importlib.util.find_spec("triton") is not None
    return os.environ.get("TRITON_INTERPRET") == "1" and importlib.util.find_spec("triton") is not None


@pytest.mark.skipif(not _triton_available(), reason="needs CUDA, or triton with TRITON_INTERPRET=1")
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("backward_impl", ["triton", "compiled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_flash_forward_backward(is_causal, backward_impl, dtype):
    from cs336_systems.flash_triton import FlashAttentionTriton

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype is torch.bfloat16:
        pytest.skip("interpreter bf16 matmul precision is not meaningful on CPU")
    torch.manual_seed(0)
    b, n_q, n_k, d = 2, 128, 128, 32
    q = torch.randn(b, n_q, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, n_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, n_k, d, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(b, n_q, d, device=device, dtype=dtype)

    old_impl, old_tiles = FlashAttentionTriton.backward_impl, (FlashAttentionTriton.Q_TILE_SIZE, FlashAttentionTriton.K_TILE_SIZE)
    try:
        FlashAttentionTriton.backward_impl = backward_impl
        FlashAttentionTriton.Q_TILE_SIZE, FlashAttentionTriton.K_TILE_SIZE = 32, 64  # Bq != Bk on purpose
        o = FlashAttentionTriton.apply(q, k, v, is_causal)
        o_ref, l_ref = _attention_and_lse(q.float(), k.float(), v.float(), is_causal)
        tol = dict(rtol=1e-2, atol=1e-2) if dtype is torch.float32 else dict(rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(o.float(), o_ref, **tol)
        l = [t for t in o.grad_fn.saved_tensors if t.shape == (b, n_q)][0]
        torch.testing.assert_close(l.float(), l_ref, **tol)

        o.backward(do)
        dq, dk, dv = _ref_grads(q.float(), k.float(), v.float(), do.float(), is_causal)
        torch.testing.assert_close(q.grad.float(), dq, **tol)
        torch.testing.assert_close(k.grad.float(), dk, **tol)
        torch.testing.assert_close(v.grad.float(), dv, **tol)
    finally:
        FlashAttentionTriton.backward_impl = old_impl
        FlashAttentionTriton.Q_TILE_SIZE, FlashAttentionTriton.K_TILE_SIZE = old_tiles
