"""Tests for the model-level pieces: attention patching, chunked LM-head
cross-entropy, activation checkpointing equivalence."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.checkpointing import CheckpointedTransformerLM, measure_saved_bytes
from cs336_systems.config import get_model_kwargs
from cs336_systems.flash_patch import flash_attention
from cs336_systems.fused_ce import chunked_lm_head_cross_entropy


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return basics_model.BasicsTransformerLM(**get_model_kwargs("tiny", context_length=32, vocab_size=64))


def _batch(vocab=64, bs=2, ctx=32):
    torch.manual_seed(1)
    return torch.randint(0, vocab, (bs, ctx)), torch.randint(0, vocab, (bs, ctx))


def test_flash_patch_matches_original_attention():
    model = _tiny_model()
    x, y = _batch()
    ref = cross_entropy(model(x), y)
    ref.backward()
    ref_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    with flash_attention("pytorch"):
        assert basics_model.scaled_dot_product_attention is not None
        out = cross_entropy(model(x), y)
    out.backward()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    for g, g_ref in zip((p.grad for p in model.parameters()), ref_grads):
        torch.testing.assert_close(g, g_ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [7, 64, 1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_lm_head_cross_entropy(chunk_size, dtype):
    torch.manual_seed(0)
    vocab, d = 50, 16
    h = torch.randn(3, 20, d, dtype=dtype, requires_grad=True)
    w = torch.randn(vocab, d, dtype=dtype, requires_grad=True)
    t = torch.randint(0, vocab, (3, 20))
    loss = chunked_lm_head_cross_entropy(h, w, t, chunk_size)
    loss.backward()

    h2 = h.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    ref = cross_entropy((h2 @ w2.t()).float(), t)
    ref.backward()
    tol = dict(rtol=1e-4, atol=1e-5) if dtype is torch.float32 else dict(rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(loss, ref, **tol)
    torch.testing.assert_close(h.grad.float(), h2.grad.float(), **tol)
    torch.testing.assert_close(w.grad.float(), w2.grad.float(), **tol)


@pytest.mark.parametrize("strategy,segment", [("segments", 1), ("segments", 2), ("recursive", None)])
def test_checkpointing_matches_and_saves_memory(strategy, segment):
    model = _tiny_model()
    ckpt = CheckpointedTransformerLM(deepcopy(model), strategy=strategy, segment_size=segment)
    x, y = _batch()

    ref_bytes, _ = measure_saved_bytes(lambda: cross_entropy(model(x), y).backward())
    ref_grads = [p.grad.clone() for p in model.parameters()]
    ckpt_bytes, _ = measure_saved_bytes(lambda: cross_entropy(ckpt(x), y).backward())
    for g, g_ref in zip((p.grad for p in ckpt.model.parameters()), ref_grads):
        torch.testing.assert_close(g, g_ref, rtol=1e-5, atol=1e-6)
    # saved_tensors_hooks also sees the recomputation during backward, but the
    # forward-only residual set is what dominates: it must be strictly smaller.
    fwd_only_ref, _ = measure_saved_bytes(lambda: model(x))
    fwd_only_ckpt, _ = measure_saved_bytes(lambda: ckpt(x))
    assert fwd_only_ckpt < fwd_only_ref, (fwd_only_ckpt, fwd_only_ref)
