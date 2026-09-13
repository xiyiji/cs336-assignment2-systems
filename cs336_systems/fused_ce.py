"""Chunked / fused LM-head + cross-entropy (handout §9 leaderboard idea).

The basics model materialises logits of shape ``(batch, seq, vocab)``; for the
leaderboard config (2 x 32768 x 151936, bf16) that is 19 GiB *before* the fp32
copies made by the loss. :class:`ChunkedLMHeadCrossEntropy` computes the mean
token loss chunk-by-chunk along the sequence, never keeping more than one
chunk of logits alive, and recomputes each chunk's logits in backward
(``dlogits = softmax - onehot``) to produce ``dH`` and ``dW`` without ever
storing the full logits tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ChunkedLMHeadCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: Tensor, weight: Tensor, targets: Tensor, chunk_size: int) -> Tensor:
        """hidden: (..., d), weight: (vocab, d), targets: (...). Returns the mean loss."""
        h = hidden.reshape(-1, hidden.shape[-1])
        t = targets.reshape(-1)
        n = h.shape[0]
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        for start in range(0, n, chunk_size):
            hc = h[start : start + chunk_size].to(weight.dtype)  # matmul in the weight's dtype (bf16 under FSDP/autocast)
            logits = (hc @ weight.t()).float()
            lse = torch.logsumexp(logits, dim=-1)
            picked = logits.gather(-1, t[start : start + chunk_size, None]).squeeze(-1)
            loss += (lse - picked).sum()
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size
        return loss / n

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        hidden, weight, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        h = hidden.reshape(-1, hidden.shape[-1])
        t = targets.reshape(-1)
        n = h.shape[0]
        dh = torch.empty_like(h)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        scale = grad_out.float() / n
        for start in range(0, n, chunk_size):
            hc = h[start : start + chunk_size].to(weight.dtype)
            tc = t[start : start + chunk_size]
            logits = (hc @ weight.t()).float()
            probs = torch.softmax(logits, dim=-1)
            probs[torch.arange(tc.shape[0], device=h.device), tc] -= 1.0
            probs *= scale  # dlogits (fp32)
            dlogits = probs.to(weight.dtype)
            dh[start : start + chunk_size] = (dlogits @ weight).to(dh.dtype)
            dw += (dlogits.t() @ hc).float()
        return dh.reshape(hidden.shape), dw.to(weight.dtype), None, None


def chunked_lm_head_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor, chunk_size: int = 4096) -> Tensor:
    return ChunkedLMHeadCrossEntropy.apply(hidden, weight, targets, chunk_size)


class FusedLMHeadLoss(torch.nn.Module):
    """``nn.Module`` form of :func:`chunked_lm_head_cross_entropy` sharing the LM-head weight.

    Replacing ``model.lm_head`` with this module lets FSDP treat the head like any
    other sharded layer (``fsdp_shard_weight`` marks it for sharding): the weight
    is all-gathered by FSDP's forward/backward hooks and the full gradient
    returned by the autograd Function is reduce-scattered like every other one.
    """

    fsdp_shard_weight = True

    def __init__(self, weight: torch.nn.Parameter, chunk_size: int = 4096):
        super().__init__()
        self.weight = weight
        self.chunk_size = chunk_size

    def forward(self, hidden: Tensor, targets: Tensor) -> Tensor:
        return ChunkedLMHeadCrossEntropy.apply(hidden, self.weight, targets, self.chunk_size)
