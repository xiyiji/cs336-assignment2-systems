"""Model configurations used throughout the assignment (handout §2.1.2, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass

VOCAB_SIZE = 10_000
BATCH_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 512
CONTEXT_LENGTHS = (128, 256, 512, 1024)
DEFAULT_ROPE_THETA = 10_000.0


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

    def num_params(self, vocab_size: int = VOCAB_SIZE) -> int:
        """Exact parameter count of ``BasicsTransformerLM`` (untied lm_head)."""
        d, dff, L = self.d_model, self.d_ff, self.num_layers
        per_layer = 4 * d * d + 3 * d * dff + 2 * d  # attn projections, SwiGLU, two RMSNorms
        return vocab_size * d + L * per_layer + d + d * vocab_size


# Table 1 (Spring 2026 handout). "These are mostly based on GPT-2 configs."
MODEL_SIZES: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelSpec(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
    # Tiny configs for CPU smoke tests / CI.
    "tiny": ModelSpec(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "mini": ModelSpec(d_model=256, d_ff=1024, num_layers=4, num_heads=4),
}


@dataclass(frozen=True)
class LeaderboardConfig:
    """§9 leaderboard model (an ~8B Qwen-like config)."""

    ctx_len: int = 32768
    vocab_size: int = 151936
    d_model: int = 4096
    d_ff: int = 11008
    num_layers: int = 34
    num_heads: int = 32
    batch_size: int = 2
    is_causal: bool = True


def get_model_kwargs(
    size: str,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vocab_size: int = VOCAB_SIZE,
    rope_theta: float = DEFAULT_ROPE_THETA,
) -> dict:
    """Keyword arguments for ``cs336_basics.model.BasicsTransformerLM``."""
    if size not in MODEL_SIZES:
        raise KeyError(f"Unknown model size {size!r}; choose from {sorted(MODEL_SIZES)}")
    spec = MODEL_SIZES[size]
    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": spec.d_model,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "d_ff": spec.d_ff,
        "rope_theta": rope_theta,
    }
