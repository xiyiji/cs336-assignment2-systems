import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from cs336_systems.config import MODEL_SIZES, LeaderboardConfig, ModelSpec, get_model_kwargs
from cs336_systems.ddp import DDP, DDP_VARIANTS, DDPBucketed, DDPOverlapIndividual, FlatDDP, NaiveDDP
from cs336_systems.flash_attention import FlashAttentionPytorch, flash_attention_pytorch, naive_attention
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

__all__ = [
    "MODEL_SIZES",
    "LeaderboardConfig",
    "ModelSpec",
    "get_model_kwargs",
    "DDP",
    "DDP_VARIANTS",
    "DDPBucketed",
    "DDPOverlapIndividual",
    "FlatDDP",
    "NaiveDDP",
    "FlashAttentionPytorch",
    "flash_attention_pytorch",
    "naive_attention",
    "FSDP",
    "ShardedOptimizer",
]
