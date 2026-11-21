"""Register native and Torch kernels once without initializing a GPU."""

from .activation import ouro_mlp, silu_and_mul

__all__ = ["ouro_mlp", "silu_and_mul"]
