"""Register native and Torch kernels once without initializing a GPU."""

from .activation import packed_silu_and_mul, silu_and_mul

__all__ = ["packed_silu_and_mul", "silu_and_mul"]
