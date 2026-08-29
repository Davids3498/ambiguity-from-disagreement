"""Counter-based PRNG: rand(seed, stream) is a pure function.

Why not torch.Generator: a generator makes scene content depend on the order in
which envs were drawn, so seed -> scene stops being a pure function the moment a
batch is resliced. splitmix64 keyed on (seed, stream) is order-independent AND
fully vectorized, which is what DAgger reshuffling in week 4 needs.

The global torch RNG is never touched anywhere in sim/.
"""
import torch
from torch import Tensor

_G = 0x9E3779B97F4A7C15 - (1 << 64)   # as signed int64; torch mul wraps
_M1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_M2 = 0x94D049BB133111EB - (1 << 64)


def _wrap64(x: int) -> int:
    """Fold a Python int into signed int64, matching torch's wrapping arithmetic."""
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >= (1 << 63) else x


def _shr(x: Tensor, k: int) -> Tensor:
    """Logical right shift on int64 (torch's >> is arithmetic)."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _mix(seeds: Tensor, stream: int) -> Tensor:
    z = seeds + _wrap64(_G * (stream + 1))
    z = (z ^ _shr(z, 30)) * _M1
    z = (z ^ _shr(z, 27)) * _M2
    return z ^ _shr(z, 31)


def rand(seeds: Tensor, stream: int) -> Tensor:
    """Uniform [0,1) float32, shape of seeds. Pure in (seed, stream)."""
    z = _mix(seeds.to(torch.int64), stream)
    return ((z & 0xFFFFFF).to(torch.float32) / float(1 << 24))


def uniform(seeds: Tensor, stream: int, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * rand(seeds, stream)
