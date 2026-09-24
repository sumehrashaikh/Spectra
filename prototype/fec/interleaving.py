"""Block (row-column) interleaving and deinterleaving.

Bits are written row-wise into a (depth x cols) matrix and read
column-wise (and vice versa for deinterleaving). Tail padding bits are
removed on deinterleave when ``original_size`` is given.
"""

from __future__ import annotations

import numpy as np


def interleave_bits(bits: np.ndarray, depth: int = 8) -> np.ndarray:
    """Row-column block interleaver (depth >= 2; depth < 2 is identity)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if depth < 2:
        return bits.copy()
    if bits.size == 0:
        raise ValueError("No bits to interleave.")
    cols = -(-bits.size // depth)  # ceil division
    padded = np.zeros(depth * cols, dtype=np.uint8)
    padded[: bits.size] = bits
    matrix = padded.reshape(depth, cols)
    return matrix.T.reshape(-1)


def deinterleave_bits(
    bits: np.ndarray,
    depth: int = 8,
    original_size: int | None = None,
) -> np.ndarray:
    """Inverse of :func:`interleave_bits`."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if depth < 2:
        return bits.copy()
    if bits.size == 0:
        raise ValueError("No bits to deinterleave.")
    cols = -(-bits.size // depth)
    padded_size = depth * cols
    padded = np.zeros(padded_size, dtype=np.uint8)
    usable = min(bits.size, padded_size)
    padded[:usable] = bits[:usable]
    matrix = padded.reshape(cols, depth)
    recovered = matrix.T.reshape(-1)
    if original_size is not None:
        recovered = recovered[:original_size]
    return recovered


# Aliases kept explicit for import clarity in pipeline/CLI code:
block_interleave = interleave_bits
block_deinterleave = deinterleave_bits
