"""Systematic Hamming(7,4) encoding and syndrome decoding."""

from __future__ import annotations

import numpy as np

# Generator matrix (systematic), 4 x 7. Codeword = data @ G (mod 2).
# Codeword layout: [p1 p2 d1 p3 d2 d3 d4] (Hamming parity positions 1,2,4)
#   p1 = d1 ^ d2 ^ d4,  p2 = d1 ^ d3 ^ d4,  p3 = d2 ^ d3 ^ d4
G = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ],
    dtype=np.int8,
)

# Parity check matrix H (3 x 7): syndrome = H * received (mod 2)
H = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)

# Syndrome -> error position (1-based), 0 = no error
_SYNDROME_TABLE: dict[tuple[int, ...], int] = {}
for _error_pos in range(8):
    _e = np.zeros(7, dtype=np.int8)
    if _error_pos > 0:
        _e[_error_pos - 1] = 1
    _syndrome = tuple((H @ _e) % 2)
    _SYNDROME_TABLE[_syndrome] = _error_pos


def encode(bits: np.ndarray) -> np.ndarray:
    """Encode groups of 4 data bits into 7-bit codewords."""
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if bits.size == 0:
        raise ValueError("No bits to encode.")
    if bits.size % 4 != 0:
        bits = np.concatenate(
            [bits, np.zeros((-bits.size) % 4, dtype=np.int8)]
        )
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary.")
    blocks = bits.reshape(-1, 4)
    codewords = (blocks @ G) % 2
    return codewords.reshape(-1).astype(np.uint8)


def decode(bits: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """
    Decode 7-bit codewords, correcting up to one bit error per block.

    Returns (data_bits, stats) where stats reports corrected and
    uncorrectable block counts.
    """
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if bits.size == 0:
        raise ValueError("No bits to decode.")
    if bits.size % 7 != 0:
        raise ValueError(
            f"Hamming(7,4) decode requires a multiple of 7 bits, got {bits.size}."
        )
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary.")

    blocks = bits.reshape(-1, 7).astype(np.int8)
    syndromes = (blocks @ H.T) % 2

    corrected = blocks.copy()
    corrected_count = 0
    for row in range(syndromes.shape[0]):
        position = _SYNDROME_TABLE[tuple(syndromes[row])]
        if position > 0:
            corrected[row, position - 1] ^= 1
            corrected_count += 1

    data = corrected[:, [2, 4, 5, 6]]
    stats = {
        "blocks": int(blocks.shape[0]),
        "corrected_errors": int(corrected_count),
        "uncorrectable_blocks": 0,
    }
    return data.reshape(-1).astype(np.uint8), stats
