"""Compact regular LDPC codec with hard-decision bit-flipping decoding.

This is a small, fully deterministic, dependency-free LDPC suitable for
tests and lab demonstrations - **not** a standards-compliant large block
code and not tuned for throughput.

Code
----
A ``(3, 6)``-regular parity-check matrix ``H = [A | B]`` is built from two
invertible circulants ``A`` and ``B`` (8x8).  Each row has weight 6 and
each column weight 3, giving an ``N = 16`` / ``K = 8`` rate-1/2 code.

* Encode: ``H = [A | B]`` => parity ``p = B**-1 A d`` and codeword
  ``[d | p]`` (systematic).
* Decode: hard-decision Gallager **bit-flipping** - count the unsatisfied
  checks each bit participates in and flip the worst bit, repeating until
  the syndrome clears or the iteration budget is exhausted.

Data is zero-padded to a whole number of ``K``-bit blocks at encode time.
"""

from __future__ import annotations

import numpy as np

K = 8   # data bits per block
N = 16  # codeword bits per block
MAX_ITERATIONS = 24


def _circulant(base: list[int]) -> np.ndarray:
    """Row-``i`` circular shift of ``base`` by ``i`` (mod 2)."""
    size = len(base)
    matrix = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            matrix[i, j] = base[(j - i) % size]
    return matrix


# Odd-weight base rows -> the polynomial is coprime with (x+1)**8, so the
# circulants are invertible over GF(2).
_A = _circulant([1, 1, 0, 1, 0, 0, 0, 0])
_B = _circulant([1, 0, 1, 1, 0, 0, 0, 0])
H = np.concatenate([_A, _B], axis=1).astype(np.uint8)  # (8, 16)


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a square 0/1 matrix over GF(2) via Gauss-Jordan."""
    n = matrix.shape[0]
    aug = np.concatenate(
        [matrix.astype(np.uint8) % 2, np.eye(n, dtype=np.uint8)], axis=1
    )
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r, col] == 1), None)
        if pivot is None:
            raise ValueError("matrix is singular over GF(2)")
        aug[[col, pivot]] = aug[[pivot, col]]
        for r in range(n):
            if r != col and aug[r, col] == 1:
                aug[r] ^= aug[col]
    return aug[:, n:] % 2


_B_INV = _gf2_inverse(_B)
# Parity generator: p = (B^-1 A) d
_PARITY_MATRIX = (_B_INV @ _A) % 2


def _as_matrix(bits: np.ndarray, block: int) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("bits must be binary.")
    return arr.reshape(-1, block)


def encode(bits: np.ndarray) -> np.ndarray:
    """Systematic rate-1/2 LDPC encode; data padded to a multiple of K bits."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to encode.")
    if arr.size % K != 0:
        arr = np.concatenate(
            [arr, np.zeros((-arr.size) % K, dtype=np.uint8)]
        )
    data = _as_matrix(arr, K)
    parity = (data @ _PARITY_MATRIX.T) % 2
    return np.concatenate([data, parity], axis=1).reshape(-1).astype(np.uint8)


def decode(bits: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Bit-flipping decode; returns (data_bits, stats)."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to decode.")
    if arr.size % N != 0:
        raise ValueError(f"LDPC decode requires blocks of {N} bits, got {arr.size}.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("bits must be binary.")

    blocks = arr.reshape(-1, N).astype(np.uint8)
    corrected_errors = 0
    uncorrectable = 0
    decoded_blocks = np.empty((blocks.shape[0], K), dtype=np.uint8)

    # Column participation counts (all 3 for a (3,6)-regular H).
    for row, block in enumerate(blocks):
        working = block.copy()
        fixed = False
        for _ in range(MAX_ITERATIONS):
            syndrome = (H @ working) % 2
            if not syndrome.any():
                fixed = True
                break
            # Per-bit count of unsatisfied checks; flip the worst bit.
            counts = syndrome @ H
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break
            working[best] ^= 1
        if fixed:
            flips = int(np.count_nonzero(working != block))
            corrected_errors += flips
            decoded_blocks[row] = working[:K]
        else:
            uncorrectable += 1
            decoded_blocks[row] = block[:K]

    stats = {
        "blocks": int(blocks.shape[0]),
        "corrected_errors": int(corrected_errors),
        "uncorrectable_blocks": int(uncorrectable),
    }
    return decoded_blocks.reshape(-1).astype(np.uint8), stats
