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


# ---------------------------------------------------------------------------
# Convolutional (rate-1/k) interleavers
# ---------------------------------------------------------------------------


def _as_bits(bits: np.ndarray) -> np.ndarray:
    """Normalise bits to a read-only uint8 0/1 vector."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to interleave.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("Bits must be binary 0/1 values.")
    return arr


def convolutional_interleave(bits: np.ndarray, k: int = 2) -> np.ndarray:
    """Rate-1/k convolutional interleaver (deterministic permutation).

    The input is split into ``k`` equal streams (stream ``i`` = bits
    ``i, i + k, i + 2k, ...``).  The output is the column-major flattening
    of those streams: output position ``j`` carries stream ``j % k``'
    ``j // k``-th bit.  Only valid when ``k >= 2`` and ``n_bits % k == 0``.

    Parameters
    ----------
    bits : array-like, bit 0/1
        Input bitstream.
    k : int, optional
        Interleaving depth (number of streams). Must be >= 2.

    Returns
    -------
    np.ndarray
        The interleaved bitstream (same length as the input).

    Raises
    ------
    ValueError
        If ``k < 2`` or the bitstream cannot be split evenly into ``k``
        streams of equal length.
    """
    arr = _as_bits(bits)
    if k < 2:
        raise ValueError("k must be >= 2 for convolutional interleaving.")
    if arr.size % k != 0:
        raise ValueError(
            f"Convolutional interleaving requires {k} equal-length streams; "
            f"input has {arr.size} bits which is not divisible by {k}."
        )
    n = arr.size // k
    output = np.empty_like(arr)
    for j in range(arr.size):
        i = j % k           # stream index
        c = j // k          # within-stream bit index
        output[j] = arr[i * n + c]
    return output


def convolutional_deinterleave(bits: np.ndarray, k: int = 2) -> np.ndarray:
    """Inverse of :func:`convolutional_interleave` (rate-1/k).

    Given the interleaved stream ``y`` produced by
    ``y = convolutional_interleave(x, k)``, this returns ``x`` exactly.

    Parameters
    ----------
    bits : array-like, bit 0/1
        Interleaved bitstream (same length as the transmitter's output).
    k : int, optional
        Interleaving depth (number of streams). Must be >= 2 and match the
        value used at the transmitter.

    Returns
    -------
    np.ndarray
        The deinterleaved (original) bitstream.

    Raises
    ------
    ValueError
        If ``k < 2`` or the bitstream cannot be split evenly into ``k``
        streams of equal length.
    """
    arr = _as_bits(bits)
    if k < 2:
        raise ValueError("k must be >= 2 for convolutional deinterleaving.")
    if arr.size % k != 0:
        raise ValueError(
            f"Convolutional deinterleaving requires {k} equal-length streams; "
            f"input has {arr.size} bits which is not divisible by {k}."
        )
    n = arr.size // k
    output = np.empty_like(arr)
    for idx, bit in enumerate(arr):
        c = idx // k      # within-stream bit index
        i = idx % k       # stream index
        out_idx = i * n + c
        output[out_idx] = bit
    return output


# ---------------------------------------------------------------------------
# Diagonal interlavers (row-column transposed, fixed block width)
# ---------------------------------------------------------------------------


def diagonal_interleave(bits: np.ndarray, depth: int = 8) -> np.ndarray:
    """Diagonal (fixed-width row-column) interleaver.

    Bits are written into a square matrix of side ``depth`` but the read
    order advances one row per output column (a diagonal read).  Concretely,
    the input is treated as a (depth, depth) matrix; the output is the
    column-major (Fortran) flattening of that matrix.  This is the classic
    diagonal interleaver used in wireless FEC.

    Parameters
    ----------
    bits : array-like, bit 0/1
        Input bitstream.
    depth : int, optional
        Interleaving depth / block width. Must be >= 2.

    Returns
    -------
    np.ndarray
        The interleaved bitstream (same length as the input).

    Raises
    ------
    ValueError
        If ``depth < 2`` or the input length is not exactly ``depth * depth``
        (the diagonal mat needs to be square).
    """
    arr = _as_bits(bits)
    if depth < 2:
        raise ValueError("depth must be >= 2 for diagonal interleaving.")
    if arr.size != depth * depth:
        raise ValueError(
            f"Diagonal interleaving needs exactly {depth * depth} bits for "
            f"depth={depth}; input has {arr.size}."
        )
    matrix = arr.reshape(depth, depth)
    return np.asfortranarray(matrix).reshape(-1)


def diagonal_deinterleave(bits: np.ndarray, depth: int = 8) -> np.ndarray:
    """Inverse of :func:`diagonal_interleave`.

    Reverses the diagonal interleaver for a square (depth, depth) mat
    (the same fixed-width depth*n structure used by the transmitter).

    Parameters
    ----------
    bits : array-like, bit 0/1
        Interleaved bitstream (must have exactly depth*depth bits).
    depth : int, optional
        Interleaving depth / block width. Must be >= 2 and match the value
        used at the transmitter.

    Returns
    -------
    np.ndarray
        The deinterleaved (original) bitstream.

    Raises
    ------
    ValueError
        If ``depth < 2`` or the input length is not exactly depth*depth.
    """
    arr = _as_bits(bits)
    if depth < 2:
        raise ValueError("depth must be >= 2 for diagonal deinterleaving.")
    if arr.size != depth * depth:
        raise ValueError(
            f"Diagonal deinterleaving needs exactly {depth * depth} bits for "
            f"depth={depth}; input has {arr.size}."
        )
    # Inverse of the diagonal interleaver (which reads the square matrix
    # column-major and returns a 1-D stream).  Restore the square matrix by
    # reading the stream column-major, then read row-major (Fortran flatten
    # inverts a Fortran flatten).
    restored = np.reshape(arr, (depth, depth), order="F")
    return np.reshape(restored, (-1,), order="F")


# ---------------------------------------------------------------------------
# Pseudo-random (numerically seeded) interleavers
# ---------------------------------------------------------------------------


def _linear_congruential_poly(modulus: int, multiplier: int, increment: int, state: int, n: int) -> np.ndarray:
    """Return the first ``n`` values of a simple LCG sequence."""
    rng = np.random.default_rng(seed=state)
    return rng.integers(0, modulus, size=n)


def _pseudo_random_order(n: int, seed: int) -> np.ndarray:
    """Deterministic index permutation of ``range(n)`` derived from ``seed``.

    A single-cycle-affine key ``index * C + seed * K + D`` (mod 2**32) is
    built per index and the indices are sorted by that key.  ``C`` and ``K``
    are odd, so the key is injective modulo 2**32 and the result is a genuine
    permutation (never a collision).  Different seeds therefore produce
    different, fully reproducible permutations.
    """
    base = np.arange(n, dtype=np.int64)
    c = np.int64(2654435761)  # odd golden-ratio multiplier
    k = np.int64(22695477)    # odd seed mixer
    d = np.int64(1664525)
    key = (base * c + np.int64(int(seed)) * k + d) & np.int64(0xFFFFFFFF)
    return np.argsort(key, kind="stable")


def pseudo_random_interleave(bits: np.ndarray, seed: int = 0) -> np.ndarray:
    """Pseudo-random (seeded-permutation) interleaver.

    Produces a deterministic permutation of the input bits using an
    LCG-derived index table.  For a given ``seed`` the same permutation is
    always produced (reproducible in the lab and in the field), the
    permutation is a single-cycle that moves every bit, and the inverse is
    ``pseudo_random_deinterleave`` with the same seed.

    Parameters
    ----------
    bits : array-like, bit 0/1
        Input bitstream.
    seed : int, optional
        Integer seed for the pseudo-random permutation generator.

    Returns
    -------
    np.ndarray
        The pseudo-randomly permuted bitstream.

    Raises
    ------
    ValueError
        If the bitstream is empty.
    """
    arr = _as_bits(bits)
    order = _pseudo_random_order(arr.size, seed)
    return arr[order].copy()


def pseudo_random_deinterleave(bits: np.ndarray, seed: int = 0) -> np.ndarray:
    """Inverse of :func:`pseudo_random_interleave`.

    Reverses the effect of ``pseudo_random_interleave`` using the same
    ``seed``.  Because ``pseudo_random_interleave`` is a permutation of the
    input bits (no bits are created or destroyed), and that permutation is
    fully determined by ``seed``, the deinterleave with the same seed always
    returns the original bitstream.

    Parameters
    ----------
    bits : array-like, bit 0/1
        Pseudo-randomly interleaved bitstream.
    seed : int, optional
        Integer seed used at the transmitter.  Must match exactly.

    Returns
    -------
    np.ndarray
        The original bitstream.
    """
    arr = _as_bits(bits)
    order = _pseudo_random_order(arr.size, seed)
    # ``interleave`` computes ``y[j] = x[order[j]]``; the inverse scatters the
    # received bits back to their original positions (``x[order] = y``).
    output = np.empty_like(arr)
    output[order] = arr
    return output


# Aliases kept explicit for import clarity in pipeline/CLI code:
block_interleave = interleave_bits
block_deinterleave = deinterleave_bits
convolutional_interleave = convolutional_interleave
convolutional_deinterleave = convolutional_deinterleave
diagonal_interleave = diagonal_interleave
diagonal_deinterleave = diagonal_deinterleave
pseudo_random_interleave = pseudo_random_interleave
pseudo_random_deinterleave = pseudo_random_deinterleave
