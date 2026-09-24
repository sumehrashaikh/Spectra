"""Repetition code: rate 1/n encode with majority-vote decode."""

from __future__ import annotations

import numpy as np


def encode(bits: np.ndarray, repetitions: int = 3) -> np.ndarray:
    """Repeat every bit ``repetitions`` times (rate 1/n)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        raise ValueError("No bits to encode.")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary.")
    return np.repeat(bits, repetitions)


def decode(bits: np.ndarray, repetitions: int = 3) -> tuple[np.ndarray, dict[str, int]]:
    """Majority-vote decode of ``repetitions``-fold repeated bits."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        raise ValueError("No bits to decode.")
    if bits.size % repetitions != 0:
        raise ValueError(
            f"Bit count ({bits.size}) is not a multiple of "
            f"repetitions ({repetitions})."
        )
    blocks = bits.reshape(-1, repetitions)
    decided = (np.sum(blocks, axis=1) * 2 > repetitions).astype(np.uint8)
    stats = {
        "blocks": int(blocks.shape[0]),
        "corrected_errors": int(np.count_nonzero(np.any(blocks != decided[:, None], axis=1))),
        "uncorrectable_blocks": 0,
    }
    return decided, stats
