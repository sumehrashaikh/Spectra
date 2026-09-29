"""Concatenated FEC: Reed-Solomon outer code + convolutional inner code.

This is the classic serial-concatenation construction (a symbol-oriented
outer code cleaning up the residual burst errors left by a bit-oriented
inner code):

* **Encode**: ``data -> RS(k=32, parity=8) -> convolutional K=7 rate-1/2``.
* **Decode**: ``stream -> Viterbi -> RS decode``.

Both stages reuse the working :mod:`prototype.fec.reed_solomon` and
:mod:`prototype.fec.convolutional` implementations; no new DSP or
algorithm is introduced.  Decoding is best-effort: inner-stage failures
are counted and the outer stage still runs, so the pipeline degrades
gracefully on noisy input.
"""

from __future__ import annotations

import numpy as np

from . import convolutional, reed_solomon


def encode(bits: np.ndarray) -> np.ndarray:
    """Encode ``bits`` with the RS outer + convolutional inner pair."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to encode.")
    outer = reed_solomon.encode(arr)
    return convolutional.encode(outer, tail=True)


def decode(bits: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Decode the concatenated stream; returns (bits, merged stats)."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to decode.")

    recovered, inner_stats = convolutional.decode(arr)
    decoded, outer_stats = reed_solomon.decode(recovered)

    stats = {
        "corrected_errors": int(
            inner_stats.get("corrected_errors", 0)
            + outer_stats.get("corrected_errors", 0)
        ),
        "uncorrectable_blocks": int(
            inner_stats.get("uncorrectable_blocks", 0)
            + outer_stats.get("uncorrectable_blocks", 0)
        ),
        "inner": dict(inner_stats),
        "outer": dict(outer_stats),
    }
    return decoded, stats
