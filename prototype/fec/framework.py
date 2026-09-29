"""
FEC framework: explicit scheme registry shared by the CLI, pipeline,
and tests. FEC is never inferred from data; callers choose the scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.exceptions import FECError
from . import concatenated, convolutional, hamming, ldpc, reed_solomon, repetition


@dataclass
class FECResult:
    """Result of an FEC encode or decode operation."""

    scheme: str
    operation: str  # "encode" | "decode"
    input_bits: int
    output_bits: int
    code_rate: float | None = None
    corrected_errors: int = 0
    uncorrectable_blocks: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


_SCHEMES: dict[str, dict[str, Any]] = {
    "hamming74": {
        "encode": lambda bits: hamming.encode(bits),
        "decode": lambda bits: hamming.decode(bits),
        "block_align": lambda n: (n // 7) * 7,
        "rate": 4 / 7,
        "description": "Systematic Hamming(7,4), corrects 1 bit per 7-bit block",
    },
    "repetition3": {
        "encode": lambda bits: repetition.encode(bits, repetitions=3),
        "decode": lambda bits: repetition.decode(bits, repetitions=3),
        "block_align": lambda n: (n // 3) * 3,
        "rate": 1 / 3,
        "description": "Rate-1/3 repetition with majority-vote decoding",
    },
    "conv12": {
        "encode": lambda bits: convolutional.encode(bits, tail=True),
        "decode": lambda bits: convolutional.decode(bits),
        "block_align": lambda n: (n // 2) * 2,
        "rate": 1 / 2,
        "description": "K=7 rate-1/2 convolutional (171,133) with Viterbi decoding",
    },
    "reedsolomon": {
        "encode": lambda bits: reed_solomon.encode(bits),
        "decode": lambda bits: reed_solomon.decode(bits),
        "block_align": lambda n: (
            n // (8 * (reed_solomon.K_DEFAULT + reed_solomon.NSYM_DEFAULT))
        )
        * (8 * (reed_solomon.K_DEFAULT + reed_solomon.NSYM_DEFAULT)),
        "rate": reed_solomon.K_DEFAULT
        / (reed_solomon.K_DEFAULT + reed_solomon.NSYM_DEFAULT),
        "description": "Shortened Reed-Solomon over GF(256), 32 data + 8 "
        "parity bytes (t=4 symbol errors per block)",
    },
    "ldpc": {
        "encode": lambda bits: ldpc.encode(bits),
        "decode": lambda bits: ldpc.decode(bits),
        "block_align": lambda n: (n // ldpc.N) * ldpc.N,
        "rate": ldpc.K / ldpc.N,
        "description": "Compact (3,6)-regular LDPC (16,8) with hard-decision "
        "bit-flipping decoding",
    },
    "concatenated": {
        "encode": lambda bits: concatenated.encode(bits),
        "decode": lambda bits: concatenated.decode(bits),
        # The inner Viterbi strips its 6 tail bits, and the outer RS needs
        # whole 40-byte blocks: 2*(320k + 6) code bits per k outer blocks.
        "block_align": lambda n: 2 * 320 * ((n - 12) // 640) + 12
        if n >= 652
        else 0,
        "rate": (reed_solomon.K_DEFAULT
                 / (reed_solomon.K_DEFAULT + reed_solomon.NSYM_DEFAULT)) / 2,
        "description": "Serial concatenation: Reed-Solomon (outer) + "
        "convolutional K=7 rate-1/2 (inner)",
    },
}


def list_schemes() -> tuple[str, ...]:
    """Names of all registered FEC schemes."""
    return tuple(_SCHEMES)


def aligned_bits_size(bits_size: int, scheme: str) -> int:
    """Largest whole-codeword length that fits in ``bits_size`` bits.

    A real capture is never exactly a whole number of codewords: the
    pulse-shaping tail (and any dropped leading symbol) truncates the
    stream.  Decoding needs whole codewords, so this reports how many of
    the received bits actually form complete codewords and how many are
    left over.  The leftovers are dropped and *reported*, never invented.
    """
    if scheme not in _SCHEMES:
        raise FECError(f"Unknown FEC scheme '{scheme}'. Available: {sorted(_SCHEMES)}")
    return int(_SCHEMES[scheme]["block_align"](int(bits_size)))


def aligned_size(bits_size: int, scheme: str) -> tuple[int, int]:
    """Return ``(usable_bits, trimmed_tail_bits)`` for a scheme."""
    usable = aligned_bits_size(bits_size, scheme)
    return usable, max(0, int(bits_size) - usable)


def describe_scheme(name: str) -> str:
    """Human-readable description of a scheme."""
    if name not in _SCHEMES:
        raise FECError(f"Unknown FEC scheme '{name}'.")
    return str(_SCHEMES[name]["description"])


def encode_bits(bits: np.ndarray, scheme: str) -> tuple[np.ndarray, FECResult]:
    """Encode bits with the named scheme; returns (encoded, FECResult)."""
    if scheme not in _SCHEMES:
        raise FECError(f"Unknown FEC scheme '{scheme}'. Available: {sorted(_SCHEMES)}")
    encoded = _SCHEMES[scheme]["encode"](bits)
    result = FECResult(
        scheme=scheme,
        operation="encode",
        input_bits=int(np.asarray(bits).size),
        output_bits=int(encoded.size),
        code_rate=float(_SCHEMES[scheme]["rate"]),
    )
    return encoded, result


def decode_bits(
    bits: np.ndarray,
    scheme: str,
    trim_partial_codeword: bool = False,
) -> tuple[np.ndarray, FECResult]:
    """Decode bits with the named scheme; returns (decoded, FECResult).

    With ``trim_partial_codeword=True`` the stream is first aligned to the
    largest whole number of codewords for the scheme, so a capture whose
    trailing symbols were lost by the receiver still decodes.  The number
    of dropped tail bits is reported in ``FECResult.extra``.
    """
    if scheme not in _SCHEMES:
        raise FECError(f"Unknown FEC scheme '{scheme}'. Available: {sorted(_SCHEMES)}")
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    trimmed = 0
    if trim_partial_codeword:
        usable, trimmed = aligned_size(int(arr.size), scheme)
        if usable <= 0:
            raise FECError(
                f"{scheme} decode needs at least one whole codeword "
                f"({arr.size} bits received)"
            )
        arr = arr[:usable]
    try:
        decoded, stats = _SCHEMES[scheme]["decode"](arr)
    except ValueError as exc:
        raise FECError(f"{scheme} decode failed: {exc}") from exc
    result = FECResult(
        scheme=scheme,
        operation="decode",
        input_bits=int(arr.size),
        output_bits=int(decoded.size),
        code_rate=float(_SCHEMES[scheme]["rate"]),
        corrected_errors=int(stats.get("corrected_errors", 0)),
        uncorrectable_blocks=int(stats.get("uncorrectable_blocks", 0)),
        extra={k: v for k, v in stats.items()
               if k not in ("corrected_errors", "uncorrectable_blocks")},
    )
    if trim_partial_codeword:
        result.extra["trimmed_tail_bits"] = int(trimmed)
    return decoded, result


def apply_encode(bits: np.ndarray, scheme: str | None) -> tuple[np.ndarray, FECResult | None]:
    """Encode with FEC when a scheme is configured; passthrough otherwise."""
    if scheme is None:
        return np.asarray(bits, dtype=np.uint8), None
    return encode_bits(bits, scheme)


def apply_decode(bits: np.ndarray, scheme: str | None) -> tuple[np.ndarray, FECResult | None]:
    """Decode with FEC when a scheme is configured; passthrough otherwise."""
    if scheme is None:
        return np.asarray(bits, dtype=np.uint8), None
    return decode_bits(bits, scheme)
