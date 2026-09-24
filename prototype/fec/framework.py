"""
FEC framework: explicit scheme registry shared by the CLI, pipeline,
and tests. FEC is never inferred from data; callers choose the scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.exceptions import FECError
from . import convolutional, hamming, repetition


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
        "rate": 4 / 7,
        "description": "Systematic Hamming(7,4), corrects 1 bit per 7-bit block",
    },
    "repetition3": {
        "encode": lambda bits: repetition.encode(bits, repetitions=3),
        "decode": lambda bits: repetition.decode(bits, repetitions=3),
        "rate": 1 / 3,
        "description": "Rate-1/3 repetition with majority-vote decoding",
    },
    "conv12": {
        "encode": lambda bits: convolutional.encode(bits, tail=True),
        "decode": lambda bits: convolutional.decode(bits),
        "rate": 1 / 2,
        "description": "K=7 rate-1/2 convolutional (171,133) with Viterbi decoding",
    },
}


def list_schemes() -> tuple[str, ...]:
    """Names of all registered FEC schemes."""
    return tuple(_SCHEMES)


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


def decode_bits(bits: np.ndarray, scheme: str) -> tuple[np.ndarray, FECResult]:
    """Decode bits with the named scheme; returns (decoded, FECResult)."""
    if scheme not in _SCHEMES:
        raise FECError(f"Unknown FEC scheme '{scheme}'. Available: {sorted(_SCHEMES)}")
    try:
        decoded, stats = _SCHEMES[scheme]["decode"](bits)
    except ValueError as exc:
        raise FECError(f"{scheme} decode failed: {exc}") from exc
    result = FECResult(
        scheme=scheme,
        operation="decode",
        input_bits=int(bits.size),
        output_bits=int(decoded.size),
        code_rate=float(_SCHEMES[scheme]["rate"]),
        corrected_errors=int(stats.get("corrected_errors", 0)),
        uncorrectable_blocks=int(stats.get("uncorrectable_blocks", 0)),
        extra={k: v for k, v in stats.items()
               if k not in ("corrected_errors", "uncorrectable_blocks")},
    )
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
