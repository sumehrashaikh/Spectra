"""
Forward error correction: CRC, Hamming(7,4), repetition, convolutional
(K=7, rate 1/2) with Viterbi decoding, and block interleaving.

FEC usage is always *explicit* configuration, never guessed from data.
"""

from .framework import (
    FECResult,
    apply_decode,
    apply_encode,
    decode_bits,
    describe_scheme,
    encode_bits,
    list_schemes,
)
from .interleaving import deinterleave_bits, interleave_bits

__all__ = [
    "FECResult",
    "apply_decode",
    "apply_encode",
    "decode_bits",
    "describe_scheme",
    "encode_bits",
    "list_schemes",
    "interleave_bits",
    "deinterleave_bits",
]
