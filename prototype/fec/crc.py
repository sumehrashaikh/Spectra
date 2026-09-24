"""CRC computation: CRC-16/CCITT-FALSE and CRC-32 (zlib)."""

from __future__ import annotations

import zlib

import numpy as np

CRC16_POLY = 0x1021  # CCITT
CRC16_INIT = 0xFFFF


def crc16_ccitt(data: bytes, poly: int = CRC16_POLY, init: int = CRC16_INIT) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection)."""
    crc = init & 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc32(data: bytes) -> int:
    """CRC-32 (IEEE 802.3, as used by zlib)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a 0/1 bit array (MSB first) into bytes, zero-padding the tail."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return b""
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary 0/1 values.")
    padding = (-bits.size) % 8
    padded = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])
    packed = np.packbits(padded, bitorder="big")
    return packed.tobytes()


def bytes_to_bits(data: bytes, num_bits: int | None = None) -> np.ndarray:
    """Unpack bytes into a 0/1 bit array (MSB first)."""
    unpacked = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    return unpacked[:num_bits] if num_bits is not None else unpacked


def crc_append(
    bits: np.ndarray,
    scheme: str = "crc16",
) -> np.ndarray:
    """Append the CRC of ``bits`` (MSB-first) as extra bits."""
    data = bits_to_bytes(bits)
    if scheme == "crc16":
        value = crc16_ccitt(data)
        width = 16
    elif scheme == "crc32":
        value = crc32(data)
        width = 32
    else:
        raise ValueError(f"Unknown CRC scheme '{scheme}'. Use 'crc16' or 'crc32'.")
    crc_bits = np.array(
        [(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8
    )
    return np.concatenate([np.asarray(bits, dtype=np.uint8), crc_bits])


def crc_check(
    bits: np.ndarray,
    scheme: str = "crc16",
) -> bool:
    """Validate a bit stream that ends with an appended CRC."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    width = 16 if scheme == "crc16" else 32
    if scheme not in ("crc16", "crc32"):
        raise ValueError(f"Unknown CRC scheme '{scheme}'.")
    if bits.size < width:
        raise ValueError("Bit stream too short to contain a CRC.")

    payload = bits[:-width]
    received_crc_bits = bits[-width:]

    expected = crc_append(payload, scheme)[-width:]
    return bool(np.array_equal(received_crc_bits, expected))
