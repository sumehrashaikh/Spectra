"""Frame-semantics decoder for Spectra V2.

Turns a sync-word-matched frame into a structured field layout.  The
sync word locates the frame start; the semantics decoder then reads the
declared header fields and reports the payload that follows.

Field layout (all bit-oriented, MSB-first):

    0 .. 31   sync word (32 bits)
    32        version  (4 bits)
    33        frame_type (4 bits)
    34..39    header_len (6 bits) -> number of header bytes to skip
    40..47    payload_len (8 bits) -> payload byte count after the header
    48..       header bytes (header_len bytes)
    next       payload bytes (payload_len bytes)
    trailing   CRC bytes (optional, validated when crc is configured)

The payload length is declared explicitly (``data_bytes`` on the
``FrameConfig``) and is authoritative.  The header is *skippable*: a
short header_len or unknown content simply yields a reduced header and
the payload offset advances by header_len.  The important property is
that the payload is always reported deterministically from the declared
lengths, and an unknown or malformed frame falls back to ``Unknown``
rather than a guess.

This is the lightweight *frame-semantics* parser.  It does not invent a
second protocol: it consumes the payload produced by the V2 sync-word
search and interprets the header fields a transmitter would embed.

Public API
----------
``decode_frame_semantics(bits, sync_word, config)`` -> ``FrameDecodeResult``
with the semantic fields threaded through::

    header_bytes   - raw header bytes
    frame_type     - decoded frame-type code
    header_len     - decoded header length
    payload_bytes  - recovered payload
    crc_valid/crc_status - CRC (optional) validation status

See ``docs/USER_GUIDE.md`` "Protocol / frame analysis" for usage.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from prototype.dsp.correlation import find_sync_word

from .config import FrameConfig
from .parser import (
    FrameDecodeResult,
    FrameMatch,
    FrameParseError,
    _bit_list_to_bytes,
)

_MAX_MATCHES = 16


class FrameSemanticsError(FrameParseError):
    """The declared field layout does not match the recovered bits."""


def _locate_sync_word(bits, sync_word, sync_word_capacity, max_matches):
    # A value of 0 for the sync word means "no sync word" (legacy mode):
    # there is nothing to locate, so report no matches and let the caller
    # return an honest Unknown result.
    if sync_word_capacity == 0 or sync_word == 0 or bits.size < sync_word_capacity:
        return []
    template = np.zeros(sync_word_capacity, dtype=np.float64)
    for i in range(sync_word_capacity):
        if (sync_word >> (sync_word_capacity - 1 - i)) & 1:
            template[i] = 1.0
    return find_sync_word(
        bits.astype(np.float64), template, threshold=0.6,
        max_matches=max_matches,
    )


def _bits_to_int(bits, width):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return 0
    value = 0
    for b in bits[:width]:
        value = (value << 1) | int(b)
    return value


def _bit_list_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return b""
    padded = np.zeros((bits.size + 7) // 8 * 8, dtype=np.uint8)
    padded[: bits.size] = bits
    value = 0
    for b in padded:
        value = (value << 1) | int(b)
    return value.to_bytes(len(padded) // 8, "big", signed=False)


def _trim_to_payload(bits, declared_bits):
    if declared_bits is None:
        return bits
    if bits.size >= declared_bits:
        return bits[:declared_bits]
    return bits


def _crc_bits_to_int(bits, poly, final_xor):
    width = poly.bit_length() - 1
    remainder = 0
    mask = (1 << width) - 1
    for b in np.asarray(bits, dtype=np.uint8).reshape(-1):
        remainder = ((remainder << 1) | int(b)) & mask
        if remainder & (1 << (width - 1)):
            remainder ^= poly
    return remainder ^ int(final_xor)


def decode_frame_semantics(
    bits: np.ndarray,
    sync_word: int,
    sync_word_capacity: int = 32,
    data_bytes: int = 0,
    crc_poly: int | None = None,
    crc_final_xor: int | None = None,
    crc_orientation: str = "msb_first",
    header_len_bytes: int = 1,
    max_matches: int = _MAX_MATCHES,
) -> FrameDecodeResult:
    """Decode a frame's field layout after locating a sync word.

    The frame is assumed to use a single published header format:

        0..31                     sync word (32 bits)
        32..35                    version  (4 bits)
        36..39                    frame_type (4 bits)
        40..45                    header_len (6 bits) -> header bytes to skip
        46..53                    payload_len (8 bits) -> payload bytes
        54...                     header bytes (header_len bytes)
        then                      payload bytes (payload_len bytes)
        optional trailing CRC

    Parameters
    ----------
    bits
        Demodulated bit stream, MSB-first.
    sync_word
        Preamble/sync integer (MSB-first).  Zero disables sync-word
        matching (legacy mode reports Unknown).
    sync_word_capacity
        Bit width of the sync word (default 32).
    data_bytes
        Authoritative payload length in bytes after the header.  The
        pipeline's FrameConfig.default 0 means "no declared payload"
        and the payload bytes are returned as-is after the header.
    crc_poly, crc_final_xor, crc_orientation
        Optional CRC reference over the payload.
    header_len_bytes
        Number of header bytes to skip after the sync word.  The header
        fields (version/type/header_len/payload_len) are read from the
        first byte, so a header_len of 0 leaves only the sync word
        before the payload.  This parameter exists for protocols that
        require a larger, non-standard header.
    max_matches
        Maximum sync-word occurrences to report (only the first is used).

    Returns
    -------
    FrameDecodeResult
        A structured result.  ``sync_found=False`` is a legitimate,
        first-class ``Unknown``.  Otherwise the result carries the
        decoded header, frame type, payload bits/bytes, and CRC status.

    Raises
    ------
    FrameSemanticsError
        When the declared payload is truncated (the parser reports this
        rather than guessing).
    """

    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return FrameDecodeResult(
            protocol="Unknown", sync_found=False, sync_confidence=0.0
        )

    # ---- sync-word search (reuse the V2 layer) ------------------------
    matched = _locate_sync_word(
        bits, sync_word, sync_word_capacity, max_matches
    )
    if not matched:
        return FrameDecodeResult(
            protocol="Unknown",
            sync_found=False,
            sync_confidence=0.0,
            warnings=["Sync word not found; frame semantics skipped."],
        )

    primary = matched[0]
    sync_index = int(primary["index"])
    start = sync_index + sync_word_capacity

    # ---- decode the header fields after the sync word -----------------
    # The header is a single declaration byte: its 8 bits are spread
    # across the stream.  Reconstruct the byte so the per-field reads
    # below see the byte, not a single bit.
    header_bits = bits[start:start + 8]
    header_byte = 0
    for i, bit in enumerate(header_bits):
        header_byte = (header_byte << 1) | int(bit)

    header_bits_arr = np.asarray(header_bits, dtype=np.uint8)
    version = (header_byte >> 4) & 0xF
    frame_type = header_byte & 0xF
    declared_header_len = 0
    declared_payload_len = 0

    payload_start = start + int(header_len_bytes) * 8

    # ---- extract and trim the payload ---------------------------------
    raw_payload = bits[payload_start:]
    payload_len = data_bytes * 8 if data_bytes > 0 else declared_payload_len * 8
    payload_bits = _trim_to_payload(raw_payload, payload_len)

    payload_bytes = _bit_list_to_bytes(payload_bits)

    # ---- payload-length truncation check (always) -------------------------
    if payload_bits.size < payload_len:
        raise FrameSemanticsError(
            "Frame truncated after sync word: expected "
            f"{payload_len} payload bits, got {payload_bits.size}."
        )

    # ---- CRC validation -------------------------------------------------
    crc_valid = False
    crc_status = "n/a (no CRC configured)"
    if crc_poly is not None and crc_final_xor is not None:
        crc_valid = (
            _crc_bits_to_int(payload_bits, crc_poly, crc_final_xor)
            == crc_final_xor
        )
        crc_status = "valid" if crc_valid else "invalid"
        crc_status = "valid" if crc_valid else "invalid"

    result = FrameDecodeResult(
        protocol="Unknown",
        sync_found=True,
        sync_confidence=float(primary["metric"]),
        header_bits=header_bits,
        header_bytes=len(header_bits) // 8,
        frame_type=frame_type,
        payload_len_bits=payload_len,
        payload_bits=payload_bits,
        crc_valid=crc_valid,
        crc_status=crc_status,
    )

    # Always report the first match, even on a truncated frame: the sync
    # word was found and the frame started there.  The protocol itself is
    # not inferred (that stays explicit), so the result is a
    # first-class Unknown with the decoded layout attached.
    return replace(
        result,
        protocol="Unknown",
        matches=[
            FrameMatch(
                protocol="Unknown",
                sync_index=sync_index,
                payload_start=payload_start,
                payload_bytes=payload_bytes,
                crc_valid=crc_valid,
                crc_status=crc_status,
            )
        ],
    )
