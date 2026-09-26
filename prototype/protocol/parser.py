"""Frame decoder for Spectra V2.

Receives a demodulated bit stream (optionally post-FEC) and searches it
for a known sync word / preamble.  Extracts payload bits/bytes and
optionally validates a CRC.  Never forces a protocol: if no sync word
matches, the result is ``Unknown`` / no-frame.

The frame layer is explicit configuration, not inference: it only runs
when a :class:`FrameConfig` is attached to
``AnalysisConfig.protocol``.  It reuses the normalized-correlation
sync-word search in ``dsp.correlation.find_sync_word`` so the same
threshold / non-maximum-suppression / guard-band model the pipeline
already ships is used.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from prototype.dsp.correlation import find_sync_word

from .config import FrameConfig

_MAX_MATCHES = 16


class FrameParseError(Exception):
    """The bit stream looks like a frame (sync word present) but the
    declared structure or CRC does not validate."""


@dataclass(frozen=True)
class FrameMatch:
    """One successfully decoded frame."""

    protocol: str
    sync_index: int
    payload_start: int
    payload_bytes: bytes
    crc_valid: bool
    crc_status: str

    @property
    def matched_bits(self) -> int:
        return self.sync_index + self.sync_word_bits


@dataclass(frozen=True)
class FrameDecodeResult:
    """Aggregate result of a frame analysis run.

    ``sync_found=False`` is a legitimate, first-class result
    (``"Unknown"``), never a failure to be hidden.
    """

    protocol: str
    sync_found: bool
    sync_confidence: float
    matches: list[FrameMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- frame semantics (optional, present when the semantics
    # --- decoder ran on a matched frame) ---------------------------------------
    header_bits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    header_bytes: int = 0
    frame_type_bits: int | None = None
    frame_type: int | None = None
    payload_len_bits: int | None = None
    payload_bits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.uint8)
    )

    # --- CRC / payload summary (mirrored from the match for convenience) -----
    crc_valid: bool = False
    crc_status: str = "n/a (no CRC configured)"

    @property
    def sync_index(self) -> int:
        return self.matches[0].sync_index if self.matches else -1

    @property
    def payload_start(self) -> int:
        return self.matches[0].payload_start if self.matches else -1

    @property
    def payload_bytes(self) -> bytes:
        return self.matches[0].payload_bytes if self.matches else b""


def _bit_list_to_bytes(bits: np.ndarray) -> bytes:
    """MSB-first integer encode of a bit array into bytes."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return b""
    padded = np.zeros((bits.size + 7) // 8 * 8, dtype=np.uint8)
    padded[: bits.size] = bits
    value = 0
    for b in padded:
        value = (value << 1) | int(b)
    return value.to_bytes(len(padded) // 8, "big", signed=False)


def _sync_word_nbits(sync_word: int) -> int:
    """Length in bits of the declared sync word (for alignment)."""
    if sync_word == 0:
        return 0
    return max(1, sync_word.bit_length())


def find_frame_signal_indices(
    samples: np.ndarray,
    sync_word: int,
    threshold: float = 0.6,
    max_matches: int = _MAX_MATCHES,
) -> list[dict]:
    """Search a *sampled* (complex) waveform for a known sync-word
    template via normalized correlation.  Returns the match dicts with
    sample indices.

    This is the signal-level primitive.  ``parse_frame_bits`` performs
    the downstream bit/byte decoding on a demodulated bit stream.
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        return []

    # Build a one-hot template of the sync word in transmission order.
    n = _sync_word_nbits(sync_word)
    template = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (sync_word >> (n - 1 - i)) & 1:
            template[i] = 1.0

    return find_sync_word(
        samples,
        template,
        threshold=threshold,
        max_matches=max_matches,
    )


def parse_frame_bits(
    bits: np.ndarray,
    sync_word: int,
    data_bytes: int = 0,
    crc_poly: int | None = None,
    crc_final_xor: int | None = None,
    crc_orientation: str = "msb_first",
    max_matches: int = _MAX_MATCHES,
) -> FrameDecodeResult:
    """Decode frames from a demodulated bit stream (MSB-first).

    Args
    ------
    bits
        Demodulated bit stream (uint8, 0/1).  First transmitted bit is
        the most-significant bit of the first byte.
    sync_word
        Preamble/sync-sequence integer (MSB-first).  Zero disables
        sync-word matching.
    data_bytes
        Expected user-data payload in bytes after the sync word.
    crc_poly, crc_final_xor
        Optional CRC reference.  If the received bits fail the CRC the
        payload is cleared and ``crc_valid=False`` is reported.
    crc_orientation
        ``"msb_first"`` (default) or ``"lsb_first"``.
    max_matches
        Maximum sync-word occurrences to report.

    Returns
    -------
    FrameDecodeResult
        ``sync_found=False`` if no sync word was detected (honest
        Unknown).  Otherwise one or more decoded frames.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return FrameDecodeResult(
            protocol="Unknown", sync_found=False, sync_confidence=0.0
        )

    if sync_word == 0:
        # Legacy "no sync" mode: report the stream as unknown frames.
        return FrameDecodeResult(
            protocol="Unknown",
            sync_found=False,
            sync_confidence=0.0,
            warnings=["Sync word disabled (sync_word=0); no frame extracted."],
        )

    # ---- sync-word search on the demodulated bit stream ---------------
    # Reuse the normalized-correlation, threshold, and
    # non-maximum-suppression path the pipeline already ships.
    n = _sync_word_nbits(sync_word)
    template = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (sync_word >> (n - 1 - i)) & 1:
            template[i] = 1.0

    matched = find_sync_word(
        bits.astype(np.float64),
        template,
        threshold=0.6,
        max_matches=max_matches,
    )

    # No sync word detected -> honest "Unknown".
    if not matched:
        return FrameDecodeResult(
            protocol="Unknown",
            sync_found=False,
            sync_confidence=0.0,
            warnings=["No sync word detected in the demodulated bit stream."],
        )

    # The first match is the authoritative frame (the one we try to
    # decode).  Remaining matches are also returned for spectrum-
    # monitoring use.
    primary = matched[0]
    sync_index = int(primary["index"])

    # Payload begins after the sync word.
    payload_start = sync_index + n

    payload_bits = bits[payload_start:]
    payload_bytes = b""
    crc_valid = False
    crc_status = "none"

    if data_bytes > 0:
        if payload_bits.size < data_bytes * 8:
            raise FrameParseError(
                "Frame truncated after sync word: expected "
                f"{data_bytes * 8} payload bits, got {payload_bits.size}."
            )
        payload_bits = payload_bits[: data_bytes * 8]
        payload_bytes = _bit_list_to_bytes(payload_bits)

        if crc_poly is not None and crc_final_xor is not None:
            computed = _crc_bits_to_int(payload_bits, crc_poly, crc_final_xor)
            crc_valid = computed == crc_final_xor
            crc_status = "valid" if crc_valid else "invalid"
        else:
            crc_status = "n/a (no CRC configured)"
    else:
        payload_bytes = _bit_list_to_bytes(payload_bits)
        crc_status = "n/a (no CRC configured)"

    return FrameDecodeResult(
        protocol="Unknown",
        sync_found=True,
        sync_confidence=float(primary["metric"]),
        matches=[
            FrameMatch(
                protocol="unknown",
                sync_index=sync_index,
                payload_start=payload_start,
                payload_bytes=payload_bytes,
                crc_valid=crc_valid,
                crc_status=crc_status,
            )
        ],
        warnings=list(primary.get("warnings", [])),
    )


def _crc_bits_to_int(bits: np.ndarray, poly: int, final_xor: int) -> int:
    """CRC over the bit stream in MSB-first transmission order."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    width = poly.bit_length() - 1
    remainder = 0
    mask = (1 << width) - 1
    for b in bits:
        remainder = ((remainder << 1) | int(b)) & mask
        if remainder & (1 << (width - 1)):
            remainder ^= poly
    return remainder ^ int(final_xor)
