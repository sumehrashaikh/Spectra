"""Protocol / frame analysis for Spectra V2.

This package builds protocol/frame decoding **on top of** the shared
V2 pipeline — it receives the demodulated bit stream (optionally after
an explicit FEC pass) and searches for a known sync word / preamble.
It never invents a protocol: when no sync word matches, the result is
``Unknown``/no-frame rather than a guessed one.

Design notes
------------
- Frame definitions are plain, declarative dataclasses (``FrameConfig``)
  so a protocol can be added without touching pipeline code.  Each
  protocol declares:
  ``name``, ``sync_word`` (integers, MSB-first), optional
  ``sync_word_capacity``, optional ``data_bytes``, and optional
  ``expected_crc`` settings.
- Matching reuses the existing normalized-correlation sync-word search
  in ``dsp.correlation.find_sync_word`` (same threshold model), which
  returns the sample index where the sync word starts.
- The frame layer is **explicit configuration, not inference**.  It
  only runs when a ``FrameConfig`` is attached to
  ``AnalysisConfig.protocol``.  There is no "auto-detect protocol".

Add a protocol
--------------
from prototype.protocol.parser import parse_frame_bits
from prototype.protocol import FrameConfig

cfg = FrameConfig(
    name="MyFrame",
    sync_word=0xAA55AA55,
    sync_word_capacity=4,   # 4-byte sync word (optional)
    data_bytes=16,          # user-data payload after the sync word
    expected_crc=None,      # (poly, final xor) or None
)
payload = parse_frame_bits(
    bits=received_bits,
    sync_word=cfg.sync_word,
    data_bytes=cfg.data_bytes,
    expected_crc=cfg.expected_crc,
    sync_word_capacity=cfg.sync_word_capacity,
)
"""

from __future__ import annotations

from .parser import (
    FrameDecodeResult,
    FrameMatch,
    FrameParseError,
    find_frame_signal_indices,
    parse_frame_bits,
)
from .semantics import decode_frame_semantics, FrameSemanticsError
from .config import FrameConfig

__all__ = [
    "FrameConfig",
    "FrameDecodeResult",
    "FrameMatch",
    "FrameParseError",
    "FrameSemanticsError",
    "find_frame_signal_indices",
    "parse_frame_bits",
    "decode_frame_semantics",
]


def make_frame_config(
    name: str,
    sync_word: int,
    *,
    sync_word_capacity: int = 32,
    data_bytes: int = 0,
    crc: tuple[int, int] | None = None,
    crc_orientation: str = "msb_first",
    payload_type: str = "",
    description: str = "",
) -> FrameConfig:
    """Build a :class:`FrameConfig` with positional safety for the
    required, protocol-unique fields (``name``, ``sync_word``).  All
    other fields are keyword-only so a protocol's shape is explicit.

    Example
    -------
    >>> cfg = make_frame_config(
    ...     "DemoTelemetry", 0xAA55AA55, data_bytes=4, crc=(0x07, 0x00)
    ... )
    """
    return FrameConfig(
        name=name,
        sync_word=sync_word,
        sync_word_capacity=sync_word_capacity,
        data_bytes=data_bytes,
        crc=crc,
        crc_orientation=crc_orientation,
        payload_type=payload_type,
        description=description,
    )

