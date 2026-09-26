"""Declarative frame definitions (config, not inference).

A protocol is described by a :class:`FrameConfig`, which the pipeline
applies to a demodulated bit stream.  Add a protocol by constructing a
new dataclass and registering it in the GUI's FEC-style selector — no
pipeline changes required.

Design
------
- ``sync_word`` is an integer, MSB-first (the first bit of the integer
  is the first transmitted bit).  It must be 0 or a power of two so
  bit-boundary alignment after sync detection is deterministic.
- ``data_bytes`` is the expected user-data payload size in bytes,
  after the sync word (and any fixed header).  If the received bit
  count after the sync word is not a multiple of 8, the frame layer
  reports an error and leaves the bits unmodified.
- ``crc`` is a ``(poly, final_xor)`` pair, or ``None`` if no CRC is
  expected.  CRC is validated on the last ``data_bytes`` bytes of the
  payload.  A failing CRC means the frame did not come from this
  protocol — it is reported honestly, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from prototype.core.exceptions import SpectraError

_CRCType = tuple[int, int]


@dataclass(frozen=True)
class FrameConfig:
    """Describes one known frame/packet format that may appear inside a
    demodulated bit stream.

    Parameters
    ----------
    name
        Human-readable protocol name shown in results and the GUI.
    sync_word
        The preamble/sync-sequence as an integer, MSB-first.  The first
        bit of the integer is the first transmitted bit.  The exact
        bit pattern (e.g. 0xAA55AA55 for a classic 32-bit sync word)
        is matched by normalized correlation; no power-of-two constraint
        is applied, so realistic sync words are supported.

        Use 0 to disable sync-word matching (legacy "no sync" mode).
    sync_word_capacity
        Number of bits in the sync word (default 32).  Used to re-align
        the stream at the *next* sync occurrence instead of at the
        payload start.  Only the documented 32-bit capacity is
        supported so extraction stays unambiguous.
    data_bytes
        Expected user-data payload size in bytes after the sync word.
        If the received bit count after the sync word is not a
        multiple of 8, extraction reports an error and leaves the bits
        unmodified.
    crc
        ``(poly, final_xor)`` pair, or ``None`` if no CRC is expected.
        The CRC is checked against the last ``data_bytes`` bytes of the
        payload.
    crc_orientation
        How the CRC field is laid out.  ``"msb_first"`` (default) means
        the most-significant bit of the first CRC byte is transmitted
        first; ``"lsb_first"`` swaps it.
    payload_type
        Downstream semantic hint shown in the GUI (e.g. ``"telemetry"``,
        ``"control"``).  Descriptive only; the frame layer never
        switches decoding strategy on it.
    description
        Optional prose shown in the GUI/README for this protocol.
    """

    name: str
    sync_word: int
    sync_word_capacity: int = 32
    data_bytes: int = 0
    crc: _CRCType | None = None
    crc_orientation: Literal["msb_first", "lsb_first"] = "msb_first"
    payload_type: str = ""
    description: str = ""

    # -- frame-semantics (optional, for the semantic decoder) ------------
    # These parameters describe the field layout the transmitter
    # embeds after the sync word.  A value of None/0 means "use the
    # canonical defaults" for a simple frame.
    header_len_bytes: int = 0  # header bytes after sync word (0 = canonical)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FrameConfig.name must not be empty.")

        if self.sync_word < 0:
            raise ValueError("FrameConfig.sync_word must not be negative.")
        if self.sync_word < 0:
            raise ValueError("FrameConfig.sync_word must not be negative.")

        # sync_word_capacity is fixed at 32 to keep the bit-boundary
        # between the sync word and the payload start deterministic.
        if self.sync_word_capacity <= 0 or self.sync_word_capacity != 32:
            raise ValueError(
                "FrameConfig.sync_word_capacity must be 32 (power-of-two "
                "single-frame alignment only)."
            )

        if self.data_bytes < 0:
            raise ValueError("FrameConfig.data_bytes must not be negative.")

        if self.crc is not None:
            poly, final_xor = self.crc
            if poly < 0 or final_xor < 0:
                raise ValueError("CRC poly and final xor must be non-negative.")

        if self.crc_orientation not in ("msb_first", "lsb_first"):
            raise ValueError(
                "FrameConfig.crc_orientation must be 'msb_first' or 'lsb_first'."
            )


@dataclass(frozen=True)
class FrameReference:
    """Shared parameters produced by the pipeline and consumed by the
    frame parser.  This is the single extension point for adding a
    protocol to the V2 chain without changing pipeline plumbing.
    """

    protocol_name: str
    sync_word: int
    sync_word_capacity: int = 32
    data_bytes: int = 0
    crc_poly: int | None = None
    crc_final_xor: int | None = None
    crc_orientation: Literal["msb_first", "lsb_first"] = "msb_first"
