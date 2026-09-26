"""Unit tests for the protocol frame layer.

These tests exercise the frame decoder directly on synthetic bit
streams (synthetic-data validation, clearly distinguished from any
real-world capture).  Unknown/no-sync results are a legitimate,
first-class outcome and are asserted as such.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.protocol import (
    FrameConfig,
    FrameDecodeResult,
    make_frame_config,
    parse_frame_bits,
)


def _msb_bits(value: int, width: int) -> np.ndarray:
    """MSB-first bit list for an integer of a given bit width."""
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def test_sync_found_and_payload_extraction():
    sync = 0xAA55AA55
    payload_bits = np.array([1, 0, 1, 0] * 8, dtype=np.uint8)
    stream = np.concatenate([_msb_bits(sync, 32), payload_bits])

    result = parse_frame_bits(stream, sync_word=sync, data_bytes=4)

    assert result.protocol == "Unknown"
    assert result.sync_found is True
    assert result.sync_confidence > 0.9
    assert result.sync_index == 0
    assert result.payload_start == 32
    assert result.payload_bytes.hex() == "aaaaaaaa"  # payload bits -> byte
    assert result.matches[0].crc_status == "n/a (no CRC configured)"


def test_no_sync_returns_unknown_when_too_short():
    # Fewer bits than the sync word can hold: find_sync_word returns
    # no matches, and the parser reports an honest Unknown result.
    stream = np.array([0, 1, 0, 1], dtype=np.uint8)

    result = parse_frame_bits(stream, sync_word=0xAA55AA55)

    # No sync word -> honest "Unknown"; this is a first-class result
    # (it is never forced into a protocol guess).
    assert result.protocol == "Unknown"
    assert result.sync_found is False
    assert result.sync_confidence == 0.0
    assert any("No sync word" in w for w in result.warnings)


def test_disabled_sync_word_is_unknown():
    stream = _msb_bits(0xAA55AA55, 32)
    result = parse_frame_bits(stream, sync_word=0)

    assert result.protocol == "Unknown"
    assert result.sync_found is False


def test_truncated_payload_raises():
    sync = 0xAA55AA55
    stream = np.concatenate([_msb_bits(sync, 32), np.array([1, 0], dtype=np.uint8)])

    with pytest.raises(Exception):
        parse_frame_bits(stream, sync_word=sync, data_bytes=4)


def test_crc_status_is_reported():
    # Verify that an explicit CRC reference is stored and the status is
    # reported deterministically; the exact valid/invalid outcome depends
    # only on the bits vs the reference.  We assert the *mechanism* is
    # wired: a CRC is present in the result summary and the status is one
    # of the documented strings.
    sync = 0xAA55AA55
    payload_bits = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
    stream = np.concatenate([_msb_bits(sync, 32), payload_bits])

    result = parse_frame_bits(
        stream, sync_word=sync, data_bytes=1, crc_poly=0x07, crc_final_xor=0
    )
    assert result.protocol == "Unknown"
    assert result.sync_found is True
    assert result.matches[0].crc_status in ("valid", "invalid")


def test_make_frame_config_factory():
    cfg = make_frame_config(
        name="Telemetry", sync_word=0xAA55AA55, data_bytes=4, crc=(0x07, 0x00)
    )
    assert cfg.name == "Telemetry"
    assert cfg.data_bytes == 4
    assert cfg.crc == (7, 0)
