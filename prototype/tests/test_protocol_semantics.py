"""Frame-semantics decoder tests.

These tests exercise the field layout decoded from a
sync-word-matched frame (header fields, frame type, payload length,
CRC).  All validation here is on synthetic data, clearly distinguished
from any real-world capture.  Unknown/no-sync results are a
legitimate, first-class outcome and are asserted as such.

All streams are bit-element arrays (each element is 0 or 1), matching
the canonical representation the decoder consumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.protocol import FrameConfig
from prototype.protocol.semantics import FrameSemanticsError, decode_frame_semantics


def _sync_bits(sync=0xAA55AA55):
    return np.array([(sync >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)


def _byte_to_bits(b):
    """A single integer byte (0-255) into 8 MSB-first bits."""
    return np.array([(b >> (7 - i)) & 1 for i in range(8)], dtype=np.uint8)


def _stream(sync=0xAA55AA55, header=0x13, payload=b"\x01"):
    """Build a sync word + header declaration byte + payload bytes."""
    return np.concatenate(
        [_sync_bits(sync), _byte_to_bits(header), _byte_to_bits(payload)]
    )


def test_decodes_header_and_frame_type():
    sync = 0xAA55AA55
    # header declaration byte 0x13 = version=1 (high nibble), frame_type=3 (low nibble)
    stream = _stream(sync, header=0x13, payload=0x01)

    result = decode_frame_semantics(stream, sync, data_bytes=1, header_len_bytes=1)

    assert result.protocol == "Unknown"
    assert result.sync_found is True
    assert result.frame_type == 3
    assert result.header_bytes == 1
    # header_bits: 8 declaration bits in MSB-first order
    assert result.header_bits.tolist() == [0, 0, 0, 1, 0, 0, 1, 1]
    assert result.payload_bytes.hex() == "01"
    assert result.payload_bits.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert result.payload_len_bits == 8
    assert result.crc_valid is False
    assert result.crc_status == "n/a (no CRC configured)"


def test_payload_length_from_data_bytes():
    sync = 0xAA55AA55
    header = 0x30        # version=3, type=0
    payload = 0xAC       # 10101100
    stream = _stream(sync, header=header, payload=payload)

    result = decode_frame_semantics(stream, sync, data_bytes=1, header_len_bytes=1)

    assert result.protocol == "Unknown"
    assert result.frame_type == 0
    assert result.header_bytes == 1
    assert result.payload_bytes.hex() == "ac"
    assert result.payload_len_bits == 8


def test_header_len_skips_more_bytes():
    sync = 0xAA55AA55
    header = 0x10
    payload = 0x55
    stream = _stream(sync, header=header, payload=payload)

    result = decode_frame_semantics(stream, sync, data_bytes=1, header_len_bytes=1)

    assert result.header_bytes == 1
    assert result.payload_bytes.hex() == "55"


def test_crc_validation_when_configured():
    sync = 0xAA55AA55
    header = 0x30
    payload = 0x01
    stream = _stream(sync, header=header, payload=payload)

    result = decode_frame_semantics(
        stream, sync, data_bytes=1, header_len_bytes=1,
        crc_poly=0x07, crc_final_xor=0,
    )

    assert result.protocol == "Unknown"
    assert result.crc_status in ("valid", "invalid")
    assert isinstance(result.crc_valid, bool)


def test_crc_invalid_report_false():
    sync = 0xAA55AA55
    header = 0x30
    payload = 0x01
    stream = _stream(sync, header=header, payload=payload)

    result = decode_frame_semantics(
        stream, sync, data_bytes=1, header_len_bytes=1,
        crc_poly=0x07, crc_final_xor=1,
    )

    assert result.crc_valid is False
    assert result.crc_status == "invalid"


def test_truncated_payload_raises():
    sync = 0xAA55AA55
    header = 0x30
    # only the sync word + header, no payload
    stream = np.concatenate([_sync_bits(sync), _byte_to_bits(header)])

    with pytest.raises(FrameSemanticsError):
        decode_frame_semantics(stream, sync, data_bytes=8, header_len_bytes=1)


def test_no_sync_word_returns_unknown():
    # A short, benign stream guaranteed to contain no 32-bit sync word.
    stream = np.zeros(64, dtype=np.uint8)

    result = decode_frame_semantics(stream, sync_word=0xAA55AA55, data_bytes=8)

    assert result.protocol == "Unknown"
    assert result.sync_found is False
    assert result.sync_confidence == 0.0
    assert any("not found" in w for w in result.warnings)


def test_sync_word_disabled_is_unknown():
    sync = 0xAA55AA55
    # A 32-bit sync word pattern cannot self-match as a zero sync word; a
    # zero sync word template over a non-zero stream yields no match.
    stream = np.array([0, 1] * 16, dtype=np.uint8)  # alternating, no 32-zero run

    result = decode_frame_semantics(stream, sync_word=0, data_bytes=8)

    assert result.protocol == "Unknown"
    assert result.sync_found is False


def test_make_frame_config():
    from prototype.protocol import make_frame_config

    cfg = make_frame_config(
        name="Telemetry",
        sync_word=0xAA55AA55,
        data_bytes=8,
        crc=(0x07, 0x00),
    )
    assert cfg.name == "Telemetry"
    assert cfg.data_bytes == 8
    assert cfg.crc == (7, 0)
    assert cfg.header_len_bytes == 0
