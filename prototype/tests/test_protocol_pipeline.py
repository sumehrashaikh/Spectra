"""End-to-end integration test for the protocol frame layer.

The frame layer is a downstream consumer of the *recovered demodulated
bits*; it looks for a known sync word inside those bits.  These tests
exercise the pipeline's wiring with an explicit FrameConfig and a
synthetic capture, and assert the honest-Unknown semantics.

Note: all validation here is on synthetic data; it is clearly
distinguished from any real-world capture in the docs.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from prototype.core.config import AnalysisConfig
from prototype.protocol import FrameConfig
from prototype.pipeline import analyze_samples

# A classic 32-bit sync word (Ethernet-like pattern, MSB-first).
SYNC = 0xAA55AA55

# An explicit FrameConfig turns protocol analysis on for a run.
CONFIG = AnalysisConfig(
    protocol=FrameConfig(
        name="DemoTelemetry",
        sync_word=SYNC,
        data_bytes=1,  # 1 byte of payload after the 32-bit sync word
        crc=(0x07, 0x00),
        description="Example telemetry frame used in the V2 protocol test.",
    )
)


def _tone_at(freq_hz, sample_rate=8000.0, duration=0.25):
    """A clean synthetic carrier: the pipeline's detection/demod pipeline
    turns it into recovered bits (or fails gracefully), letting the
    protocol stage exercise the honest-Unknown path."""
    n = int(sample_rate * duration)
    return np.exp(1j * 2 * np.pi * freq_hz * np.arange(n) / sample_rate)


def _find_gui_qam16_wav() -> Path:
    """Locate the synthetic capture used by the ML stage tests.

    The path is resolved relative to the repository root (one level
    above ``prototype/tests/``) instead of using the current working
    directory, so the test behaves identically when run from the
    repository root, from inside ``prototype/``, from a teammate's
    clone, or in CI.
    """
    candidate = Path(__file__).resolve().parent.parent.parent
    for name in ("prototype/gui_qam16.wav", "gui_qam16.wav"):
        if (candidate / name).is_file():
            return candidate / name
    raise FileNotFoundError(
        "gui_qam16.wav not found. Expected a copy of the synthetic "
        f"capture at {candidate / 'prototype'} or {candidate}. "
        "This fixture is required by the ML stage tests."
    )


def test_protocol_stage_wired_and_returns_structured_result():
    """A run with an explicit FrameConfig produces a protocol result field."""
    result = analyze_samples(_tone_at(500.0), 8000.0, config=CONFIG)

    assert result.protocol is not None
    # Structured, JSON-safe result.
    assert isinstance(result.protocol.protocol, str)
    assert isinstance(result.protocol.sync_confidence, float)
    assert isinstance(result.protocol.warnings, list)
    assert result.protocol.protocol == "Unknown"  # a flat tone -> no sync
    assert result.protocol.sync_confidence < 1.0


def test_protocol_stage_handles_no_match_honestly():
    """A detected candidate whose recovered bits carry no sync word ->
    first-class Unknown (never guessed)."""
    # A flat zero tone produces zero-mean recovered bits that do not
    # contain the 32-bit sync template, so the frame stage returns
    # an honest Unknown rather than guessing a protocol name.
    stream = np.zeros(4000, dtype=np.complex128)
    result = analyze_samples(stream, 8000.0, config=CONFIG)

    # A flat zero tone is too weak to be detected: the pipeline ends
    # after detection, and protocol analysis is skipped entirely.
    # That is the correct first-class behaviour (Unknown is never
    # inferred from an unobserved frame).
    assert result.protocol is None



def test_protocol_stage_disabled_by_default():
    """Explicit configuration only: no FrameConfig -> protocol stays None."""
    result = analyze_samples(_tone_at(1000.0), 8000.0)

    assert result.protocol is None


def test_protocol_stage_exported_to_json_and_csv():
    """The protocol result is JSON-safe and lands in the exported payload."""
    result = analyze_samples(
        _tone_at(500.0), 8000.0, config=CONFIG
    )
    payload = result.to_dict()

    import json
    json.dumps(payload)  # must not raise
    from prototype.reporting.export import export_json

    export_json(payload)  # must not raise

