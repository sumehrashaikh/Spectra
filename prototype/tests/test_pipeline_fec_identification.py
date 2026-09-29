"""Focused tests for the automatic FEC-identification stage in the pipeline.

The identifier is wired into ``pipeline.analyze_samples`` for AUTO mode.
These tests isolate that stage through the smallest dependency boundary
(the demodulation stub, exactly as ``test_pipeline_interleaving.py`` does)
so no RF waveform, DSP, classifier or synchroniser is involved.

Behaviour covered
-----------------
1. AUTO + genuinely encoded bits -> AUTO_DETECTED with the right scheme,
   and the decoded stream is exposed with ``source == "auto_identified"``.
2. AUTO + no structural evidence -> UNKNOWN/UNRESOLVED, no forced decode
   (never a guess).
3. NONE mode -> identification does not run at all.
4. The received (demodulated) bitstream is preserved and never mutated.
5. Provenance records the ``fec_identification`` step.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from prototype.core.config import (
    AnalysisConfig,
    FECConfig,
    FECMode,
)
from prototype.detection.detector import SignalCandidate
from prototype.fec import encode_bits
from prototype.parameters.extractor import SignalParameters
from prototype.parameters.symbol_rate import SymbolRateEstimate
from prototype.pipeline import analyze_samples


# ---------------------------------------------------------------------------
# Harness: stub everything upstream of the bits
# ---------------------------------------------------------------------------


def _demodulate_returning(bits: np.ndarray):
    """Demodulation stub returning an exact, known bitstream."""

    demod_bits = np.asarray(bits, dtype=np.uint8)

    def _dummy_demodulate(
        modulation,
        isolated,
        synchronized_signal,
        symbol_rate,
        config,
        provenance,
        warnings,
        capture_symbol_samples=False,
    ):
        summary = {
            "modulation": "BPSK",
            "num_symbols": int(demod_bits.size // 2),
            "num_bits": int(demod_bits.size),
            "decision_margin": 1.0,
            "constellation": {"symbol_rms": 1.414, "symbol_peak": 1.414},
        }
        return summary, demod_bits

    return _dummy_demodulate


def _enable_demodulation(monkeypatch, bits: np.ndarray) -> None:
    """Short-circuit the upstream chain so the pipeline sees ``bits``."""

    dummy_params = SignalParameters(
        center_frequency=0.0,
        peak_frequency=0.0,
        bandwidth=1.0,
        rms=1.0,
        peak=1.0,
        power=1.0,
        noise_power=1.0,
        snr_db=10.0,
        num_samples=1024,
        sample_rate=1.0,
        duration=1.0,
        bandwidth_3db=1.0,
        bandwidth_6db=1.0,
        bandwidth_99=1.0,
        papr_db=0.0,
        crest_factor=1.0,
        dynamic_range_db=0.0,
        mean_amplitude=1.0,
        dc_offset=0.0,
    )

    monkeypatch.setattr(
        "prototype.detection.detector.detect_candidates",
        lambda *args, **_kwargs: [
            SignalCandidate(
                center_frequency=0.5,
                bandwidth=1.0,
                peak_frequency=0.5,
                peak_power=1e6,
                confidence=1.0,
                start_frequency=0.0,
                end_frequency=1.0,
                start_index=0,
                end_index=128,
            )
        ],
    )
    monkeypatch.setattr(
        "prototype.parameters.extractor.extract_parameters",
        lambda *args, **_kwargs: dummy_params,
    )
    monkeypatch.setattr(
        "prototype.parameters.symbol_rate.estimate_symbol_rate",
        lambda *args, **_kwargs: SymbolRateEstimate(
            symbol_rate=100.0,
            samples_per_symbol=80.0,
            confidence=1.0,
            method="test_mock",
        ),
    )
    monkeypatch.setattr(
        "prototype.pipeline._demodulate",
        _demodulate_returning(bits),
    )


def _config(*, fec_mode: str = FECMode.AUTO, scheme: str | None = None):
    """AnalysisConfig with a fixed FEC block; interleaving disabled so the
    bits reaching the FEC stage are exactly the demodulated bits."""
    return replace(
        AnalysisConfig(),
        fec=FECConfig(
            mode=fec_mode,
            scheme=scheme,
            crc=None,
            interleave_depth=1,
            interleaving_mode=FECMode.NONE,
        ),
    )


def _run(monkeypatch, bits: np.ndarray, **kwargs):
    _enable_demodulation(monkeypatch, bits)
    return analyze_samples(
        samples=np.zeros(bits.size, dtype=np.complex64),
        sample_rate=1.0,
        config=kwargs.pop("config", _config()),
        reference_bits=kwargs.pop("reference_bits", None),
    )


# ---------------------------------------------------------------------------
# 1. AUTO + genuinely encoded bits -> AUTO_DETECTED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme",
    ["hamming74", "repetition3", "conv12"],
)
def test_auto_identifies_encoded_stream(monkeypatch, scheme):
    rng = np.random.default_rng(11)
    payload = rng.integers(0, 2, 256).astype(np.uint8)
    encoded, _ = encode_bits(payload, scheme)
    encoded = np.asarray(encoded, dtype=np.uint8)

    result = _run(monkeypatch, encoded, config=_config(fec_mode=FECMode.AUTO))

    identification = result.demodulation.get("fec_identification")
    assert identification is not None, "identification stage did not run"
    assert identification["status"] == "AUTO_DETECTED"
    assert identification["best_scheme"] == scheme
    assert identification["confidence"] > 0.0

    # The decoded stream is exposed and marked as auto-identified.
    fec = result.demodulation.get("fec")
    assert fec is not None, "auto-identified scheme was not decoded"
    assert fec["scheme"] == scheme
    assert fec["source"] == "auto_identified"


def test_auto_detected_decode_recovers_payload(monkeypatch):
    """A cleanly encoded stream is auto-decoded back to its payload."""
    rng = np.random.default_rng(3)
    payload = rng.integers(0, 2, 256).astype(np.uint8)
    encoded, _ = encode_bits(payload, "hamming74")
    encoded = np.asarray(encoded, dtype=np.uint8)

    result = _run(monkeypatch, encoded, config=_config(fec_mode=FECMode.AUTO))

    decoded = np.asarray(result.demodulation["fec"]["decoded_bits"], dtype=np.uint8)
    assert decoded.size == payload.size
    assert np.array_equal(decoded, payload)


# ---------------------------------------------------------------------------
# 2. AUTO + no evidence -> honest UNKNOWN, never a guess
# ---------------------------------------------------------------------------


def test_auto_without_evidence_reports_unknown_and_does_not_decode(monkeypatch):
    rng = np.random.default_rng(5)
    noise_bits = rng.integers(0, 2, 512).astype(np.uint8)

    result = _run(monkeypatch, noise_bits, config=_config(fec_mode=FECMode.AUTO))

    identification = result.demodulation.get("fec_identification")
    assert identification is not None
    assert identification["status"] in ("UNKNOWN", "UNRESOLVED", "FAILED")
    assert identification["best_scheme"] is None

    # No scheme may be forced when there is no evidence.
    assert result.demodulation.get("fec") is None


# ---------------------------------------------------------------------------
# 3. NONE mode -> the stage does not run
# ---------------------------------------------------------------------------


def test_none_mode_skips_identification(monkeypatch):
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, 256).astype(np.uint8)

    result = _run(monkeypatch, bits, config=_config(fec_mode=FECMode.NONE))

    assert result.demodulation.get("fec_identification") is None


def test_manual_mode_skips_identification_but_decodes_explicitly(monkeypatch):
    rng = np.random.default_rng(13)
    payload = rng.integers(0, 2, 256).astype(np.uint8)
    encoded, _ = encode_bits(payload, "repetition3")
    encoded = np.asarray(encoded, dtype=np.uint8)

    result = _run(
        monkeypatch,
        encoded,
        config=_config(fec_mode=FECMode.MANUAL, scheme="repetition3"),
    )

    assert result.demodulation.get("fec_identification") is None
    fec = result.demodulation.get("fec")
    assert fec is not None
    assert fec["source"] == "explicit_config"
    assert fec["scheme"] == "repetition3"


# ---------------------------------------------------------------------------
# 4. The received bits are preserved and never mutated
# ---------------------------------------------------------------------------


def test_received_bits_are_preserved(monkeypatch):
    rng = np.random.default_rng(17)
    payload = rng.integers(0, 2, 256).astype(np.uint8)
    encoded, _ = encode_bits(payload, "hamming74")
    encoded = np.asarray(encoded, dtype=np.uint8)
    original = encoded.copy()

    result = _run(monkeypatch, encoded, config=_config(fec_mode=FECMode.AUTO))

    received = np.asarray(result.demodulation["received_bits"], dtype=np.uint8)
    assert np.array_equal(received, original), "received bits were altered"
    assert np.array_equal(encoded, original), "demodulated bits were mutated"


# ---------------------------------------------------------------------------
# 5. Provenance records the stage
# ---------------------------------------------------------------------------


def test_provenance_records_fec_identification_step(monkeypatch):
    rng = np.random.default_rng(21)
    payload = rng.integers(0, 2, 256).astype(np.uint8)
    encoded, _ = encode_bits(payload, "conv12")
    encoded = np.asarray(encoded, dtype=np.uint8)

    result = _run(monkeypatch, encoded, config=_config(fec_mode=FECMode.AUTO))

    steps = {step["name"]: step for step in (result.provenance or {}).get("steps", [])}
    assert "fec_identification" in steps
    assert steps["fec_identification"]["status"] == "ok"
    assert steps["fec_identification"]["detail"]["best_scheme"] == "conv12"


# ---------------------------------------------------------------------------
# 6. Serialization: the identification result survives to_dict()
# ---------------------------------------------------------------------------


def test_identification_result_is_json_safe(monkeypatch):
    rng = np.random.default_rng(23)
    payload = rng.integers(0, 2, 256).astype(np.uint8)
    encoded, _ = encode_bits(payload, "hamming74")
    encoded = np.asarray(encoded, dtype=np.uint8)

    result = _run(monkeypatch, encoded, config=_config(fec_mode=FECMode.AUTO))

    import json

    payload_dict = result.to_dict()
    dumped = json.dumps(payload_dict)  # must not raise
    assert '"fec_identification"' in dumped
