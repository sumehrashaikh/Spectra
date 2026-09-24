"""End-to-end pipeline tests (pipeline.py) over synthetic signals."""

import numpy as np
import pytest

from prototype.core.exceptions import PipelineError
from prototype.pipeline import analyze_samples
from prototype.parameters.symbol_rate import rrc_filter


def _qpsk_signal(
    num_symbols=512, fs=8000.0, sps=8, carrier=500.0, noise=0.03, seed=42
):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (num_symbols, 2))
    const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    gray = np.array([0, 1, 3, 2])
    symbols = const[gray[bits[:, 0] * 2 + bits[:, 1]]]
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_filter(sps), mode="same")
    t = np.arange(shaped.size) / fs
    x = shaped * np.exp(1j * 2 * np.pi * carrier * t)
    if noise > 0:
        x = x + noise * (
            rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
        ) / np.sqrt(2)
    return x, bits.reshape(-1)


def _bpsk_signal(
    num_symbols=512, fs=8000.0, sps=8, carrier=500.0, noise=0.03, seed=42
):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_symbols)
    symbols = (2 * bits - 1).astype(complex)
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_filter(sps), mode="same")
    t = np.arange(shaped.size) / fs
    x = shaped * np.exp(1j * 2 * np.pi * carrier * t)
    if noise > 0:
        x = x + noise * (
            rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
        ) / np.sqrt(2)
    return x, bits


def _bfsk_signal(num_symbols=400, fs=8000.0, sps=80, f0=500.0, f1=700.0, seed=42):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_symbols)
    parts = []
    for bit in bits:
        t = np.arange(sps) / fs
        parts.append(np.exp(1j * 2 * np.pi * (f1 if bit else f0) * t))
    return np.concatenate(parts), bits


class TestFullPipeline:
    def test_qpsk_end_to_end(self):
        x, reference = _qpsk_signal()
        result = analyze_samples(x, 8000.0, reference_bits=reference)
        data = result.to_dict()

        assert data["detections"], "expected at least one detection"
        assert data["classification"]["modulation"] == "QPSK"
        assert data["symbol_rate"]["symbol_rate_hz"] == pytest.approx(
            1000.0, abs=10.0
        )
        assert data["demodulation"]["modulation"] == "QPSK"
        assert data["demodulation"]["num_bits"] == 1024
        assert data["ber"]["ber"] < 0.01
        assert data["provenance"]["steps"]

    def test_bpsk_end_to_end(self):
        x, reference = _bpsk_signal(noise=0.02)
        result = analyze_samples(x, 8000.0, reference_bits=reference)
        data = result.to_dict()
        assert data["classification"]["modulation"] == "BPSK"
        assert data["ber"]["ber"] < 0.01
        assert "polarity" in data["ber"]["ambiguity_resolution"]

    def test_bfsk_end_to_end(self):
        x, reference = _bfsk_signal()
        result = analyze_samples(x, 8000.0, reference_bits=reference)
        data = result.to_dict()
        assert data["classification"]["modulation"] == "BFSK"
        assert data["demodulation"]["num_bits"] >= 300
        # BER for coherent BFSK with known tones at high SNR should be low
        assert data["ber"]["ber"] < 0.05

    def test_without_reference_no_ber(self):
        x, _ = _qpsk_signal()
        result = analyze_samples(x, 8000.0)
        assert result.ber is None

    def test_provenance_complete(self):
        x, _ = _qpsk_signal()
        result = analyze_samples(x, 8000.0)
        names = {step["name"] for step in result.provenance["steps"]}
        assert {"preprocessing", "detection", "isolation"} <= names
        assert result.provenance["started_utc"]
        assert result.provenance["finished_utc"]
        assert result.provenance["software_version"]

    def test_processing_modes(self):
        x, _ = _qpsk_signal()
        for mode in ("quick", "deep"):
            result = analyze_samples(x, 8000.0, mode=mode)
            assert result.classification is not None

    def test_no_candidates_detected(self):
        # pure noise: detection should find nothing and return gracefully
        rng = np.random.default_rng(0)
        noise = 0.5 * (
            rng.standard_normal(8192) + 1j * rng.standard_normal(8192)
        )
        result = analyze_samples(noise, 8000.0)
        assert result.detections == []
        assert any("No signal candidates" in w for w in result.warnings)

    def test_invalid_samples_raise(self):
        with pytest.raises(PipelineError):
            analyze_samples(np.array([np.nan + 0j]), 8000.0)

    def test_second_candidate_analysis(self):
        # two well-separated tones; candidate_index=1 selects the weaker
        t = np.arange(16000) / 8000.0
        strong = np.exp(1j * 2 * np.pi * 500.0 * t)
        weak = 0.3 * np.exp(1j * 2 * np.pi * 2500.0 * t)
        x = strong + weak
        result = analyze_samples(x, 8000.0, candidate_index=None) if False else None

        from prototype.core.config import processing_mode_config
        from dataclasses import replace

        config = replace(
            processing_mode_config("balanced"), candidate_index=1
        )
        result2 = analyze_samples(x, 8000.0, config=config)
        assert result2.selected_candidate is not None
        assert abs(
            result2.selected_candidate["center_frequency"] - 2500.0
        ) < 200.0


class TestGracefulDegradation:
    def test_pure_tone_classified_without_crash(self):
        t = np.arange(8000) / 8000.0
        tone = np.exp(1j * 2 * np.pi * 500.0 * t)
        result = analyze_samples(tone, 8000.0)
        # pipeline must complete regardless of classification outcome
        assert result.provenance["steps"]
        # no stage marked failed
        statuses = {step["status"] for step in result.provenance["steps"]}
        assert "failed" not in statuses

    def test_json_serializable(self):
        x, reference = _qpsk_signal()
        result = analyze_samples(x, 8000.0, reference_bits=reference)
        import json

        payload = json.dumps(result.to_dict())  # must not raise
        assert "QPSK" in payload
