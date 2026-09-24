"""Tests for core/config, core/provenance, symbol-rate estimators, reporting."""

import json

import numpy as np
import pytest

from prototype.core.config import (
    AnalysisConfig,
    DetectionConfig,
    config_to_dict,
    processing_mode_config,
)
from prototype.core.exceptions import ConfigurationError
from prototype.core.provenance import (
    hash_samples,
    new_provenance,
    provenance_to_dict,
    sha256_file,
    skip_stage,
)
from prototype.parameters.symbol_rate import (
    estimate_symbol_rate,
    estimate_symbol_rate_fsk,
    generate_rrc_bpsk,
    rrc_filter,
)
from prototype.reporting.export import export_csv, export_html, export_json


class TestConfig:
    def test_defaults_valid(self):
        config = AnalysisConfig()
        assert config.detection.threshold_db == 10.0

    def test_invalid_detection_rejected(self):
        with pytest.raises(ConfigurationError):
            DetectionConfig(threshold_db=-1.0)

    def test_modes(self):
        for mode in ("quick", "balanced", "deep", "realtime"):
            assert isinstance(processing_mode_config(mode), AnalysisConfig)

    def test_unknown_mode(self):
        with pytest.raises(ConfigurationError):
            processing_mode_config("ultra")

    def test_serializable(self):
        data = config_to_dict(processing_mode_config("deep"))
        json.dumps(data)

    def test_frozen(self):
        config = DetectionConfig()
        with pytest.raises(Exception):
            config.threshold_db = 99.0


class TestProvenance:
    def test_step_recording(self):
        provenance = new_provenance()
        with provenance.record_step("unit_test") as step:
            step["detail"] = 42
        record = provenance_to_dict(provenance)
        assert record["steps"][0]["name"] == "unit_test"
        assert record["steps"][0]["status"] == "ok"
        assert record["steps"][0]["detail"]["detail"] == 42

    def test_skip_status(self):
        provenance = new_provenance()
        with provenance.record_step("skipped_stage"):
            skip_stage("not applicable")
        record = provenance_to_dict(provenance)
        assert record["steps"][0]["status"] == "skipped"

    def test_failure_status_propagates(self):
        provenance = new_provenance()
        with pytest.raises(RuntimeError):
            with provenance.record_step("failing_stage"):
                raise RuntimeError("boom")
        record = provenance_to_dict(provenance)
        assert record["steps"][0]["status"] == "failed"

    def test_file_hash(self, tmp_path):
        target = tmp_path / "data.bin"
        target.write_bytes(b"0123456789abcdef" * 32)
        digest = sha256_file(target)
        assert len(digest) == 64
        assert digest == sha256_file(target)
        other = tmp_path / "data2.bin"
        other.write_bytes(b"different")
        assert digest != sha256_file(other)

    def test_hash_samples(self):
        x = np.array([1 + 1j, 2 + 2j])
        assert hash_samples(x, 8000) == hash_samples(x, 8000)
        assert hash_samples(x, 8000) != hash_samples(x, 4000)


class TestSymbolRate:
    def test_rrc_shaped_bpsk(self):
        shaped, bits = generate_rrc_bpsk(8000, 100, 2.0)
        estimate = estimate_symbol_rate(shaped, 8000, 50, 500)
        assert estimate.symbol_rate == pytest.approx(100.0, abs=1.0)
        assert estimate.confidence > 50.0

    def test_rectangular_bpsk_constant_envelope(self):
        rng = np.random.default_rng(42)
        sps = 80
        bits = rng.integers(0, 2, 200)
        symbols = 2 * bits - 1
        baseband = np.repeat(symbols, sps).astype(np.complex128)
        t = np.arange(baseband.size) / 8000
        x = baseband * np.exp(1j * 2 * np.pi * 500 * t)
        estimate = estimate_symbol_rate(x, 8000, 20, 500)
        assert estimate.symbol_rate == pytest.approx(100.0, abs=1.0)

    def test_qpsk(self):
        rng = np.random.default_rng(1)
        sps = 80
        bits = rng.integers(0, 2, (500, 2))
        const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        gray = np.array([0, 1, 3, 2])
        symbols = const[gray[bits[:, 0] * 2 + bits[:, 1]]]
        upsampled = np.zeros(500 * sps, dtype=complex)
        upsampled[::sps] = symbols
        shaped = np.convolve(upsampled, rrc_filter(sps), mode="same")
        t = np.arange(shaped.size) / 8000
        x = shaped * np.exp(1j * 2 * np.pi * 500 * t)
        estimate = estimate_symbol_rate(x, 8000, 50, 500)
        assert estimate.symbol_rate == pytest.approx(100.0, abs=1.0)

    def test_fsk_estimator(self):
        rng = np.random.default_rng(42)
        sps = 80
        bits = rng.integers(0, 2, 200)
        parts = []
        for bit in bits:
            t = np.arange(sps) / 8000
            parts.append(np.exp(1j * 2 * np.pi * (700.0 if bit else 500.0) * t))
        samples = np.concatenate(parts)
        estimate = estimate_symbol_rate_fsk(samples, 8000, 10, 1000)
        assert estimate.symbol_rate == pytest.approx(100.0, abs=5.0)
        assert estimate.method == "fsk_run_length"

    def test_too_short_rejected(self):
        with pytest.raises(ValueError):
            estimate_symbol_rate(np.zeros(10, dtype=complex), 8000, 20, 100)


class TestReporting:
    def _sample_result(self):
        return {
            "input": {"file": "x.wav", "sample_rate": 8000.0},
            "detections": [
                {"center_frequency": 500.0, "bandwidth": 200.0},
            ],
            "classification": {"modulation": "QPSK", "confidence": 99.0},
            "ber": None,
            "warnings": ["one warning"],
            "provenance": {"software_version": "2.1.0", "steps": []},
            "complex_value": 1 + 2j,
            "numpy_value": np.float64(3.5),
        }

    def test_json_export(self, tmp_path):
        payload = export_json(self._sample_result())
        parsed = json.loads(payload)
        assert parsed["complex_value"] == {"real": 1.0, "imag": 2.0}
        assert parsed["numpy_value"] == 3.5
        out = tmp_path / "r.json"
        export_json(self._sample_result(), out)
        assert out.is_file()

    def test_csv_export(self, tmp_path):
        text = export_csv(self._sample_result())
        assert "parameter,value" in text
        assert "classification.modulation,QPSK" in text
        out = tmp_path / "r.csv"
        export_csv(self._sample_result(), out)
        assert out.is_file()

    def test_html_export(self, tmp_path):
        html = export_html(self._sample_result(), title="Test Report")
        assert "Test Report" in html
        assert "QPSK" in html
        assert "one warning" in html
        assert "Not certified" in html  # honest disclaimer present
        out = tmp_path / "r.html"
        export_html(self._sample_result(), out)
        assert out.is_file()

    def test_html_handles_missing_sections(self):
        html = export_html({"input": {"file": "only.wav"}})
        assert "only.wav" in html
