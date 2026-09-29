"""Focused tests for the additional FEC schemes required by SIH-147.

Covers * Reed-Solomon (GF(256), shortened),
       * compact (3,6)-regular LDPC with bit-flipping,
       * serial concatenation (RS outer + convolutional inner),

through the existing registry (``encode_bits`` / ``decode_bits``) and an
end-to-end pipeline decode with an explicitly configured scheme.  No new
DSP is exercised and the pipeline's demodulation boundary is stubbed the
same way as the existing focused pipeline suites.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prototype.core.config import AnalysisConfig, FECConfig, FECMode
from prototype.core.exceptions import FECError
from prototype.detection.detector import SignalCandidate
from prototype.fec import (
    decode_bits,
    describe_scheme,
    encode_bits,
    list_schemes,
)
from prototype.fec import concatenated, ldpc, reed_solomon
from prototype.parameters.extractor import SignalParameters
from prototype.parameters.symbol_rate import SymbolRateEstimate
from prototype.pipeline import analyze_samples


def _bits(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistry:
    def test_new_schemes_registered(self):
        schemes = set(list_schemes())
        assert {"reedsolomon", "ldpc", "concatenated"} <= schemes

    @pytest.mark.parametrize("name", ["reedsolomon", "ldpc", "concatenated"])
    def test_describe_is_non_empty(self, name):
        assert isinstance(describe_scheme(name), str)
        assert describe_scheme(name)

    def test_unknown_scheme_raises(self):
        with pytest.raises(FECError):
            encode_bits(_bits(32, seed=1), "not_a_scheme")

    def test_fec_config_accepts_new_schemes(self):
        for name in ("reedsolomon", "ldpc", "concatenated"):
            cfg = FECConfig(mode=FECMode.MANUAL, scheme=name)
            assert cfg.scheme == name


# ===========================================================================
# Reed-Solomon
# ===========================================================================

class TestReedSolomon:
    def test_round_trip(self):
        payload = _bits(256, seed=1)
        encoded = reed_solomon.encode(payload)
        decoded, stats = reed_solomon.decode(encoded)
        assert np.array_equal(decoded[: payload.size], payload)
        assert stats["corrected_errors"] == 0

    def test_corrects_up_to_capacity(self):
        payload = _bits(256, seed=2)
        encoded = reed_solomon.encode(payload)
        rng = np.random.default_rng(9)
        n_bytes = encoded.size // 8
        for n_errors in (1, 2, 3, 4):
            noisy = encoded.copy()
            # Corrupt one bit in each of n_errors *distinct* symbols.
            byte_positions = rng.choice(n_bytes, size=n_errors, replace=False)
            for b in byte_positions:
                noisy[int(b) * 8] ^= 1
            decoded, stats = reed_solomon.decode(noisy)
            assert np.array_equal(decoded[: payload.size], payload)
            assert stats["uncorrectable_blocks"] == 0
            assert stats["corrected_errors"] == n_errors  # distinct bytes

    def test_more_symbol_errors_than_capacity_is_flagged(self):
        payload = _bits(256, seed=3)
        encoded = reed_solomon.encode(payload)
        noisy = encoded.copy()
        for i in range(6):  # 6 corrupted bytes > t=4
            noisy[i * 8] ^= 1
            noisy[i * 8 + 1] ^= 1
        _, stats = reed_solomon.decode(noisy)
        assert stats["uncorrectable_blocks"] >= 1

    def test_registry_round_trip(self):
        payload = _bits(512, seed=4)
        encoded, enc = encode_bits(payload, "reedsolomon")
        decoded, dec = decode_bits(encoded, "reedsolomon")
        assert np.array_equal(decoded[: payload.size], payload)
        assert enc.code_rate == pytest.approx(0.8)
        assert dec.corrected_errors == 0

    def test_rejects_partial_block(self):
        payload = _bits(256, seed=5)
        encoded = reed_solomon.encode(payload)
        with pytest.raises(ValueError):
            reed_solomon.decode(encoded[:-8])


# ===========================================================================
# LDPC
# ===========================================================================

class TestLDPC:
    def test_matrix_is_regular(self):
        assert ldpc.H.shape == (8, 16)
        assert set(ldpc.H.sum(axis=0).tolist()) == {3}
        assert set(ldpc.H.sum(axis=1).tolist()) == {6}

    def test_round_trip(self):
        payload = _bits(8 * 6, seed=1)
        encoded = ldpc.encode(payload)
        decoded, stats = ldpc.decode(encoded)
        assert np.array_equal(decoded, payload)
        assert stats["uncorrectable_blocks"] == 0

    def test_corrects_single_error_per_block(self):
        payload = _bits(8 * 6, seed=2)
        encoded = ldpc.encode(payload)
        rng = np.random.default_rng(7)
        noisy = encoded.copy()
        for b in range(6):
            noisy[b * ldpc.N + rng.integers(0, ldpc.N)] ^= 1
        decoded, stats = ldpc.decode(noisy)
        assert np.array_equal(decoded, payload)
        assert stats["corrected_errors"] == 6
        assert stats["uncorrectable_blocks"] == 0

    def test_heavy_noise_is_declared_uncorrectable(self):
        payload = _bits(8 * 4, seed=3)
        encoded = ldpc.encode(payload)
        noisy = encoded.copy()
        for b in range(4):
            # flip half the block -> beyond bit-flipping capacity
            noisy[b * ldpc.N : b * ldpc.N + 8] ^= 1
        _, stats = ldpc.decode(noisy)
        assert stats["uncorrectable_blocks"] == 4

    def test_registry_round_trip(self):
        payload = _bits(128, seed=4)
        encoded, enc = encode_bits(payload, "ldpc")
        decoded, dec = decode_bits(encoded, "ldpc")
        assert np.array_equal(decoded[: payload.size], payload)
        assert enc.code_rate == pytest.approx(0.5)

    def test_rejects_partial_block(self):
        with pytest.raises(ValueError):
            ldpc.decode(_bits(20, seed=5))


# ===========================================================================
# Concatenated (RS outer + convolutional inner)
# ===========================================================================

class TestConcatenated:
    def test_round_trip(self):
        payload = _bits(256, seed=1)
        encoded = concatenated.encode(payload)
        decoded, stats = concatenated.decode(encoded)
        assert np.array_equal(decoded[: payload.size], payload)
        assert stats["uncorrectable_blocks"] == 0
        assert "inner" in stats and "outer" in stats

    def test_registry_round_trip(self):
        payload = _bits(256, seed=2)
        encoded, enc = encode_bits(payload, "concatenated")
        decoded, dec = decode_bits(encoded, "concatenated")
        assert np.array_equal(decoded[: payload.size], payload)
        # RS rate 0.8 * conv 0.5
        assert enc.code_rate == pytest.approx(0.4)
        assert dec.corrected_errors == 0

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            concatenated.encode(np.array([], dtype=np.uint8))


# ===========================================================================
# Pipeline reachability (explicit scheme -> recovered payload)
# ===========================================================================

def _enable_pipeline(monkeypatch, received_bits: np.ndarray) -> None:
    params = SignalParameters(
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
        lambda *a, **k: [
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
        "prototype.parameters.extractor.extract_parameters", lambda *a, **k: params
    )
    monkeypatch.setattr(
        "prototype.parameters.symbol_rate.estimate_symbol_rate",
        lambda *a, **k: SymbolRateEstimate(
            symbol_rate=100.0, samples_per_symbol=80.0, confidence=1.0, method="test_mock"
        ),
    )

    def _dummy_demod(
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
            "num_symbols": len(received_bits) // 2,
            "num_bits": int(len(received_bits)),
            "decision_margin": 1.0,
            "constellation": {"symbol_rms": 1.414, "symbol_peak": 1.414},
        }
        return summary, np.asarray(received_bits, dtype=np.uint8).copy()

    monkeypatch.setattr("prototype.pipeline._demodulate", _dummy_demod)


@pytest.mark.parametrize(
    "scheme",
    ["hamming74", "repetition3", "conv12", "reedsolomon", "ldpc", "concatenated"],
)
def test_pipeline_manual_fec_recovers_payload(monkeypatch, scheme):
    payload = _bits(256, seed=42)
    encoded, _ = encode_bits(payload, scheme)
    _enable_pipeline(monkeypatch, encoded)

    config = replace(
        AnalysisConfig(),
        fec=FECConfig(
            mode=FECMode.MANUAL,
            scheme=scheme,
            interleaving_mode=FECMode.NONE,
        ),
    )
    result = analyze_samples(
        samples=np.zeros(encoded.size, dtype=np.complex64),
        sample_rate=1.0,
        config=config,
        reference_bits=None,
    )

    fec = result.demodulation.get("fec")
    assert fec is not None, f"pipeline did not decode {scheme}"
    assert fec["scheme"] == scheme
    assert fec["source"] == "explicit_config"
    # The recovered bitstream must contain the original payload as prefix.
    recovered = np.asarray(fec["decoded_bits"], dtype=np.uint8)
    assert np.array_equal(recovered[: payload.size], payload)
