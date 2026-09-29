"""Focused tests for the extended de-interleaving families (SIH-147).

Covers the four families already present in ``prototype.fec.interleaving``:

* block (row-column)
* convolutional (rate-1/k)
* diagonal (square, fixed width)
* pseudo-random (seeded permutation)

and their manual wiring through ``FECConfig`` / the pipeline:

* pure round-trip determinism for every family and parameter,
* the pseudo-random regression (seed must be honoured *and* the
  deinterleaver must be a true inverse, not a second application of the
  permutation),
* ``FECConfig.manual_deinterleave_plan`` reasoning for inapplicable
  parameter/length combinations,
* an end-to-end MANUAL pipeline round trip (interleaved bits in ->
  original payload out) for each family, with no RF waveform or DSP
  (the demodulation boundary is stubbed, exactly like the existing
  ``test_pipeline_interleaving`` suite).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prototype.core.config import AnalysisConfig, FECConfig, FECMode
from prototype.detection.detector import SignalCandidate
from prototype.fec.interleaving import (
    convolutional_deinterleave,
    convolutional_interleave,
    deinterleave_bits,
    diagonal_deinterleave,
    diagonal_interleave,
    interleave_bits,
    pseudo_random_deinterleave,
    pseudo_random_interleave,
)
from prototype.parameters.extractor import SignalParameters
from prototype.parameters.symbol_rate import SymbolRateEstimate
from prototype.pipeline import analyze_samples


def _bits(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


# ===========================================================================
# Pure interleaver round trips
# ===========================================================================

class TestBlockRoundTrip:
    @pytest.mark.parametrize("n", [64, 128, 250])
    @pytest.mark.parametrize("depth", [2, 4, 8])
    def test_block_round_trip(self, n, depth):
        payload = _bits(n, seed=n + depth)
        interleaved = interleave_bits(payload, depth=depth)
        recovered = deinterleave_bits(
            interleaved, depth=depth, original_size=payload.size
        )
        assert np.array_equal(recovered, payload)


class TestConvolutionalRoundTrip:
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_convolutional_round_trip(self, k):
        payload = _bits(120, seed=k)
        interleaved = convolutional_interleave(payload, k=k)
        recovered = convolutional_deinterleave(interleaved, k=k)
        assert np.array_equal(recovered, payload)
        # a genuine permutation: same multiset, different order
        assert np.array_equal(np.sort(interleaved), np.sort(payload))

    def test_convolutional_requires_divisible_length(self):
        with pytest.raises(ValueError):
            convolutional_interleave(_bits(10, seed=1), k=4)

    def test_convolutional_requires_k_at_least_two(self):
        with pytest.raises(ValueError):
            convolutional_interleave(_bits(8, seed=1), k=1)


class TestDiagonalRoundTrip:
    @pytest.mark.parametrize("depth", [2, 4, 8])
    def test_diagonal_round_trip(self, depth):
        payload = _bits(depth * depth, seed=depth)
        interleaved = diagonal_interleave(payload, depth=depth)
        recovered = diagonal_deinterleave(interleaved, depth=depth)
        assert np.array_equal(recovered, payload)

    def test_diagonal_requires_square_length(self):
        with pytest.raises(ValueError):
            diagonal_interleave(_bits(10, seed=1), depth=4)


class TestPseudoRandomRoundTrip:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
    def test_pseudo_random_is_true_inverse(self, seed):
        """Regression: deinterleave must invert the permutation, not re-apply it."""
        payload = _bits(200, seed=seed)
        interleaved = pseudo_random_interleave(payload, seed=seed)
        recovered = pseudo_random_deinterleave(interleaved, seed=seed)
        assert np.array_equal(recovered, payload)

    def test_pseudo_random_is_seed_sensitive(self):
        """Regression: different seeds must produce different permutations."""
        payload = _bits(256, seed=3)
        a = pseudo_random_interleave(payload, seed=0)
        b = pseudo_random_interleave(payload, seed=7)
        assert not np.array_equal(a, b)

    def test_pseudo_random_is_a_permutation(self):
        payload = _bits(150, seed=11)
        interleaved = pseudo_random_interleave(payload, seed=5)
        assert np.array_equal(np.sort(interleaved), np.sort(payload))

    def test_pseudo_random_seed_mismatch_does_not_recover(self):
        payload = _bits(64, seed=9)
        interleaved = pseudo_random_interleave(payload, seed=2)
        wrong = pseudo_random_deinterleave(interleaved, seed=8)
        assert not np.array_equal(wrong, payload)

    def test_pseudo_random_rejects_empty(self):
        with pytest.raises(ValueError):
            pseudo_random_interleave(np.array([], dtype=np.uint8), seed=1)


# ===========================================================================
# FECConfig manual dispatch
# ===========================================================================

def _fec(family: str, depth: int) -> FECConfig:
    return FECConfig(
        mode=FECMode.AUTO,
        interleaving_mode=FECMode.MANUAL,
        interleave_family=family,
        interleave_depth=depth,
    )


class TestConfigDispatch:
    def test_unknown_family_rejected(self):
        with pytest.raises(Exception):
            FECConfig(interleave_family="not_a_family")

    def test_block_dispatch(self):
        payload = _bits(64, seed=1)
        cfg = _fec("block", 4)
        assert np.array_equal(cfg.deinterleave_bits(interleave_bits(payload, 4)), payload)

    def test_convolutional_dispatch(self):
        payload = _bits(64, seed=1)
        cfg = _fec("convolutional", 4)
        assert np.array_equal(
            cfg.deinterleave_bits(convolutional_interleave(payload, 4)), payload
        )

    def test_diagonal_dispatch(self):
        payload = _bits(64, seed=1)
        cfg = _fec("diagonal", 8)
        assert np.array_equal(
            cfg.deinterleave_bits(diagonal_interleave(payload, 8)), payload
        )

    def test_pseudo_random_dispatch(self):
        payload = _bits(64, seed=1)
        cfg = _fec("pseudo_random", 3)  # depth field doubles as the seed
        assert np.array_equal(
            cfg.deinterleave_bits(pseudo_random_interleave(payload, 3)), payload
        )

    def test_non_manual_is_passthrough(self):
        payload = _bits(64, seed=1)
        cfg = FECConfig(interleaving_mode=FECMode.AUTO, interleave_family="diagonal")
        assert np.array_equal(cfg.deinterleave_bits(payload), payload)

    def test_plan_reports_inapplicable_reasons(self):
        assert _fec("diagonal", 4).manual_deinterleave_plan(10)[0] is False
        assert _fec("convolutional", 4).manual_deinterleave_plan(10)[0] is False
        assert _fec("block", 1).manual_deinterleave_plan(10)[0] is False
        assert _fec("pseudo_random", 5).manual_deinterleave_plan(0)[0] is False
        assert _fec("block", 4).manual_deinterleave_plan(10)[0] is True


# ===========================================================================
# Pipeline MANUAL round trip for each family
# ===========================================================================

def _make_config(family: str, depth: int) -> AnalysisConfig:
    return replace(
        AnalysisConfig(),
        fec=FECConfig(
            mode=FECMode.AUTO,
            scheme=None,
            crc=None,
            interleave_family=family,
            interleave_depth=depth,
            interleaving_mode=FECMode.MANUAL,
        ),
    )


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
        "prototype.parameters.extractor.extract_parameters",
        lambda *a, **k: params,
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


class TestPipelineManualFamilies:
    def _run(self, monkeypatch, family, depth, payload, interleaved):
        _enable_pipeline(monkeypatch, interleaved)
        result = analyze_samples(
            samples=np.zeros(interleaved.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(family, depth),
            reference_bits=None,
        )
        summary = result.demodulation.get("interleaving_result", {})
        recovered = result.demodulation.get("deinterleaved_bits")
        return result, summary, recovered

    def test_pipeline_manual_block_recovers_payload(self, monkeypatch):
        payload = _bits(128, seed=1)
        _, summary, recovered = self._run(
            monkeypatch, "block", 4, payload, interleave_bits(payload, 4)
        )
        assert summary.get("family") == "block"
        assert summary.get("applicable") is True
        assert np.array_equal(recovered, payload)

    def test_pipeline_manual_convolutional_recovers_payload(self, monkeypatch):
        payload = _bits(128, seed=2)
        _, summary, recovered = self._run(
            monkeypatch, "convolutional", 4, payload, convolutional_interleave(payload, 4)
        )
        assert summary.get("family") == "convolutional"
        assert np.array_equal(recovered, payload)

    def test_pipeline_manual_diagonal_recovers_payload(self, monkeypatch):
        payload = _bits(64, seed=3)
        _, summary, recovered = self._run(
            monkeypatch, "diagonal", 8, payload, diagonal_interleave(payload, 8)
        )
        assert summary.get("family") == "diagonal"
        assert np.array_equal(recovered, payload)

    def test_pipeline_manual_pseudo_random_recovers_payload(self, monkeypatch):
        payload = _bits(128, seed=4)
        _, summary, recovered = self._run(
            monkeypatch, "pseudo_random", 7, payload, pseudo_random_interleave(payload, 7)
        )
        assert summary.get("family") == "pseudo_random"
        assert np.array_equal(recovered, payload)

    def test_pipeline_manual_inapplicable_warns_and_leaves_bits(self, monkeypatch):
        payload = _bits(100, seed=5)  # not depth*depth for diagonal depth=8
        result, summary, recovered = self._run(
            monkeypatch, "diagonal", 8, payload, payload.copy()
        )
        assert summary.get("applicable") is False
        assert recovered is None
        assert any("deinterleaving not applied" in w for w in result.warnings)
        # received bits remain intact
        assert np.array_equal(result.demodulation.get("received_bits"), payload)
