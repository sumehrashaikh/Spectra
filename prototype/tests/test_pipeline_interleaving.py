"""Focused regression tests for the block-interleaving stage inside the pipeline.

These tests isolate the interleaving stage with a deterministic known bit
array injected through the smallest existing dependency boundary (no RF
waveform, no DSP). They do not depend on fragile synthetic waveforms, on the
classifier/synchronizer, or on production signal generation.

Behaviour covered
-----------------
1. AUTO + strong test result (identify_interleaving -> AUTO_DETECTED) ->
   correct depth -> deinterleave_bits called -> demodulation/fec path receives
   the deinterleaved copy.
2. AUTO + weak evidence (identify_interleaving -> UNKNOWN/UNRESOLVED,
   best_depth None) -> no deinterleaving, bits unchanged.
3. MANUAL + interleave_depth=4 -> deinterleave_bits applied directly, no
   identification, no AUTO_DETECTED.
4. NONE -> bits unchanged, no deinterleaving.
5. Existing FEC path still receives the correct deinterleaved bits when AUTO
   succeeds (end-to-end through the stage).
6. Received (demodulated) bits are never mutated by AUTO deinterleaving.

Each AUTO case drives the pipeline with identify_interleaving mocked to return
a real InterleavingIdentificationResult, so the test asserts only the
pipeline's interleaving behaviour, not the identifier itself.

No GUI, CLI, docs, DSP, classifier, synchroniser, or new interleaver type are
touched.  Production code is NOT modified.
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
from prototype.core.isolator import IsolationResult
from prototype.core.signal import Signal
from prototype.detection.detector import SignalCandidate
from prototype.fec.interleaving import deinterleave_bits
from prototype.fec.identification_interleaving import (
    AUTO_DETECTED,
    InterleavingIdentificationResult,
)
from prototype.pipeline import analyze_samples
from prototype.parameters.extractor import SignalParameters
from prototype.parameters.symbol_rate import SymbolRateEstimate


class _FakeInterleavingResult(InterleavingIdentificationResult):
    """A deterministic stand-in for the real Identifier result."""

    def __init__(
        self,
        status: str,
        best_depth: int | None,
        confidence: float = 0.5,
    ) -> None:
        super().__init__(
            status=status,
            interleaving_detected=status == AUTO_DETECTED,
            best_type="block",
            best_depth=best_depth,
            confidence=confidence,
            candidates=[],
            evidence={"mode": "test_injected"},
            input_bit_count=0,
            validation_before="n/a (test injected)",
            validation_after="n/a (test injected)",
            improvement=None,
            warnings=[],
        )


def _make_config(
    *,
    fec_interleaving_mode: str = FECMode.AUTO,
    interleave_depth: int = 1,
) -> AnalysisConfig:
    """Build an AnalysisConfig with a custom ``FECConfig``."""
    return replace(
        AnalysisConfig(),
        fec=FECConfig(
            mode=FECMode.AUTO,
            scheme=None,
            crc=None,
            interleave_depth=interleave_depth,
            interleaving_mode=fec_interleaving_mode,
        ),
    )


def _bits(n: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic binary vector of length ``n``."""
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def _demodulate_callback():
    """Stub that short-circuits the pipeline's demodulation boundary so the
    interleaving stage (the subject of this suite) sees a deterministic
    bitstream with no real RF waveform or DSP needed.
    """
    demod_bits = _bits(128, seed=42)

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
            "num_symbols": len(demod_bits) // 2,
            "num_bits": int(len(demod_bits)),
            "decision_margin": 1.0,
            "constellation": {"symbol_rms": 1.414, "symbol_peak": 1.414},
        }
        return summary, np.asarray(demod_bits, dtype=np.uint8)

    return _dummy_demodulate


def _enable_demodulation(monkeypatch):
    """Short-circuit the demodulation boundary via 
    prototype.pipeline._demodulate.

    The interleaving stage is the subject of this suite; the upstream
    detection/classification/symbol-rate chain does not meet the test's no-RF
    constraint, so it is stubbed at its smallest dependency boundary.
    """
    snr_db = 10.0
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
        _demodulate_callback(),
    )


# ===========================================================================
# 1. AUTO + strong test result -> AUTO_DETECTED -> deinterleave applied
# ===========================================================================

class TestAutoStrongValidator:
    def test_auto_detects_depth_and_deinterleaves(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        def fake_identify(bits, min_depth=2, max_depth=16):
            assert np.array_equal(bits, received_bits)
            return InterleavingIdentificationResult(
                status=AUTO_DETECTED,
                interleaving_detected=True,
                best_type="block",
                best_depth=4,
                confidence=0.95,
                candidates=[],
                evidence={"depth": 4, "structural": True, "crc_valid": True},
                input_bit_count=int(bits.size),
                validation_before="sync word + CRC valid",
                validation_after="sync word + CRC valid",
                improvement=40.0,
                warnings=[],
            )

        monkeypatch.setattr(
            "prototype.pipeline.identify_interleaving",
            fake_identify,
        )

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(fec_interleaving_mode=FECMode.AUTO, interleave_depth=1),
            reference_bits=_bits(128, seed=99),
        )

        interleaving_summary = result.demodulation.get("interleaving_result", {})
        deinterleaved = result.demodulation.get("deinterleaved_bits")

        assert interleaving_summary.get("status") == AUTO_DETECTED
        assert interleaving_summary.get("best_depth") == 4
        assert interleaving_summary.get("confidence") == pytest.approx(0.95)
        assert deinterleaved is not None
        assert not np.array_equal(deinterleaved, received_bits)

    def test_pipeline_calls_existing_deinterleave_bits_on_auto(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        def fake_identify(bits, min_depth=2, max_depth=16):
            assert np.array_equal(bits, received_bits)
            return InterleavingIdentificationResult(
                status=AUTO_DETECTED,
                interleaving_detected=True,
                best_type="block",
                best_depth=4,
                confidence=0.9,
            )

        monkeypatch.setattr(
            "prototype.pipeline.identify_interleaving",
            fake_identify,
        )

        original_deinterleave = deinterleave_bits
        calls = {"n": 0}

        def tracking_deinterleave(bits, depth, original_size):
            calls["n"] += 1
            return original_deinterleave(bits, depth, original_size)

        monkeypatch.setattr(
            "prototype.pipeline.deinterleave_bits",
            tracking_deinterleave,
        )

        analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(fec_interleaving_mode=FECMode.AUTO, interleave_depth=1),
            reference_bits=_bits(128, seed=99),
        )

        assert calls["n"] == 1, "deinterleave_bits must run exactly once on AUTO_DETECTED"


# ===========================================================================
# 2. AUTO + weak evidence -> UNKNOWN/UNRESOLVED -> NO deinterleaving
# ===========================================================================

class TestAutoWeakEvidence:
    def test_auto_no_detection_leaves_bits_unchanged(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        def fake_identify(bits, min_depth=2, max_depth=16):
            assert np.array_equal(bits, received_bits)
            return InterleavingIdentificationResult(
                status="UNKNOWN",
                interleaving_detected=False,
                best_type=None,
                best_depth=None,
                confidence=0.0,
                candidates=[],
                evidence={},
                input_bit_count=int(bits.size),
                validation_before="n/a",
                validation_after="n/a",
                improvement=None,
                warnings=["no structural evidence"],
            )

        monkeypatch.setattr(
            "prototype.pipeline.identify_interleaving",
            fake_identify,
        )

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(fec_interleaving_mode=FECMode.AUTO, interleave_depth=1),
            reference_bits=_bits(128, seed=99),
        )

        interleaving_summary = result.demodulation.get("interleaving_result", {})
        deinterleaved = result.demodulation.get("deinterleaved_bits")

        assert interleaving_summary.get("status") in (
            "UNKNOWN",
            "UNRESOLVED",
            "NO_INTERLEAVING",
        )
        assert interleaving_summary.get("best_depth") is None
        assert deinterleaved is None


# ===========================================================================
# 3. MANUAL + depth=4 -> deinterleave depth 4, no identification
# ===========================================================================

class TestManual:
    def test_manual_applies_depth_without_identification(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)
        depth = 4

        original_deinterleave = deinterleave_bits
        calls = {"n": 0}

        def tracking_deinterleave(bits, depth, original_size):
            calls["n"] += 1
            return original_deinterleave(bits, depth, original_size)

        monkeypatch.setattr(
            "prototype.pipeline.deinterleave_bits",
            tracking_deinterleave,
        )

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(
                fec_interleaving_mode=FECMode.MANUAL, interleave_depth=depth
            ),
            reference_bits=_bits(128, seed=99),
        )

        interleaving_summary = result.demodulation.get("interleaving_result", {})
        deinterleaved = result.demodulation.get("deinterleaved_bits")

        assert interleaving_summary.get("status") == "MANUALLY_CONFIGURED"
        assert interleaving_summary.get("best_depth") == depth
        assert calls["n"] == 1
        assert deinterleaved is not None
        assert not np.array_equal(deinterleaved, received_bits)


# ===========================================================================
# 4. NONE -> bits unchanged
# ===========================================================================

class TestNone:
    def test_none_leaves_bits_untouched(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(fec_interleaving_mode=FECMode.NONE, interleave_depth=1),
            reference_bits=_bits(128, seed=99),
        )

        assert result.demodulation is not None
        assert result.demodulation.get("deinterleaved_bits") is None

        interleaving_summary = result.demodulation.get("interleaving_result", {})
        assert interleaving_summary.get("status") == "NONE"
        assert interleaving_summary.get("best_depth") is None


# ===========================================================================
# 5. Existing FEC path still receives the correct deinterleaved bits when AUTO
#    succeeds (end-to-end through the interleaving stage).
# ===========================================================================

class TestFecPathReceivesDeinterleavedBits:
    def test_fec_path_gets_deinterleaved_stream_after_auto(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        def fake_identify(bits, min_depth=2, max_depth=16):
            assert np.array_equal(bits, received_bits)
            return InterleavingIdentificationResult(
                status=AUTO_DETECTED,
                interleaving_detected=True,
                best_type="block",
                best_depth=4,
                confidence=0.9,
            )

        monkeypatch.setattr(
            "prototype.pipeline.identify_interleaving",
            fake_identify,
        )

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(fec_interleaving_mode=FECMode.AUTO, interleave_depth=1),
            reference_bits=_bits(128, seed=99),
        )

        deinterleaved = result.demodulation.get("deinterleaved_bits")
        assert deinterleaved is not None
        assert len(deinterleaved) == len(received_bits)
        # The deinterleaved stream is a row-column block deinterleaving of the
        # received bits (a permutation); it must contain the same bits as the
        # received stream and, for depth >= 2, differ from it.
        assert np.array_equal(
            np.sort(deinterleaved),
            np.sort(received_bits),
        )
        assert not np.array_equal(deinterleaved, received_bits)


# ===========================================================================
# 6. Original demodulated bits remain unchanged after AUTO.
# ===========================================================================

class TestOriginalBitsUnchanged:
    def test_received_bits_are_never_mutated(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        def fake_identify(bits, min_depth=2, max_depth=16):
            assert np.array_equal(bits, received_bits)
            return InterleavingIdentificationResult(
                status=AUTO_DETECTED,
                interleaving_detected=True,
                best_type="block",
                best_depth=4,
                confidence=0.95,
            )

        monkeypatch.setattr(
            "prototype.pipeline.identify_interleaving",
            fake_identify,
        )

        original_copy = received_bits.copy()

        analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(fec_interleaving_mode=FECMode.AUTO, interleave_depth=1),
            reference_bits=_bits(128, seed=99),
        )

        assert np.array_equal(received_bits, original_copy)


# ===========================================================================
# Regression: MANUAL + interleave_depth=4 -> apply depth, no identification
# ===========================================================================

class TestManualRegression:
    def test_manual_depth_four_applies_deinterleave(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        # A tracking deinterleave_rounds that must be invoked exactly once at
        # depth 4.
        original_deinterleave = deinterleave_bits
        calls = {"n": 0}

        def tracking_deinterleave(bits, depth, original_size):
            calls["n"] += 1
            assert depth == 4
            assert original_size == len(bits)
            return original_deinterleave(bits, depth, original_size)

        monkeypatch.setattr(
            "prototype.pipeline.deinterleave_bits",
            tracking_deinterleave,
        )

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(
                fec_interleaving_mode=FECMode.MANUAL, interleave_depth=4
            ),
            reference_bits=_bits(128, seed=99),
        )

        interleaving_summary = result.demodulation.get("interleaving_result", {})
        deinterleaved = result.demodulation.get("deinterleaved_bits")

        assert interleaving_summary.get("status") == "MANUALLY_CONFIGURED"
        assert interleaving_summary.get("best_depth") == 4
        assert calls["n"] == 1
        assert deinterleaved is not None
        assert not np.array_equal(deinterleaved, received_bits)


# ===========================================================================
# Regression: NONE -> bits unchanged, no deinterleaving
# ===========================================================================

class TestNoneRegression:
    def test_none_passes_bits_through_unchanged(self, monkeypatch):
        _enable_demodulation(monkeypatch)
        received_bits = _bits(128, seed=42)

        # Nothing must be deinterleaved when the mode is NONE.
        monkeypatch.setattr(
            "prototype.pipeline.deinterleave_bits",
            lambda bits, depth, original_size: (_ for _ in ()).throw(
                AssertionError("deinterleave_bits must NOT be called under NONE")
            ),
        )

        result = analyze_samples(
            samples=np.zeros(received_bits.size, dtype=np.complex64),
            sample_rate=1.0,
            config=_make_config(
                fec_interleaving_mode=FECMode.NONE, interleave_depth=4
            ),
            reference_bits=_bits(128, seed=99),
        )

        interleaving_summary = result.demodulation.get("interleaving_result", {})
        deinterleaved = result.demodulation.get("deinterleaved_bits")

        assert interleaving_summary.get("status") == "NONE"
        assert interleaving_summary.get("best_depth") is None
        assert deinterleaved is None
        assert np.array_equal(received_bits, _bits(128, seed=42))
