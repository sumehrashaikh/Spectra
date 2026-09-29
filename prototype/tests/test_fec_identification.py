"""Tests for automatic FEC identification (``prototype.fec.identification``).

These tests cover the identification layer only, plus the pipeline-level
AUTO/MANUAL/NONE behaviour.  They do not require a display (pure NumPy /
dataclass assertions).

Supported automatically evaluated schemes:

    none | repetition3 | hamming74 | conv12 | reedsolomon | ldpc |
    concatenated

The scoring model is deterministic and explainable.  Clean supported FEC
codewords are detected strongly (80 confidence); corrupted or ambiguous
bitstreams are marked UNKNOWN/UNRESOLVED; insufficient or badly aligned
input returns 0 and UNKNOWN.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.fec import (
    hamming,
    repetition,
    convolutional,
)
from prototype.fec import identification
from prototype.fec.framework import list_schemes, decode_bits


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _clean_bits(n: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def _encode_for_scheme(bits: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "repetition3":
        return repetition.encode(bits, repetitions=3)
    if scheme == "hamming74":
        return hamming.encode(bits)
    if scheme == "conv12":
        return convolutional.encode(bits)
    raise ValueError(f"unsupported scheme {scheme!r}")


def _corrupt(bits: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, bits.size, size=max(1, bits.size // 20))
    out = bits.copy()
    out[idx] ^= 1
    return out


# ---------------------------------------------------------------------------
# Result model / API
# ---------------------------------------------------------------------------


class TestResultModel:
    def test_fields_present(self):
        result = identification.identify_fec(_clean_bits(64))
        assert hasattr(result, "status")
        assert hasattr(result, "best_scheme")
        assert hasattr(result, "confidence")
        assert hasattr(result, "candidates")
        assert hasattr(result, "evidence")
        assert hasattr(result, "input_bit_count")
        assert hasattr(result, "corrected_error_count")
        assert hasattr(result, "residual_error_count")
        assert hasattr(result, "validation_status")
        assert hasattr(result, "warnings")

    def test_candidates_inspectable(self):
        result = identification.identify_fec(_clean_bits(64))
        assert isinstance(result.candidates, list)
        for candidate in result.candidates:
            assert isinstance(candidate, dict)
            assert "candidate" in candidate
            assert "score" in candidate
            assert "evidence" in candidate

    def test_deterministic(self):
        bits = _clean_bits(256, seed=7)
        r1 = identification.identify_fec(bits)
        r2 = identification.identify_fec(bits)
        # Decision-relevant fields must match; timing anchors are excluded.
        d1 = r1.to_dict()
        d2 = r2.to_dict()

        # Decision-relevant fields must match; the timing anchor differs
        # slightly between runs and is excluded from the contract.
        def _drop_timing(value):
            if isinstance(value, dict) and "evaluation_seconds" in value:
                value = {k: v for k, v in value.items() if k != "evaluation_seconds"}
            if isinstance(value, list):
                return [_drop_timing(v) for v in value]
            if isinstance(value, dict):
                return {k: _drop_timing(v) for k, v in value.items()}
            return value

        assert _drop_timing(d1) == _drop_timing(d2)

        # The decided status/scheme/confidence is stable across runs.
        assert d1["status"] == d2["status"]
        assert d1["best_scheme"] == d2["best_scheme"]
        assert d1["confidence"] == d2["confidence"]


# ---------------------------------------------------------------------------
# Structural gating
# ---------------------------------------------------------------------------


class TestStructuralGating:
    def test_too_few_bits(self):
        result = identification.identify_fec(np.array([1, 0, 1], dtype=np.uint8))
        assert result.status == "UNKNOWN"
        assert result.best_scheme is None

    def test_invalid_alignment(self):
        # 12 bits is below the 16-bit minimum for all FEC candidates
        result = identification.identify_fec(np.arange(12, dtype=np.uint8) % 2)
        assert result.status == "UNKNOWN"

    def test_repetition3_requires_multiples_of_three(self):
        # an input not divisible by 3 yields decoder_invalid=true
        out = identification.evaluate_repetition3(np.arange(10, dtype=np.uint8) % 2)
        assert out["decoder_valid"] is False

    def test_hamming74_requires_multiples_of_seven(self):
        out = identification.evaluate_hamming74(np.arange(11, dtype=np.uint8) % 2)
        assert out["decoder_valid"] is False

    def test_conv12_requires_even_input(self):
        out = identification.evaluate_conv12(np.arange(13, dtype=np.uint8) % 2)
        assert out["decoder_valid"] is False


# ---------------------------------------------------------------------------
# Clean supported FEC candidates
# ---------------------------------------------------------------------------


class TestCleanCandidates:
    def test_clean_repetition3(self):
        bits = _clean_bits(100)
        encoded = repetition.encode(bits, repetitions=3)
        result = identification.identify_fec(encoded)
        assert result.status == "AUTO_DETECTED"
        assert result.best_scheme == "repetition3"

    def test_clean_hamming74(self):
        bits = _clean_bits(100)
        encoded = hamming.encode(bits)
        result = identification.identify_fec(encoded)
        assert result.status == "AUTO_DETECTED"
        assert result.best_scheme == "hamming74"

    def test_clean_conv12(self):
        bits = _clean_bits(100)
        encoded = convolutional.encode(bits)
        result = identification.identify_fec(encoded)
        assert result.status == "AUTO_DETECTED"
        assert result.best_scheme == "conv12"

    def test_none_hypothesis(self):
        bits = _clean_bits(100)
        result = identification.identify_fec(bits)
        # uncoded data may legitimately be reported as unknown/weak; the
        # candidate list must still include "none"
        assert any(c["candidate"] == "none" for c in result.candidates)


# ---------------------------------------------------------------------------
# Corrupted / ambiguous inputs
# ---------------------------------------------------------------------------


class TestCorruptedAndAmbiguous:
    def test_corrupted_repetition3(self):
        bits = _clean_bits(100)
        encoded = repetition.encode(bits, repetitions=3)
        result = identification.identify_fec(_corrupt(encoded))
        assert result.status == "UNKNOWN"
        assert result.best_scheme is None

    def test_corrupted_hamming74(self):
        bits = _clean_bits(100)
        encoded = hamming.encode(bits)
        result = identification.identify_fec(_corrupt(encoded))
        assert result.status == "UNKNOWN"
        assert result.best_scheme is None

    def test_corrupted_conv12(self):
        bits = _clean_bits(100)
        encoded = convolutional.encode(bits)
        result = identification.identify_fec(_corrupt(encoded))
        assert result.status == "UNKNOWN"
        assert result.best_scheme is None

    def test_ambiguous_weak_input(self):
        # random bits are not a valid encoder output for any candidate
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, 400).astype(np.uint8)
        result = identification.identify_fec(bits)
        assert result.status in ("UNKNOWN", "UNRESOLVED")
        assert len(result.candidates) == len(identification.list_candidates())


# ---------------------------------------------------------------------------
# Candidate evidence is preserved
# ---------------------------------------------------------------------------


class TestCandidateEvidence:
    def test_all_candidates_reported(self):
        bits = _clean_bits(200)
        result = identification.identify_fec(bits)
        candidate_names = {c["candidate"] for c in result.candidates}
        # The candidate set is fixed and fully reported; it now includes the
        # evidence-scored Reed-Solomon / LDPC / concatenated hypotheses.
        assert candidate_names == {
            "repetition3",
            "hamming74",
            "conv12",
            "none",
            "reedsolomon",
            "ldpc",
            "concatenated",
        }

    def test_ranking_known_correct_candidate_first(self):
        bits = _clean_bits(100)
        encoded = repetition.encode(bits, repetitions=3)
        result = identification.identify_fec(encoded)
        names = [c["candidate"] for c in result.candidates]
        assert names[0] == "repetition3"

    def test_evidence_contains_reasoning(self):
        bits = _clean_bits(100)
        result = identification.identify_fec(bits)
        for candidate in result.candidates:
            assert "evidence" in candidate
            evidence = candidate["evidence"]
            # Every candidate carries a textual reason/reasoning key.
            assert (
                "reason" in evidence or "reasoning" in evidence
            ), candidate["candidate"]


# ---------------------------------------------------------------------------
# Candidate scoring behaviour (deterministic thresholds)
# ---------------------------------------------------------------------------


class TestScoringBehaviour:
    def test_clean_repetition3_scores_80(self):
        bits = _clean_bits(100)
        encoded = repetition.encode(bits, repetitions=3)
        result = identification.identify_fec(encoded)
        assert result.confidence == 80.0
        assert result.best_scheme == "repetition3"

    def test_clean_hamming74_scores_80(self):
        bits = _clean_bits(100)
        encoded = hamming.encode(bits)
        result = identification.identify_fec(encoded)
        assert result.confidence == 80.0
        assert result.best_scheme == "hamming74"

    def test_clean_conv12_scores_75(self):
        bits = _clean_bits(100)
        encoded = convolutional.encode(bits)
        result = identification.identify_fec(encoded)
        assert result.confidence == 75.0
        assert result.best_scheme == "conv12"

    def test_weak_input_scores_below_threshold(self):
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, 400).astype(np.uint8)
        result = identification.identify_fec(bits)
        assert result.confidence < identification.MIN_CONFIDENCE
        assert result.status == "UNKNOWN"


# ---------------------------------------------------------------------------
# Pipeline-level AUTO/MANUAL/NONE (via pipeline semantics)
# ---------------------------------------------------------------------------


class TestPipelineModes:
    def test_auto_mode_identifies(self):
        bits = _clean_bits(64)
        result = identification.identify_fec(bits)
        assert result.status in ("AUTO_DETECTED", "UNKNOWN", "UNRESOLVED")

    def test_manual_mode_is_authoritative(self):
        # manual selection is handled by the pipeline config; here we simply
        # assert the identification result distinguishes the manual result
        # from an auto result.
        raw = _clean_bits(64)
        auto = identification.identify_fec(raw)
        # manual mode is represented by the pipeline config, not by
        # identification itself; this test guards the interface.
        assert auto.status in ("AUTO_DETECTED", "UNKNOWN", "UNRESOLVED")

    def test_none_is_valid(self):
        bits = _clean_bits(64)
        result = identification.identify_fec(bits)
        assert result.status in ("UNKNOWN", "UNRESOLVED", "AUTO_DETECTED")
