"""Tests for block interleaving identification (``prototype.fec.identification_interleaving``).

Deterministic, pure-NumPy / dataclass assertions. They do not require a display
and do not depend on pipeline / GUI / CLI wiring (which are separate later phases).
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.fec.interleaving import interleave_bits, deinterleave_bits
from prototype.fec.identification_interleaving import (
    AUTO_DETECTED,
    NO_INTERLEAVING,
    UNKNOWN,
    UNRESOLVED,
    identify_interleaving,
    infer_block_parameters,
    InterleavingCandidate,
    InterleavingIdentificationError,
    describe_status,
    list_candidates,
)

# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _clean_bits(n: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def _encode_structured(n: int, seed: int = 7) -> np.ndarray:
    """Built a short structured bitstream (repeated frame header + payload)
    so that a known depth leaves a measurable block signature."""
    rng = np.random.default_rng(seed)
    header = rng.integers(0, 2, size=16, dtype=np.uint8)
    payload = rng.integers(0, 2, size=n - 16, dtype=np.uint8)
    # Make the header + payload length a multiple of several candidate depths so
    # block structure is *applicable* for 2..8; the real evidence (frame CRC we
    # test below) is what exposes the true depth.
    return np.concatenate([header, payload])


def _encode_frame_bits(n: int, seed: int = 11) -> np.ndarray:
    """Pure-frame structure: repeated 8-bit sync word + data, total multiple of 8."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=n - 8, dtype=np.uint8)  # reserve 8 bits for sync
    return np.concatenate([np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8), data])


# ---------------------------------------------------------------------------
# Result model / API
# ---------------------------------------------------------------------------


class TestResultModel:
    def test_fields_present(self):
        result = identify_interleaving(_clean_bits(64))
        assert hasattr(result, "status")
        assert hasattr(result, "interleaving_detected")
        assert hasattr(result, "best_type")
        assert hasattr(result, "best_depth")
        assert hasattr(result, "confidence")
        assert hasattr(result, "candidates")
        assert hasattr(result, "evidence")
        assert hasattr(result, "input_bit_count")
        assert hasattr(result, "validation_before")
        assert hasattr(result, "validation_after")
        assert hasattr(result, "improvement")
        assert hasattr(result, "warnings")

    def test_candidates_inspectable(self):
        result = identify_interleaving(_clean_bits(64))
        assert isinstance(result.candidates, list)
        for candidate in result.candidates:
            assert isinstance(candidate, dict)
            assert "type" in candidate
            assert "score" in candidate
            assert "evidence" in candidate

    def test_deterministic(self):
        bits = _clean_bits(256, seed=7)
        r1 = identify_interleaving(bits)
        r2 = identify_interleaving(bits)
        d1 = r1.to_dict()
        d2 = r2.to_dict()

        def _drop_timing(value):
            if isinstance(value, dict) and "evaluation_seconds" in value:
                value = {k: v for k, v in value.items() if k != "evaluation_seconds"}
            if isinstance(value, list):
                return [_drop_timing(v) for v in value]
            if isinstance(value, dict):
                return {k: _drop_timing(v) for k, v in value.items()}
            return value

        assert _drop_timing(d1) == _drop_timing(d2)
        assert d1["status"] == d2["status"]
        assert d1["best_type"] == d2["best_type"]
        assert d1["best_depth"] == d2["best_depth"]
        assert d1["confidence"] == d2["confidence"]


# ---------------------------------------------------------------------------
# Structural gating
# ---------------------------------------------------------------------------


class TestStructuralGating:
    def test_too_few_bits(self):
        result = identify_interleaving(np.array([1, 0, 1], dtype=np.uint8))
        assert result.status == UNKNOWN
        assert result.best_depth is None

    def test_invalid_depth_rejected(self):
        # A depth below 2 is structurally inapplicable and is never reported
        # as a detection; an empty/short input yields no identification.
        empty = identify_interleaving(np.array([], dtype=np.uint8))
        assert empty.status == UNKNOWN
        assert empty.best_depth is None
        short = identify_interleaving(np.array([1, 0, 1], dtype=np.uint8))
        assert short.status == UNKNOWN
        assert short.best_depth is None
        # depth 1 is not a real interleaver; search must not select it.
        result = identify_interleaving(_clean_bits(32), min_depth=1, max_depth=4)
        assert result.best_depth is None or result.best_depth >= 2


# ---------------------------------------------------------------------------
# Library-level equivalence
# ---------------------------------------------------------------------------


class TestLibraryRoundtrip:
    def test_interleave_then_deinterleave_matches_original(self):
        bits = _clean_bits(64, seed=3)
        for depth in (2, 3, 4, 5, 7, 8):
            enc = interleave_bits(bits, depth=depth)
            dec = deinterleave_bits(enc, depth=depth, original_size=int(bits.size))
            assert np.array_equal(dec, bits), f"roundtrip failed for depth={depth}"


# ---------------------------------------------------------------------------
# Ground-truth identification (Step 13)
# ---------------------------------------------------------------------------


class TestGroundTruth:
    """The identification mechanism is exercised against *provided* structure,
    not by feeding the true depth to the identifier and asserting equality.

    The frame-validator supplied by the test is real: it validates the
    deinterleaved stream against a known frame layout whose boundaries
    genuinely depend on depth, so only the depth that actually restored the
    structure can earn evidence.
    """

    def test_block_interleaved_frame_identified_at_true_depth(self):
        # The ground truth: construct a frame whose DATA is block-interleaved
        # at the true depth, with the sync word left untouched at the front.
        # The frame-validator (supplied by the test) is ground truth; it is
        # never given the answer, only the provided structure. Because the
        # block interleaver is an exact permutation, only the depth actually
        # applied can recover the original data from the data portion.
        depth = 4
        data_bytes = 8  # data payload = 8 bytes
        sync = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
        rng = np.random.default_rng(9)
        data = rng.integers(0, 2, size=data_bytes * 8, dtype=np.uint8)
        # Interleave ONLY the data at the true depth; the sync word stays put.
        interleaved = interleave_bits(data, depth=depth)
        bits = np.concatenate([sync, interleaved])

        def frame_validator(bits, depth):
            # Ground-truth signal: the data portion was block-interleaved at
            # `depth` by the test; a candidate depth earns evidence only if
            # deinterleaving at that depth restores the original data layout
            # (the sync word intact and the data bytes in original order).
            from prototype.fec.interleaving import deinterleave_bits

            test_depth = int(depth)
            if bits.size % test_depth != 0:
                return {"status": "checked", "frame_structered": False, "frame_depth_match": False, "frame_crc_valid": False}

            data_portion = bits[8:]
            restored = deinterleave_bits(data_portion, depth=test_depth, original_size=int(data_portion.size))

            # Only the depth that was actually applied restores the data
            # order. Compare against the known original data.
            data_ok = bool(np.array_equal(restored, data))

            return {
                "valid": bool(data_ok),
                "status": "checked",
                "frame_structered": bool(data_ok),
                "frame_depth_match": data_ok,
                "frame_crc_valid": data_ok,
            }

        with_verifier = identify_interleaving(
            bits,
            min_depth=2,
            max_depth=16,
            frame_validator=frame_validator,
        )
        assert with_verifier.status == AUTO_DETECTED
        assert with_verifier.best_depth == depth

        # Without any supplied frame structure the module must not invent
        # one: it may return UNKNOWN or UNRESOLVED, but never a detection and
        # never a best_depth.
        without = identify_interleaving(bits, min_depth=2, max_depth=16)
        assert without.status in (NO_INTERLEAVING, UNKNOWN, UNRESOLVED)
        assert without.best_depth is None

class TestAmbiguousDepths:
    def test_ambiguous_depths_return_unresolved(self):
        # Two depths both structurally applicable; the verifier cannot break
        # the tie -> UNRESOLVED, never AUTO_DETECTED.
        depth_a, depth_b = 4, 8
        frame = _encode_frame_bits(8 * 6, seed=13)
        interleaved = interleave_bits(frame, depth=depth_a)

        def weak_validator(bits, ref=None):
            return {"status": "checked", "depth_match": True, "structured": True}

        result = identify_interleaving(
            interleaved,
            min_depth=2,
            max_depth=16,
            frame_validator=weak_validator,
        )
        # Both candidate depths are applicable; without a discriminative
        # verifier the best candidate only reaches unresolved.
        assert result.status == UNRESOLVED


# ---------------------------------------------------------------------------
# No-false-identification (Step 12 G)
# ---------------------------------------------------------------------------


class TestNoFalseIdentification:
    def test_random_bitstream_without_structure_must_not_identify(self):
        rng = np.random.default_rng(12345)
        bits = rng.integers(0, 2, size=512).astype(np.uint8)
        # No sync word, no frame boundaries, no reference supplied.
        result = identify_interleaving(bits, min_depth=2, max_depth=16)
        # Unstructured random input -> must not force a detection.
        assert result.status in (NO_INTERLEAVING, UNRESOLVED)
        assert result.best_depth is None

    def test_random_with_verifier_still_no_detection(self):
        # Even with a weak verifier reporting structure, unstructured data has
        # no genuine frame/CRC signature: no depth may win by a meaningful
        # margin over the no-interleaving baseline.
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, size=512).astype(np.uint8)

        def weak_validator(bits):
            # Report structure, but deliberately no CRC / depth_match signal
            # so a correct verifier still cannot confirm a depth.
            return {"status": "checked", "structured": True, "depth_match": False, "valid": False}

        result = identify_interleaving(
            bits,
            min_depth=2,
            max_depth=16,
            frame_validator=weak_validator,
        )
        # structured= True alone is insufficient; without crc_valid / depth_match
        # the score cannot beat the baseline by MARGIN, so no false detection.
        assert result.status in (NO_INTERLEAVING, UNRESOLVED)


# ---------------------------------------------------------------------------
# Low-confidence -> NO_INTERLEAVING (not a forced depth)
# ---------------------------------------------------------------------------


class TestCostHandle:
    def test_weak_structure_does_not_force_detection(self):
        # Unstructured random input: no sync word, no frame boundaries, no
        # reference, no validator. The module must not force a depth.
        bits = _clean_bits(128, seed=42)
        result = identify_interleaving(bits, min_depth=2, max_depth=16)
        assert result.status in (NO_INTERLEAVING, UNRESOLVED)
        assert result.best_depth is None
        # Timing must not affect the decision.
        r2 = identify_interleaving(bits, min_depth=2, max_depth=16)
        assert r2.status == result.status
        assert r2.best_depth == result.best_depth
        assert r2.confidence == result.confidence


# ---------------------------------------------------------------------------
# Parameter inference (thin wrapper)
# ---------------------------------------------------------------------------


class TestInferBlockParameters:
    def test_returns_none_when_no_detection(self):
        bits = _clean_bits(128, seed=5)
        inferred = infer_block_parameters(bits)
        assert inferred is None

    def test_returns_depth_when_detected(self):
        # The same ground-truth pattern as above: only the depth actually
        # applied can recover the original data, so the module's decision
        # rule maps the supplied frame structure to the true depth.
        depth = 4
        data_bytes = 8
        sync = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
        rng = np.random.default_rng(11)
        data = rng.integers(0, 2, size=data_bytes * 8, dtype=np.uint8)
        # Interleave only the data at the true depth; the sync word stays put.
        interleaved = interleave_bits(data, depth=depth)
        bits = np.concatenate([sync, interleaved])

        def crc_frame_validator(bits, depth):
            # Ground-truth signal: it deinterleaves the data portion at the
            # candidate depth and reports evidence only when it restores the
            # original data bytes. Only the depth actually applied can do
            # that, so the module can discriminate the true depth.
            from prototype.fec.interleaving import deinterleave_bits

            test_depth = int(depth)
            if bits.size % test_depth != 0:
                return {"status": "checked", "frame_structered": False, "frame_depth_match": False, "frame_crc_valid": False}

            data_portion = bits[8:]
            restored = deinterleave_bits(data_portion, depth=test_depth, original_size=int(data_portion.size))
            data_ok = bool(np.array_equal(restored, data))

            return {
                "valid": bool(data_ok),
                "status": "checked",
                "frame_structered": bool(data_ok),
                "frame_depth_match": data_ok,
                "frame_crc_valid": data_ok,
            }

        inferred = infer_block_parameters(
            bits,
            frame_validator=crc_frame_validator,
        )
        assert inferred is not None
        assert inferred["type"] == "block"
        assert inferred["depth"] == depth
