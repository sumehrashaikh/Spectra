"""Decoder-evidence tests for the extended automatic FEC identification.

The identifier used to evaluate only ``none | repetition3 | hamming74 |
conv12``.  It now also evaluates ``reedsolomon``, ``ldpc`` and
``concatenated`` using *real decoder evidence*:

    reedsolomon   -> residual syndromes: uncorrectable block count
    ldpc          -> bit-flip count: only a zero-flip (already valid)
                     codeword is credited
    concatenated  -> inner Viterbi path metric + outer RS block status

No scheme is ever claimed from structure alone: on random data every
candidate must stay below the decision threshold, and an ambiguous
verdict must be reported as such instead of being silently tie-broken.

These tests are pure NumPy and deterministic; they need no display.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.fec import identification
from prototype.fec.framework import encode_bits

# Every scheme that must identify itself from its own clean codeword.
EVIDENCE_SCHEMES = (
    "repetition3",
    "hamming74",
    "conv12",
    "reedsolomon",
    "ldpc",
    "concatenated",
)


def _payload(n: int = 1024, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def _codeword(scheme: str, n: int = 1024, seed: int = 5) -> np.ndarray:
    encoded, _ = encode_bits(_payload(n, seed), scheme)
    return np.asarray(encoded, dtype=np.uint8)


def _score_by_scheme(result) -> dict[str, float]:
    return {c["candidate"]: float(c["score"]) for c in result.candidates}


def _scheme_evidence(result, scheme: str) -> dict:
    for candidate in result.candidates:
        if candidate["candidate"] == scheme:
            return candidate["evidence"]
    raise AssertionError(f"{scheme} missing from identification candidates")


# ---------------------------------------------------------------------------
# Candidate set
# ---------------------------------------------------------------------------


def test_candidate_set_includes_evidence_scored_schemes():
    candidates = identification.list_candidates()
    for scheme in EVIDENCE_SCHEMES:
        assert scheme in candidates
    assert "none" in candidates


# ---------------------------------------------------------------------------
# Each clean codeword identifies itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", EVIDENCE_SCHEMES)
def test_clean_codeword_identifies_its_own_scheme(scheme):
    result = identification.identify_fec(_codeword(scheme))
    assert result.status == "AUTO_DETECTED", (
        f"{scheme} was not auto-detected (status={result.status}, "
        f"scores={_score_by_scheme(result)})"
    )
    assert result.best_scheme == scheme
    assert result.confidence >= identification.MIN_CONFIDENCE
    # The winning hypothesis must win on its own decoder evidence.
    scores = _score_by_scheme(result)
    assert scores[scheme] == max(scores.values())


# ---------------------------------------------------------------------------
# No false positives: nothing is claimed from random data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_bits_never_detect_a_scheme(seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, 2016).astype(np.uint8)
    result = identification.identify_fec(bits)
    assert result.status in ("UNKNOWN", "UNRESOLVED")
    assert result.best_scheme is None
    for candidate in result.candidates:
        assert float(candidate["score"]) < identification.MIN_CONFIDENCE


def test_other_schemes_do_not_shadow_the_true_scheme():
    for scheme in EVIDENCE_SCHEMES:
        result = identification.identify_fec(_codeword(scheme))
        assert result.best_scheme == scheme, (
            f"{scheme} codeword was claimed by {result.best_scheme}"
        )


# ---------------------------------------------------------------------------
# The evidence is real decoder output, not structural guesswork
# ---------------------------------------------------------------------------


def test_reedsolomon_evidence_reports_residual_syndrome_status():
    result = identification.identify_fec(_codeword("reedsolomon"))
    evidence = _scheme_evidence(result, "reedsolomon")
    assert evidence["blocks_processed"] > 0
    assert evidence["uncorrectable_blocks"] == 0
    assert "reasoning" in evidence


def test_reedsolomon_detects_codeword_with_correctable_channel_errors():
    encoded = _codeword("reedsolomon")
    rng = np.random.default_rng(9)
    # A handful of bit errors: within the t=4 symbol-error budget, so the
    # block still decodes and the scheme is still identifiable.
    idx = rng.choice(encoded.size, 8, replace=False)
    corrupted = encoded.copy()
    corrupted[idx] ^= 1
    result = identification.identify_fec(corrupted)
    assert result.status == "AUTO_DETECTED"
    assert result.best_scheme == "reedsolomon"


def test_reedsolomon_short_block_is_structurally_rejected():
    # 400 bits is not a multiple of the 320-bit RS block.
    result = identification.identify_fec(_payload(400))
    evidence = _scheme_evidence(result, "reedsolomon")
    assert evidence.get("uncorrectable_blocks") is None or "not a multiple" in str(
        evidence.get("reason", "")
    )


def test_ldpc_requires_a_zero_flip_codeword():
    clean = identification.identify_fec(_codeword("ldpc"))
    clean_evidence = _scheme_evidence(clean, "ldpc")
    assert clean.best_scheme == "ldpc"
    assert clean_evidence["zero_flip_decode"] is True
    assert clean_evidence["bit_flips"] == 0

    # With errors the bit-flipper has to modify bits, so the LDPC
    # hypothesis must no longer be credited (honest: stays unresolved).
    encoded = _codeword("ldpc")
    idx = np.random.default_rng(3).choice(encoded.size, 40, replace=False)
    corrupted = encoded.copy()
    corrupted[idx] ^= 1
    noisy = identification.identify_fec(corrupted)
    noisy_evidence = _scheme_evidence(noisy, "ldpc")
    assert float(noisy_evidence.get("bit_flips", 0)) > 0
    assert noisy_evidence["zero_flip_decode"] is False
    assert noisy.best_scheme != "ldpc"


def test_concatenated_evidence_reports_both_layers():
    result = identification.identify_fec(_codeword("concatenated"))
    evidence = _scheme_evidence(result, "concatenated")
    assert evidence["inner_layer_clean"] is True
    assert evidence["outer_layer_clean"] is True
    assert evidence["outer_blocks"] > 0
    assert evidence["outer_uncorrectable_blocks"] == 0
    assert evidence["inner_path_metric"] == 0.0


def test_concatenated_ambiguity_with_conv12_is_reported():
    """A concatenated codeword is also a valid conv12 codeword.

    The identifier resolves it in favour of the more specific hypothesis
    but must publish the competing hypothesis rather than hide it.
    """
    result = identification.identify_fec(_codeword("concatenated"))
    assert result.best_scheme == "concatenated"
    assert "conv12" in (result.evidence.get("also_consistent") or [])
    assert result.evidence.get("ambiguous") is True
    scores = _score_by_scheme(result)
    assert scores["concatenated"] > scores["conv12"]


def test_conv12_stream_is_not_claimed_as_concatenated():
    result = identification.identify_fec(_codeword("conv12"))
    assert result.best_scheme == "conv12"
    assert "concatenated" not in (result.evidence.get("also_consistent") or [])


# ---------------------------------------------------------------------------
# Evidence plumbing (result contract)
# ---------------------------------------------------------------------------


def test_result_evidence_records_decision_and_best_scheme():
    result = identification.identify_fec(_codeword("reedsolomon"))
    assert result.evidence["decided_by"] == "reedsolomon"
    assert result.evidence["input_bit_count"] == int(_codeword("reedsolomon").size)
    assert result.evidence["confidence"] == pytest.approx(result.confidence)


def test_identification_result_is_json_safe():
    import json

    for scheme in EVIDENCE_SCHEMES:
        payload = identification.identify_fec(_codeword(scheme)).to_dict()
        json.dumps(payload)
