"""Reference-validated FEC / interleaving and PSK alignment regressions.

The transmitted reference is the same stream the BER stage uses.  These
tests pin down what the receiver may do with it:

1.  AUTO FEC identification is validated against the reference decode, so a
    lossy received stream that the blind identifier cannot decide alone is
    still decoded to the payload (conv12, RS + block).
2.  The block deinterleaver depth is resolved jointly with the scheme when
    the structural identifier alone reports UNRESOLVED.
3.  The QPSK codeword-alignment search resolves the whole-symbol origin on
    both sides of zero (the previous search could only advance the
    candidate, never the reference, so a resolvable stream stayed at ~0.5).
4.  Without a reference nothing is invented: the blind path still reports
    no decoded payload.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.tests import demo_captures as dc


def _case(modulation: str, fec: str | None, family: str | None):
    for case in dc.demo_cases():
        if (
            case["modulation"] == modulation
            and case["fec_scheme"] == fec
            and case["interleave_family"] == family
        ):
            return dc.build_capture(**case)
    raise AssertionError(
        f"demo case not found: {modulation}/{fec}/{family}"
    )


def _prefix_matches(decoded: np.ndarray, expected: np.ndarray) -> float:
    decoded = np.asarray(decoded, dtype=np.uint8).reshape(-1)
    expected = np.asarray(expected, dtype=np.uint8).reshape(-1)
    n = min(decoded.size, expected.size)
    if n == 0:
        return 0.0
    return float(np.mean(decoded[:n] == expected[:n]))


def test_reference_validated_fec_decodes_conv12_without_interleaving():
    from prototype.pipeline import analyze_samples

    capture = _case("16-QAM", "conv12", None)
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    fec = (result.demodulation or {}).get("fec") or {}
    assert fec.get("scheme") == "conv12"
    assert fec.get("source") == "auto_identified_reference_validated"
    assert fec.get("status") == "decoded_reference_validated"
    # The decoded payload is the transmitted payload (the capture loses its
    # pulse-shaping tail, so the last bits may be absent - never invented).
    assert _prefix_matches(fec["decoded_bits"], capture.payload_bits) > 0.98
    # Post-FEC BER is measured against the reference decode.
    assert fec.get("post_fec_ber") is not None
    assert fec["post_fec_ber"] <= 0.01
    assert result.ber is not None and result.ber["ber"] <= 0.01


def test_reference_validated_search_records_the_identification_verdict():
    """A successful reference-validated search fills ``fec_identification``.

    The GUI/JSON "Auto FEC" row reads that slot; without this the row said
    "not run" even though the scheme had just been validated against the
    transmitted reference.
    """
    from prototype.pipeline import analyze_samples

    capture = _case("16-QAM", "conv12", None)
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    identification = (result.demodulation or {}).get("fec_identification") or {}
    assert identification.get("status") == "AUTO_DETECTED"
    assert identification.get("best_scheme") == "conv12"
    assert identification.get("confirmed_by") == "reference_payload_agreement"
    assert float(identification.get("confidence") or 0.0) > 0.0


def test_reference_validated_search_resolves_block_depth_and_reedsolomon():
    from prototype.pipeline import analyze_samples

    capture = _case("16-QAM", "reedsolomon", "block")
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") == "AUTO_DETECTED"
    assert interleaving.get("best_depth") == 8
    assert interleaving.get("family") == "block"
    assert (
        interleaving.get("evidence", {}).get("mode")
        == "reference_validated_joint_search"
    )

    fec = (result.demodulation or {}).get("fec") or {}
    assert fec.get("scheme") == "reedsolomon"
    assert fec.get("uncorrectable_blocks") == 0
    decoded = np.asarray(fec["decoded_bits"], dtype=np.uint8)
    assert decoded.size == capture.payload_bits.size
    assert np.array_equal(decoded, capture.payload_bits)
    assert fec.get("post_fec_ber") == 0.0


def test_reference_validated_search_resolves_pseudo_random_interleaving():
    """The joint search covers every de-interleaver family, not just block.

    ``16-QAM concatenated pseudo_random`` used to stay UNRESOLVED in AUTO
    mode because only block depths were probed; the seed genuinely
    interleaves the concatenated code stream, so validating whole
    (family, parameter, scheme) hypotheses against the reference resolves
    both stages.
    """
    from prototype.pipeline import analyze_samples

    capture = _case("16-QAM", "concatenated", "pseudo_random")
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") == "AUTO_DETECTED"
    assert interleaving.get("family") == "pseudo_random"
    assert interleaving.get("best_depth") == 3
    assert (
        interleaving.get("evidence", {}).get("mode")
        == "reference_validated_joint_search"
    )

    fec = (result.demodulation or {}).get("fec") or {}
    assert fec.get("scheme") == "concatenated"
    decoded = np.asarray(fec["decoded_bits"], dtype=np.uint8)
    assert np.array_equal(decoded, capture.payload_bits)


def test_interleaving_free_fec_capture_is_not_forced_through_a_deinterleaver():
    """No interleaver at all: the search rejects every permutation and the
    convolutional capture is still resolved end to end."""
    from prototype.pipeline import analyze_samples

    capture = _case("16-QAM", "conv12", None)
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("family") is None
    assert interleaving.get("status") == "NONE"
    decoded = np.asarray(
        ((result.demodulation or {}).get("fec") or {})["decoded_bits"],
        dtype=np.uint8,
    )
    prefix = _prefix_matches(decoded, capture.payload_bits)
    assert prefix > 0.98


def test_qpsk_alignment_resolves_whole_symbol_origin():
    """The origin search must cover reference-ahead offsets, not just the
    candidate-ahead side; a resolvable QPSK stream must report aligned=True
    with a measured BER rather than the previous ~0.5 fold."""
    from prototype.pipeline import analyze_samples

    capture = _case("QPSK", None, "block")
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    alignment = (result.demodulation or {}).get("codeword_alignment") or {}
    assert alignment.get("aligned") is True, alignment
    assert alignment.get("aligned_bit_errors", 1) < alignment.get(
        "raw_bit_errors", 0
    )
    assert result.ber is not None
    assert result.ber["status"] == "measured"
    assert result.ber["ber"] < float(alignment["raw_bit_errors"]) / max(
        1, alignment["raw_compared_bits"]
    )


def test_blind_fec_identification_stays_honest_without_reference():
    """No reference: the lossy received stream may not clear the identifier,
    and no decoded payload may be claimed."""
    from prototype.pipeline import analyze_samples

    capture = _case("16-QAM", "conv12", None)
    result = analyze_samples(capture.samples, capture.sample_rate)

    identification = (result.demodulation or {}).get("fec_identification") or {}
    assert identification.get("status") in ("UNKNOWN", "UNRESOLVED", "FAILED")
    assert identification.get("best_scheme") is None
    assert (result.demodulation or {}).get("fec") is None
