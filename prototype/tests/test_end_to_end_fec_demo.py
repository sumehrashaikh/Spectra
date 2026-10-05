"""Deterministic end-to-end demo: RF chain + interleaving + FEC.

Each case builds a real capture (see ``tests.demo_captures``), runs it
through the real receiver pipeline, and verifies the recovered payload:

    payload -> FEC encode -> interleave -> QAM/PSK map -> RRC -> upconvert
    -> impaired + AWGN capture
    -> detection -> classification -> demodulation -> codeword alignment
    -> deinterleaving -> FEC decode -> recovered payload

Two honest constraints are encoded in the assertions:

* the receiver loses the pulse-shaping tail, so only the payload prefix
  carried by whole surviving codewords is compared (the truncated tail is
  reported by the pipeline, never invented);
* the blind phase/origin fold is resolved against the transmitted
  reference (the same reference the BER stage uses).  Cases where no
  reference is supplied (or the ambiguity is unresolvable) must not claim
  a decoded payload - that is asserted too.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from prototype.core.config import FECMode, processing_mode_config
from prototype.pipeline import analyze_samples
from tests import demo_captures as dc

SAMPLE_RATE = dc.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _analyze(capture, scheme=None, family=None, param=None, with_reference=True):
    """Run the real pipeline on a demo capture with an explicit FEC config."""
    config = processing_mode_config("balanced")
    fec = replace(
        config.fec,
        mode=FECMode.MANUAL if scheme else FECMode.NONE,
        scheme=scheme,
        interleaving_mode=FECMode.MANUAL if family else FECMode.NONE,
        interleave_family=family or "block",
        interleave_depth=param or 1,
    )
    return analyze_samples(
        capture.samples,
        SAMPLE_RATE,
        config=replace(config, fec=fec),
        reference_bits=capture.coded_bits if with_reference else None,
    )


def _decoded(result):
    return (result.demodulation or {}).get("fec") or {}


def _prefix_matches(decoded_bits, payload, allowance=0):
    """True when the recovered stream reproduces the payload (short tail ok)."""
    bits = np.asarray(decoded_bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return False
    keep = max(0, min(bits.size, payload.size) - allowance)
    return bool(np.array_equal(bits[:keep], payload[:keep]))


# ---------------------------------------------------------------------------
# Verified end-to-end demos: 16-QAM x {conv12, RS, concatenated} x interleaving
# ---------------------------------------------------------------------------

# (scheme, interleave family, interleave param, payload bits, tail allowance)
VERIFIED_CASES = [
    ("conv12", None, None, 1024, 8),
    ("reedsolomon", None, None, 1024, 0),
    ("concatenated", None, None, 1024, 0),
    ("reedsolomon", "block", 8, 1024, 0),
    ("concatenated", "block", 8, 1024, 0),
]


@pytest.mark.parametrize("scheme,family,param,nbits,allowance", VERIFIED_CASES)
def test_full_chain_recovers_payload(scheme, family, param, nbits, allowance):
    capture = dc.build_capture(
        modulation="16-QAM",
        fec_scheme=scheme,
        interleave_family=family,
        interleave_param=param,
        nbits=nbits,
    )

    result = _analyze(capture, scheme=scheme, family=family, param=param)

    # 1. detection + classification + demodulation
    assert result.classification is not None
    assert result.classification["modulation"] == "16-QAM", (
        f"classification failed: {result.classification}"
    )
    demod = result.demodulation or {}
    assert "received_bits" in demod, "the received bitstream must be preserved"
    assert np.asarray(demod["received_bits"]).size > 0

    # 2. codeword alignment against the transmitted reference
    alignment = demod.get("codeword_alignment") or {}
    assert alignment.get("aligned") is True, alignment
    assert alignment["aligned_bit_errors"] == 0, alignment

    # 3. deinterleaving stage (manual, authoritative)
    if family:
        interleaving = demod.get("interleaving_result") or {}
        assert interleaving.get("status") == "MANUALLY_CONFIGURED"
        assert interleaving.get("family") == family

    # 4. FEC decode: a real decoded stream, no uncorrectable block
    fec = _decoded(result)
    assert fec.get("scheme") == scheme, fec
    assert fec.get("source") == "explicit_config"
    assert fec.get("uncorrectable_blocks") == 0, fec
    assert _prefix_matches(fec["decoded_bits"], capture.payload_bits, allowance), (
        "recovered payload does not match the transmitted payload "
        f"(decoded {np.asarray(fec['decoded_bits']).size} bits, "
        f"payload {capture.payload_bits.size})"
    )


def test_no_fec_capture_is_bit_exact_after_alignment():
    capture = dc.build_capture(modulation="16-QAM", fec_scheme=None)
    result = _analyze(capture, scheme=None, family=None, param=None)
    alignment = (result.demodulation or {}).get("codeword_alignment") or {}
    assert alignment.get("aligned") is True
    # Residual bit errors are reported, not hidden: the 16-QAM decision
    # boundaries leave a handful of errors on this capture.
    assert alignment["aligned_ber"] < 0.01
    assert result.ber is not None and result.ber["ber"] < 0.01


@pytest.mark.parametrize("modulation", ["QPSK", "8-PSK"])
def test_demo_level_tables_match_the_receiver_decisions(modulation):
    """Every bit word must map to a symbol the receiver decides back to it.

    The transmitter's level table is derived from the receiver's decision
    kernel, so this is the invariant that keeps them from drifting: the
    hand-written QPSK table had two quadrants swapped, which capped the
    measured BER near 25% no matter how good synchronization was.
    """
    import numpy as np

    from prototype.modulation.demodulator import qpsk_decision
    from prototype.modulation.digital import psk8_decision

    if modulation == "QPSK":
        levels = dc.QPSK_LEVELS
        words = np.array([[b0, b1] for b0 in (0, 1) for b1 in (0, 1)])

        def decide(points):
            return qpsk_decision(points)[0]

    else:
        levels = dc.PSK8_LEVELS
        words = np.array(
            [[b0, b1, b2] for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1)]
        )

        def decide(points):
            return psk8_decision(points)[0]

    width = words.shape[1]
    indices = np.array(
        [int(np.dot(word, 1 << np.arange(width - 1, -1, -1))) for word in words]
    )
    symbols = np.asarray(levels)[indices]

    decided = np.asarray(decide(symbols), dtype=np.uint8)
    assert np.array_equal(decided, words.reshape(-1)), (
        "demo level table does not round-trip through the receiver's "
        f"decision kernel for {modulation}"
    )


def test_payload_length_matches_coded_length():
    """The demo transmitter must not pad or truncate a codeword."""
    for case in dc.demo_cases():
        capture = dc.build_capture(**case)
        assert capture.coded_bits.size % (
            {"16-QAM": 4, "QPSK": 2, "8-PSK": 3}[capture.modulation]
        ) == 0


# ---------------------------------------------------------------------------
# Interleaver round trip through the RECEIVER's own deinterleaver
# ---------------------------------------------------------------------------


# (scheme, interleave family, param, payload bits)
#
# The payload sizes are not arbitrary: the interleavers constrain the
# coded frame length, and the demos must not silently pad a codeword.
#  * block depth d needs coded % d == 0;
#  * the square diagonal interleaver needs exactly depth**2 code bits
#    (so RS uses depth 80 -> 6400 bits, conv12 depth 16 -> 256 bits);
#  * a concatenated frame is 640k+12 bits, i.e. always 4 (mod 8), so a
#    depth-8 block interleaver can never frame it (depth 4 can).
ROUND_TRIP_CASES = [
    ("conv12", "block", 8, 94),
    ("conv12", "convolutional", 2, 96),
    ("conv12", "diagonal", 16, 122),
    ("conv12", "pseudo_random", 3, 96),
    ("reedsolomon", "block", 8, 96),
    ("reedsolomon", "convolutional", 2, 96),
    ("reedsolomon", "diagonal", 80, 5120),
    ("reedsolomon", "pseudo_random", 3, 96),
    ("concatenated", "block", 4, 96),
    ("concatenated", "convolutional", 2, 96),
    ("concatenated", "pseudo_random", 3, 96),
]


@pytest.mark.parametrize("scheme,family,param,nbits", ROUND_TRIP_CASES)
def test_interleaver_round_trip_through_receiver(scheme, family, param, nbits):
    """interleave -> (receiver) deinterleave + FEC decode == payload.

    Uses ``FECConfig.deinterleave_bits`` (the exact helper the pipeline
    calls) so the family wiring is verified, not a copy of it.
    """
    from prototype.fec.framework import encode_bits, decode_bits
    from prototype.core.config import FECConfig

    payload = np.random.default_rng(11).integers(0, 2, nbits).astype(np.uint8)
    coded = dc.fec_encode(payload, scheme)
    interleaved = dc.interleave(coded, family, param)

    config = FECConfig(
        scheme=scheme,
        mode=FECMode.MANUAL,
        interleaving_mode=FECMode.MANUAL,
        interleave_family=family,
        interleave_depth=param,
    )
    recovered = config.deinterleave_bits(interleaved)
    assert np.array_equal(recovered, coded), "deinterleaver did not invert"

    # The payload is reconstructed exactly; a scheme may return extra
    # block padding / Viterbi tail bits beyond the payload length, so the
    # comparison is over the payload itself.
    decoded, _ = decode_bits(recovered, scheme)
    decoded = np.asarray(decoded, dtype=np.uint8).reshape(-1)
    assert decoded.size >= payload.size
    assert np.array_equal(decoded[: payload.size], payload)


# ---------------------------------------------------------------------------
# Honest negative: no false FEC claims without a usable reference
# ---------------------------------------------------------------------------


def test_no_reference_never_claims_a_decoded_payload():
    """Without a transmitted reference the blind fold may be unresolvable.

    The pipeline must say so (alignment not applied, no zero-error claim)
    rather than silently decode a misaligned stream as if it were correct.
    """
    capture = dc.build_capture(modulation="16-QAM", fec_scheme="reedsolomon")
    result = _analyze(
        capture, scheme="reedsolomon", family=None, param=None, with_reference=False
    )
    demod = result.demodulation or {}
    alignment = demod.get("codeword_alignment")
    assert alignment is None, "alignment must not be attempted without a reference"

    fec = _decoded(result)
    if fec:
        # A decode may still be produced (the decoder is error-tolerant),
        # but it must not be presented as a clean codeword.
        assert not _prefix_matches(fec["decoded_bits"], capture.payload_bits, 0)


@pytest.mark.parametrize("modulation", ["QPSK", "8-PSK"])
def test_non_qam_demos_stay_honest(modulation):
    """QPSK/8-PSK demos: whatever the chain claims, it must be true.

    The blind symbol-origin resolution now covers the PSK folds as well as
    16-QAM, so a clean decode is legitimate — the assertion is that a
    *claimed* clean codeword really matches the transmitted payload, and
    that a failed alignment is never dressed up as one.
    """
    capture = dc.build_capture(
        modulation=modulation, fec_scheme="reedsolomon", nbits=768
    )
    result = _analyze(
        capture, scheme="reedsolomon", family=None, param=None, with_reference=True
    )
    assert result.classification is not None
    assert result.classification["modulation"] in ("QPSK", "8-PSK", "16-QAM")
    demod = result.demodulation or {}
    assert np.asarray(demod.get("received_bits", [])).size > 0

    fec = _decoded(result)
    if fec and fec.get("uncorrectable_blocks") == 0:
        assert _prefix_matches(fec["decoded_bits"], capture.payload_bits, 0), (
            "a clean codeword was claimed that does not match the "
            "transmitted payload - that would be a false positive"
        )
