"""Reference-free read-outs: quality estimate, interleaving verdict, ML row.

Three operator-facing gaps are pinned here:

1.  Without a transmitted reference the BER row used to read only "no
    reference loaded"; the receiver now also reports an EVM-based
    estimate (clearly labelled as an estimate, never as a measurement).
2.  A capture whose demodulated stream already matches the transmitted
    reference reported ``UNRESOLVED`` for interleaving even though no
    deinterleaving step is required; it now reports ``NONE`` with the
    evidence that produced it.
3.  When the CNN's argmax may not be presented as the ML verdict — an
    unvalidated artifact, or a capture whose scores miss the per-capture
    floors — the ML row shows the deterministic classification
    (mirroring) instead of the network's argmax, while the network's own
    class stays in the payload.  A validated artifact that clears the
    floors keeps its own verdict instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.tests import demo_captures as dc


def _capture(**overrides):
    case = {
        "modulation": "16-QAM",
        "fec_scheme": None,
        "interleave_family": None,
        "interleave_param": None,
    }
    case.update(overrides)
    return dc.build_capture(**case)


def test_quality_estimate_is_reported_without_a_reference():
    from prototype.pipeline import analyze_samples

    capture = _capture()
    result = analyze_samples(capture.samples, capture.sample_rate)

    quality = (result.demodulation or {}).get("quality_estimate") or {}
    assert quality.get("method") == "evm_estimate"
    assert 0.0 < float(quality["evm"]) < 1.0
    assert float(quality["snr_db"]) > 0.0
    assert 0.0 <= float(quality["ber_estimate"]) <= 0.5
    assert "not a measured BER" in quality["note"]
    # The BER itself stays unmeasured: honesty is preserved.
    assert result.ber is None


def test_quality_estimate_scales_with_noise():
    """A noisier capture must report a worse SNR and a higher BER estimate."""
    from prototype.pipeline import analyze_samples

    clean = _capture(noise_amplitude=0.002)
    noisy = _capture(noise_amplitude=0.12)

    clean_q = analyze_samples(
        clean.samples, clean.sample_rate
    ).demodulation["quality_estimate"]
    noisy_q = analyze_samples(
        noisy.samples, noisy.sample_rate
    ).demodulation["quality_estimate"]

    assert noisy_q["evm"] > clean_q["evm"]
    assert noisy_q["snr_db"] < clean_q["snr_db"]
    assert noisy_q["ber_estimate"] >= clean_q["ber_estimate"]


def test_interleaving_reports_none_when_no_deinterleaving_is_required():
    from prototype.pipeline import analyze_samples

    capture = _capture()
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") == "NONE"
    assert interleaving.get("evidence", {}).get("mode") == "reference_agreement"
    assert interleaving.get("family") is None
    # And the BER is measured, because the reference was supplied.
    assert result.ber is not None
    assert result.ber["ber"] <= 0.01


def test_interleaving_stays_unresolved_without_any_reference():
    from prototype.pipeline import analyze_samples

    capture = _capture()
    result = analyze_samples(capture.samples, capture.sample_rate)

    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") in ("UNRESOLVED", "UNKNOWN")
    assert interleaving.get("best_depth") is None


def test_coded_capture_still_resolves_its_interleaver():
    """The new NONE shortcut must not pre-empt the joint FEC search."""
    from prototype.pipeline import analyze_samples

    capture = _capture(
        fec_scheme="reedsolomon", interleave_family="block", interleave_param=8
    )
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") == "AUTO_DETECTED"
    assert interleaving.get("family") == "block"
    assert interleaving.get("best_depth") == 8


def test_uncoded_qpsk_resolves_and_measures_its_ber():
    """A QPSK demo capture is bit-exact against its own reference.

    The QPSK lattice fold, its axis-aligned twin and the symbol origin are
    all searched, so an uncoded QPSK capture reports a measured BER of 0
    rather than a plateau near 0.25 (the plateau came from the demo
    transmitter's quadrant/bit table, not from synchronization).
    """
    from prototype.pipeline import analyze_samples

    capture = _capture(modulation="QPSK")
    result = analyze_samples(
        capture.samples, capture.sample_rate, reference_bits=capture.coded_bits
    )

    assert result.classification["modulation"] == "QPSK"
    assert result.ber is not None and result.ber["status"] == "measured"
    assert result.ber["ber"] == 0.0
    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") == "NONE"


def test_corrective_resync_replaces_a_wrong_coarse_label():
    """The fine classifier can overrule the label the sync was chosen for.

    A QPSK capture is labelled 16-QAM by the coarse (waveform) stage, so
    the first synchronization optimizes a 16-QAM lattice; the corrective
    re-sync runs the QPSK chain, and the classification records it.
    """
    from prototype.pipeline import analyze_samples

    capture = _capture(modulation="QPSK")
    result = analyze_samples(capture.samples, capture.sample_rate)

    classification = result.classification or {}
    assert classification.get("modulation") == "QPSK"
    assert classification["coarse_result"]["modulation"] == "16-QAM"
    assert classification.get("stage") == "fine_refined"

    refinement = (result.synchronization or {}).get("refinement") or {}
    assert refinement.get("accepted") is True
    assert refinement.get("fine_label_before") == "QPSK"
    assert refinement.get("method") in ("lattice_fit", "mth_power")


def test_ml_display_mirrors_the_dsp_classification():
    from dataclasses import replace

    from prototype.core.config import processing_mode_config
    from prototype.pipeline import analyze_samples

    capture = _capture()
    base = processing_mode_config("balanced")
    config = replace(base, ml=replace(base.ml, enabled=True))
    result = analyze_samples(capture.samples, capture.sample_rate, config=config)

    if result.ml is None:
        pytest.skip("No ML artifact packaged")

    if result.ml.get("validated") and result.ml.get("presented_as") == "prediction":
        pytest.skip("Packaged CNN is validated; it keeps its own verdict")

    assert result.ml.get("display_class") == "16-QAM"
    assert result.ml.get("display_source") == "dsp_mirror"
    assert result.ml.get("mirrored_from_dsp") is True
    # Honesty is preserved in the payload: the network's own class survives.
    assert result.ml.get("ml_raw_class")
    assert result.ml.get("raw_note")

    # And the deterministic result is not overridden by the mirror.
    assert (result.classification or {}).get("modulation") == "16-QAM"
    assert (result.fusion or {}).get("final_modulation") == "16-QAM"
