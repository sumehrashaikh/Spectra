"""ML prediction honesty: shared label vocabulary + validated-only labels.

Three real defects are pinned here:

1.  The CNN labels come from ``ml.dataset`` ("16QAM", "8PSK") while the
    deterministic classifier spells them "16-QAM" / "8-PSK".  Without a
    shared vocabulary every ML QAM/PSK prediction was rewritten to
    "Unknown" inside the fusion stage and could never be compared.
2.  An artifact whose holdout accuracy is below the ``ML_VALIDATION_FLOOR``
    must report its argmax as *evidence*, never as a prediction the GUI
    shows as fact.  A validated artifact keeps its own verdict, but only
    where this capture's scores also clear the per-capture confidence and
    agreement floors — so both levels of the gate are exercised here.
3.  A capture too short to form one artifact frame is reported as a
    *skipped* stage (``status="skipped"``, ``skip_reason``), not as a
    successful classification, and the note must say why instead of
    calling a validated artifact "unvalidated".
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.ml.cnn import ML_VALIDATION_FLOOR, predict_modulation
from prototype.ml.fusion import (
    FusionConfig,
    canonical_modulation,
    fuse_classification,
    is_comparable,
)


def test_canonical_labels_unify_both_vocabularies():
    assert canonical_modulation("16QAM") == "16-QAM"
    assert canonical_modulation("16-QAM") == "16-QAM"
    assert canonical_modulation("8PSK") == "8-PSK"
    assert canonical_modulation("8-PSK") == "8-PSK"
    assert canonical_modulation("qpsk") == "QPSK"
    assert canonical_modulation("BFSK") == "BFSK"
    assert canonical_modulation("2FSK") == "BFSK"
    # Out-of-vocabulary classes are returned unchanged, not squashed to
    # "Unknown": the caller must be able to tell them apart.
    assert canonical_modulation("AM-DSB") == "AM-DSB"
    assert canonical_modulation("Noise") == "Noise"
    assert canonical_modulation(None) == "Unknown"
    assert canonical_modulation("") == "Unknown"


def test_comparability_matches_the_dsp_vocabulary():
    assert is_comparable("16QAM") is True
    assert is_comparable("8PSK") is True
    assert is_comparable("AM-DSB") is False


def _ml(label, score=0.9, agreement=1.0, validated=True):
    return {
        "predicted_class": label,
        "confidence": score,
        "frame_agreement": agreement,
        "validated": validated,
        "presented_as": "prediction" if validated else "evidence",
    }


def test_ml_qam_label_is_compared_with_dsp_instead_of_unknown():
    record = fuse_classification(
        dsp_modulation="16-QAM",
        ml_prediction=_ml("16QAM"),
        ml_active=True,
        fusion=FusionConfig(),
    )

    assert record["ml_modulation"] == "16-QAM"
    assert record["ml_comparable"] is True
    assert record["disagreement"] is False
    assert record["final_modulation"] == "16-QAM"


def test_ml_out_of_vocabulary_class_is_evidence_not_a_rival_answer():
    record = fuse_classification(
        dsp_modulation="QPSK",
        ml_prediction=_ml("AM-DSB", score=0.9),
        ml_active=True,
        fusion=FusionConfig(),
    )

    assert record["ml_modulation"] == "AM-DSB"
    assert record["ml_comparable"] is False
    assert record["final_modulation"] == "QPSK"
    assert "outside the deterministic" in (record["ml_warn"] or "")


def test_ml_agreement_is_recorded_when_labels_match():
    record = fuse_classification(
        dsp_modulation="QPSK",
        ml_prediction=_ml("QPSK", score=0.8),
        ml_active=True,
        fusion=FusionConfig(),
    )

    assert record["disagreement"] is False
    assert record["ml_raw_score"] == pytest.approx(0.8)


def test_predict_modulation_reports_evidence_not_a_label_when_unvalidated():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(8192) + 1j * rng.standard_normal(8192)

    result = predict_modulation(samples)
    if result is None:
        pytest.skip("No ML artifact packaged")

    assert "presented_as" in result
    assert "validated" in result
    assert "predicted_class_canonical" in result
    assert "comparable" in result

    if not result["validated"]:
        # The shipped artifact must not present its argmax as a label.
        assert result["presented_as"] == "evidence"
        assert result["confidence"] < 1.0

    if result["validated"]:
        assert (
            float(result.get("validation_accuracy") or 0.0)
            >= ML_VALIDATION_FLOOR
        )


def test_predict_modulation_short_capture_keeps_the_gate_fields():
    result = predict_modulation(np.zeros(64, dtype=complex))

    assert result is not None
    assert result["predicted_class"] is None
    assert result["presented_as"] == "evidence"
    assert result["validated"] in (True, False)


# ------------------------------------------------------------------
# A skipped stage is not a classification, and a validated model is
# never described as "unvalidated" just because a capture was too short.
# ------------------------------------------------------------------

def _mirror(payload, modulation="16-QAM", confidence=93.1):
    from prototype.pipeline import _mirror_ml_display

    return _mirror_ml_display(
        payload, {"modulation": modulation, "confidence": confidence}
    )


def test_short_capture_mirror_says_inference_was_skipped():
    """The core regression: too short is a *skip*, not an invalid model."""

    reason = (
        "ML stage skipped inference: the capture is shorter than one "
        "1024-sample frame (the artifact's input length)."
    )
    payload = {
        "num_frames": 0,
        "predicted_class": None,
        "confidence": 0.0,
        "validated": True,
        "presented_as": "evidence",
        "frame_length": 1024,
        "note": reason,
    }
    mirror = _mirror(payload)

    assert mirror["inference_ran"] is False
    assert mirror["display_class"] == "16-QAM"
    assert "skipped inference" in mirror["note"]
    assert "1024" in mirror["note"]
    assert "deterministic DSP classification" in mirror["note"]
    # A validated artifact must not be labelled unvalidated here.
    assert "unvalidated" not in mirror["note"].lower()
    # The stage's own reason survives verbatim for audit.
    assert mirror["raw_note"] == reason
    assert mirror["ml_raw_class"] is None


def test_mirror_note_never_calls_a_validated_model_unvalidated():
    """Inference ran and the artifact is validated, but the scores missed
    the per-capture floors: a per-capture gate, not a training one."""

    payload = {
        "num_frames": 8,
        "predicted_class": "64QAM",
        "confidence": 0.46,
        "validated": True,
        "presented_as": "evidence",
        "frame_length": 1024,
        "note": "CNN scores are uncalibrated …",
    }
    mirror = _mirror(payload)

    assert mirror["inference_ran"] is True
    assert "unvalidated" not in mirror["note"].lower()
    assert "confidence/agreement floors" in mirror["note"]
    # The network's own answer still survives for audit.
    assert mirror["ml_raw_class"] == "64QAM"


def test_mirror_note_keeps_unvalidated_wording_when_it_is_true():
    """The wording is only banned where it is false."""

    payload = {
        "num_frames": 8,
        "predicted_class": "QPSK",
        "confidence": 0.9,
        "validated": False,
        "presented_as": "evidence",
        "frame_length": 1024,
        "note": "…",
    }
    mirror = _mirror(payload)

    assert mirror["inference_ran"] is True
    assert "unvalidated" in mirror["note"].lower()


def test_validated_model_clearing_the_floors_keeps_its_own_verdict():
    payload = {
        "num_frames": 8,
        "predicted_class": "QPSK",
        "confidence": 0.6,
        "validated": True,
        "presented_as": "prediction",
        "frame_length": 1024,
        "note": "…",
    }
    assert _mirror(payload) is None


def test_pipeline_status_distinguishes_skipped_from_inference():
    """End to end through the real pipeline: a capture shorter than one
    artifact frame must not be reported as a successful classification."""

    from dataclasses import replace

    from prototype.core.config import processing_mode_config
    from prototype.pipeline import analyze_samples
    from prototype.tests import demo_captures as dc

    capture = dc.build_capture(
        modulation="16-QAM", fec_scheme=None,
        interleave_family=None, interleave_param=None,
    )
    base = processing_mode_config("balanced")
    config = replace(base, ml=replace(base.ml, enabled=True))

    short = analyze_samples(
        capture.samples[:700], float(capture.sample_rate), config=config
    )
    if short.ml is None:
        pytest.skip("No ML artifact packaged")

    ml = short.ml
    assert ml["num_frames"] == 0
    assert ml["predicted_class"] is None
    assert ml["presented_as"] == "evidence"
    assert ml["status"] == "skipped"
    assert ml["inference_ran"] is False
    assert ml["skip_reason"] == "insufficient_input"
    # Whether the mirror applied depends on the DSP stage still finding a
    # classification in 700 samples; either way the note must be truthful.
    if ml["validated"]:
        assert "unvalidated" not in ml["note"].lower()
    # Both short-capture paths (mirrored and unmapped) must say the stage
    # was skipped, and must name the artifact's frame length.
    assert "skipped inference" in ml["note"]
    assert str(ml["frame_length"]) in ml["note"]

    # A long-enough capture must still report a real inference.
    full = analyze_samples(
        capture.samples, float(capture.sample_rate), config=config
    )
    assert full.ml["status"] == "ok"
    assert full.ml["inference_ran"] is True
    assert "skip_reason" not in full.ml
    assert full.ml["num_frames"] > 0
