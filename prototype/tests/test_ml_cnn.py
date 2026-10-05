"""Tests for the ML subsystem (ml/ package).

The CNN is optional evidence: every test here must pass without a
deep-learning framework, using only NumPy and the packaged artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prototype.ml.cnn import (
    FRAME_LENGTH,
    ML_VALIDATION_FLOOR,
    NUM_CLASSES,
    ModulationCNN,
    extract_frames,
    get_engine,
    normalize_frames,
    predict_modulation,
)


# ------------------------------------------------------------------
# Fixture location (repository-independent)
# ------------------------------------------------------------------

def _resolve_repo_root() -> Path:
    """Return this repository's source root (the directory that
    contains ``prototype/``).

    The captured signals live at the repository root. This helper
    always resolves the root no matter which directory the tests were
    launched from (developer: ``pytest`` from the repo root or from
    ``prototype/``; teammate: fresh clone; CI: container checkout).
    """
    candidate = Path(__file__).resolve().parent.parent.parent
    # Quite often the repository root is the current working directory
    # (e.g. ``pytest`` run from the repo root), in which case the
    # container layout matches the developer layout.
    if (candidate / "prototype" / "ml" / "cnn.py").is_file():
        return candidate
    raise RuntimeError(
        "Could not resolve the repository root from the test file "
        "location. Expected <repo>/prototype/ml/cnn.py. "
        f"Checked {candidate}."
    )


def _find_gui_qam16_wav() -> Path:
    """Locate the synthetic capture used by the ML stage tests.

    Layout of the shipped checkout:

        <repo>/prototype/
        ├── gui_qam16.wav
        └── tests/
             └── test_ml_cnn.py

    The path is resolved relative to the repository root (one level
    above ``tests/``) instead of using the current working directory,
    so the same test passes for a developer, a teammate cloning the
    repository, or in CI.
    """
    repo_root = _resolve_repo_root()
    for candidate in (repo_root / "prototype" / "gui_qam16.wav",):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "gui_qam16.wav not found. Expected a copy of the synthetic "
        f"capture at {repo_root / 'prototype'}. "
        "This fixture is required by the ML stage tests."
    )


# --------------------------------------------------------------
# Artifact + engine
# --------------------------------------------------------------

def _any_artifact() -> object:
    from prototype.ml import cnn as cnn_module

    for candidate in (cnn_module._TRAINED, cnn_module._DEFAULT):
        if candidate.is_file():
            return candidate
    pytest.skip("No ML artifact packaged")


def test_engine_loads_and_validates_config():
    path = _any_artifact()
    engine = ModulationCNN(path)

    assert len(engine.labels) == NUM_CLASSES
    # The input length is a property of the artifact, not a module
    # constant (ML v3 ships 1024-sample artifacts).  What must hold is
    # that the declaration and the geometry read back from the weights
    # agree — resolve_architecture raises otherwise.
    assert engine.config["input_shape"] == [engine.frame_length, 2]
    assert engine.architecture["frame_length"] == engine.frame_length
    assert engine.frame_length >= FRAME_LENGTH
    assert engine.frame_length % 2 == 0
    assert engine.config["activation_head"] == "softmax"
    assert engine.config["num_classes"] == NUM_CLASSES


def test_engine_tensor_shapes_match_architecture():
    path = _any_artifact()
    engine = ModulationCNN(path)
    t = engine._tensors
    blocks = engine.blocks

    assert 1 <= len(blocks) <= 4
    assert blocks[0].in_channels == 2
    for index, block in enumerate(blocks):
        assert t[f"conv{index}.kernel"].shape == (
            block.kernel, block.in_channels, block.out_channels
        )
        assert t[f"conv{index}.bias"].shape == (block.out_channels,)
        if index:
            assert block.in_channels == blocks[index - 1].out_channels
        assert t[f"bn{index}.gamma"].shape[0] == block.out_channels
        for name in ("beta", "mean", "variance"):
            assert t[f"bn{index}.{name}"].shape == (block.out_channels,)
        assert bool(np.all(t[f"bn{index}.variance"] > 0.0))

    # The classifier must consume exactly the head width the pooling and
    # convolution stack produces.  Getting this wrong is how a flatten
    # head silently reads the wrong axis order.
    assert t["dense1.kernel"].shape[1] == NUM_CLASSES
    if engine.has_hidden_dense:
        assert t["dense0.kernel"].shape[0] == engine.architecture["head_input"]
        assert t["dense1.kernel"].shape[0] == t["dense0.kernel"].shape[1]
    else:
        assert t["dense1.kernel"].shape[0] == engine.architecture["head_input"]


def test_softmax_output_is_normalized():
    path = _any_artifact()
    engine = ModulationCNN(path)

    rng = np.random.default_rng(0)
    samples = rng.standard_normal(16384) + 0j
    frames = normalize_frames(
        extract_frames(samples, frame_length=engine.frame_length)
    )
    scores = engine.predict_frames(frames)

    assert frames.shape[1] == engine.frame_length
    assert frames.shape[0] > 1
    assert scores.shape == (frames.shape[0], NUM_CLASSES)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    assert bool(np.all(scores >= 0.0))


def test_shipped_artifact_validation_claim_is_consistent():
    """The packaged model must not claim more than the gate allows."""

    engine = get_engine()
    if engine is None or not engine.trained:
        pytest.skip("No trained ML artifact packaged")

    accuracy = engine.validation_accuracy
    assert accuracy is not None
    assert 0.0 <= accuracy <= 1.0
    assert engine.validated == (accuracy >= ML_VALIDATION_FLOOR)
    assert engine.status


# --------------------------------------------------------------
# Frame extraction + normalization
# --------------------------------------------------------------

def test_extract_frames_shapes_and_coverage():
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(8192) + 1j * rng.standard_normal(8192)

    frames = extract_frames(samples, max_frames=4)
    assert frames.shape == (4, FRAME_LENGTH, 2)
    assert frames.dtype == np.float32

    # Largest-stride coverage: first frame from the start, last from
    # the end of the capture.
    np.testing.assert_allclose(
        frames[0, :, 0], samples[:FRAME_LENGTH].real, atol=1e-6
    )
    np.testing.assert_allclose(
        frames[-1, :, 0], samples[-FRAME_LENGTH:].real, atol=1e-6
    )


def test_extract_frames_rejects_short_capture():
    with pytest.raises(ValueError):
        extract_frames(np.zeros(100, dtype=complex))


def test_normalize_frames_is_unit_rms_and_amplitude_invariant():
    rng = np.random.default_rng(2)
    frames = rng.standard_normal((3, FRAME_LENGTH, 2)).astype(np.float32)

    scaled = normalize_frames(frames * 100.0)
    reference = normalize_frames(frames)

    np.testing.assert_allclose(scaled, reference, rtol=1e-4)
    rms = np.sqrt(np.mean(scaled.astype(np.float64) ** 2, axis=(1, 2)))
    np.testing.assert_allclose(rms, 1.0, rtol=1e-4)


# --------------------------------------------------------------
# Prediction wrapper (uses the packaged artifact)
# --------------------------------------------------------------

def test_predict_modulation_payload_contract():
    rng = np.random.default_rng(3)
    samples = rng.standard_normal(8192) + 1j * rng.standard_normal(8192)

    result = predict_modulation(samples)

    if result is None:
        pytest.skip("No ML artifact packaged")

    assert set(result) >= {
        "predicted_class", "confidence", "num_frames",
        "top3", "labels_source", "artifact",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top3"]) == 3
    # Scores must be sorted descending.
    scores = [entry["score"] for entry in result["top3"]]
    assert scores == sorted(scores, reverse=True)
    # Neutral labels (untrained artifact) carry their honesty note.
    assert "class" in result["top3"][0]


def test_predict_modulation_reports_short_capture():
    """A capture shorter than one frame is a real signal, not a silent skip.

    ``predict_modulation`` returns a detailed status dict so the caller can
    distinguish "nothing available at all" from "capture too short to
    classify". The pipeline's ML toggle must not silently drop such frames.
    """
    engine = get_engine()
    if engine is None:
        pytest.skip("No ML artifact packaged")

    # Shorter than the smallest frame any shipped artifact has used, so
    # the rejection cannot be an artefact of the current frame length.
    result = predict_modulation(np.zeros(64, dtype=complex))

    assert result is not None
    assert result["num_frames"] == 0
    assert result["predicted_class"] is None
    assert result["confidence"] == 0.0
    assert result["presented_as"] == "evidence"
    assert result["frame_length"] == engine.frame_length
    # The payload must say the stage was *skipped*, not imply the model
    # produced a verdict, and it must name the artifact's frame length.
    assert result["inference_ran"] is False
    assert result["skip_reason"] == "insufficient_input"
    assert "skipped inference" in result["note"]
    assert (
        f"shorter than one {engine.frame_length}-sample frame"
        in result["note"]
    )


def test_get_engine_is_cached():
    first = get_engine()
    second = get_engine()
    if first is None:
        pytest.skip("No ML artifact packaged")
    assert first is second


# --------------------------------------------------------------
# Labels override mechanism
# --------------------------------------------------------------

def test_labels_json_override(tmp_path):
    path = _any_artifact()
    labels = tmp_path / "model.labels.json"
    labels.write_text(json.dumps([f"M{i:02d}" for i in range(16)]),
                      encoding="utf-8")

    engine = ModulationCNN(path, labels_json=labels)
    assert engine.labels[:2] == ["M00", "M01"]
    assert "labels.json" in engine.labels_source


def test_labels_json_wrong_length_is_rejected(tmp_path):
    path = _any_artifact()
    labels = tmp_path / "bad.labels.json"
    labels.write_text(json.dumps(["A", "B"]), encoding="utf-8")

    with pytest.raises(ValueError):
        ModulationCNN(path, labels_json=labels)


# --------------------------------------------------------------
# Training path (NumPy backprop, gradient-checked)
# --------------------------------------------------------------

def test_gradient_check_passes():
    from prototype.ml.train import gradient_check

    error = gradient_check()
    assert error < 1e-5, f"backprop gradient error {error}"


def test_training_reduces_loss_on_tiny_run():
    from prototype.ml.dataset import CLASS_NAMES
    from prototype.ml.train import Model, softmax_xentropy_forward, train

    model, summary = train(
        frames_per_class=2,
        epochs=2,
        batch_size=16,
        seed=5,
        log=lambda *_: None,
    )

    assert len(summary["history"]) == 2
    assert summary["history"][-1]["loss"] < summary["history"][0]["loss"]
    assert set(summary["per_class_val_accuracy"]) == set(CLASS_NAMES)


def test_untrained_artifact_declares_itself():
    """The shipped conversion must not pretend to be trained."""
    path = _any_artifact()
    engine = ModulationCNN(path)
    trained_flag = engine.config.get("trained", False)
    if not trained_flag:
        assert "untrained" in engine.config.get("note", "").lower() or (
            "not embedded" in engine.config.get("note", "").lower()
        )


# --------------------------------------------------------------
# Dataset generator
# --------------------------------------------------------------

def test_dataset_generator_shapes_and_labels():
    from prototype.ml.dataset import CLASS_NAMES, build_dataset

    frames, labels = build_dataset(frames_per_class=1, seed=9)
    assert frames.shape == (len(CLASS_NAMES), FRAME_LENGTH, 2)
    assert sorted(labels.tolist()) == list(range(len(CLASS_NAMES)))
    assert frames.dtype == np.float32
    assert bool(np.all(np.isfinite(frames)))


def _load_gui_qam16_capture():
    """Load the synthetic capture used by the ML stage tests.

    The path is resolved relative to the repository root, not the
    current working directory, so the test behaves identically when
    run from the repository root, from inside ``prototype/``, from a
    teammate's clone, or in CI.
    """
    from prototype.core.loader import load_wav

    samples, sample_rate = load_wav(str(_find_gui_qam16_wav()))
    return samples, sample_rate


def test_pipeline_ml_stage_off_by_default():
    from dataclasses import replace

    from prototype.core.config import processing_mode_config
    from prototype.pipeline import analyze_samples

    config = processing_mode_config("balanced")
    assert config.ml.enabled is False

    samples, sample_rate = _load_gui_qam16_capture()
    result = analyze_samples(samples, sample_rate, config=config)
    assert result.ml is None


def test_pipeline_ml_stage_enabled_payload():
    from dataclasses import replace

    from prototype.core.config import processing_mode_config
    from prototype.pipeline import analyze_samples

    samples, sample_rate = _load_gui_qam16_capture()
    base = processing_mode_config("balanced")
    config = replace(base, ml=replace(base.ml, enabled=True))

    result = analyze_samples(samples, sample_rate, config=config)

    if result.ml is None:
        pytest.skip("No ML artifact packaged")

    assert result.ml["num_frames"] >= 1
    steps = {
        step["name"]: step["status"]
        for step in result.provenance.get("steps", [])
    }
    assert steps.get("ml_classification") == "ok"

    payload = result.to_dict()
    assert payload["ml"]["predicted_class"] == result.ml["predicted_class"]
