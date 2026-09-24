"""Tests for the ML subsystem (ml/ package).

The CNN is optional evidence: every test here must pass without a
deep-learning framework, using only NumPy and the packaged artifacts.
"""

import json

import numpy as np
import pytest

from prototype.ml.cnn import (
    FRAME_LENGTH,
    NUM_CLASSES,
    ModulationCNN,
    extract_frames,
    get_engine,
    normalize_frames,
    predict_modulation,
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
    assert engine.config["input_shape"] == [FRAME_LENGTH, 2]
    assert engine.config["activation_head"] == "softmax"
    # The config embedded in the artifact must match the runtime's
    # architecture constants.
    assert engine.config["num_classes"] == NUM_CLASSES


def test_engine_tensor_shapes_match_architecture():
    path = _any_artifact()
    engine = ModulationCNN(path)
    t = engine._tensors

    assert t["conv0.kernel"].shape == (7, 2, 64)
    assert t["conv1.kernel"].shape == (5, 64, 128)
    assert t["conv2.kernel"].shape == (3, 128, 256)
    assert t["dense0.kernel"].shape[1] == 256
    assert t["dense1.kernel"].shape == (256, NUM_CLASSES)

    for idx in range(3):
        channels = t[f"bn{idx}.gamma"].shape[0]
        assert t[f"bn{idx}.beta"].shape == (channels,)
        assert t[f"bn{idx}.mean"].shape == (channels,)
        assert t[f"bn{idx}.variance"].shape == (channels,)
        assert bool(np.all(t[f"bn{idx}.variance"] > 0.0))


def test_softmax_output_is_normalized():
    path = _any_artifact()
    engine = ModulationCNN(path)

    rng = np.random.default_rng(0)
    frames = normalize_frames(extract_frames(rng.standard_normal(4096) + 0j))
    scores = engine.predict_frames(frames)

    assert scores.shape == (frames.shape[0], NUM_CLASSES)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    assert bool(np.all(scores >= 0.0))


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
    result = predict_modulation(np.zeros(64, dtype=complex))

    assert result is not None
    assert result["num_frames"] == 0
    assert result["predicted_class"] is None
    assert result["confidence"] == 0.0
    assert "Capture shorter than one 512-sample frame" in result["note"]


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


def test_pipeline_ml_stage_off_by_default():
    from dataclasses import replace

    from prototype.core.config import processing_mode_config

    config = processing_mode_config("balanced")
    assert config.ml.enabled is False

    from prototype.core.loader import load_wav
    from prototype.pipeline import analyze_samples

    samples, sample_rate = load_wav("gui_qam16.wav")
    result = analyze_samples(samples, sample_rate, config=config)
    assert result.ml is None


def test_pipeline_ml_stage_enabled_payload():
    from dataclasses import replace

    from prototype.core.config import processing_mode_config
    from prototype.core.loader import load_wav
    from prototype.pipeline import analyze_samples

    samples, sample_rate = load_wav("gui_qam16.wav")
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
