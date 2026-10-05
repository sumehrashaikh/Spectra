"""Regression tests for the ML training/validation upgrade.

These pin the properties that the previous training path did not have:

* the split is stratified *and* leaks no realization across folds,
* the training report carries per-class accuracy, a confusion matrix and a
  per-SNR breakdown,
* the exported artifact records its own architecture and provenance,
* the NumPy runtime reproduces the training forward pass numerically
  (parity), which is what makes the exported artifact trustworthy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


# --------------------------------------------------------------
# dataset / split
# --------------------------------------------------------------

def test_default_frame_config_is_unchanged():
    """The historical generator must be reproduced exactly by default."""

    from prototype.ml.dataset import (
        _random_frame_config,
        _random_frame_config_v2,
    )

    first = _random_frame_config(np.random.default_rng(11))
    second = _random_frame_config_v2(np.random.default_rng(11))
    assert first == second
    assert second.timing_offset_samples == 0
    assert second.multipath_taps is None


def test_channel_variation_adds_timing_and_optional_multipath():
    from prototype.ml.dataset import _random_frame_config_v2

    timing = [
        _random_frame_config_v2(np.random.default_rng(seed),
                                channel_variation=True)
        for seed in range(20)
    ]
    assert any(config.timing_offset_samples > 0 for config in timing)
    assert all(config.multipath_taps is None for config in timing)

    with_multipath = [
        _random_frame_config_v2(np.random.default_rng(seed),
                                multipath_fraction=1.0)
        for seed in range(20)
    ]
    assert all(config.multipath_taps for config in with_multipath)
    # Gains must decay, matching a plausible multipath profile.
    for config in with_multipath:
        gains = [gain for _delay, gain in config.multipath_taps]
        assert all(gain < 1.0 for gain in gains)
        assert gains == sorted(gains, reverse=True)


def test_stratified_split_is_leakage_free_and_covers_everything():
    from prototype.ml.split import stratified_realization_split

    # 4 frames per realization, 6 classes, deliberately interleaved so a
    # naive frame-level split would leak.
    labels, realizations = [], []
    for klass in range(6):
        for realization in range(10):
            for _ in range(4):
                labels.append(klass)
                realizations.append(klass * 100 + realization)

    labels = np.asarray(labels)
    realizations = np.asarray(realizations)

    folds = stratified_realization_split(
        labels, realizations, fractions=(0.7, 0.15, 0.15), seed=3
    )

    train, validation, test = (
        folds["train"], folds["validation"], folds["test"]
    )

    # Coverage: every frame lands in exactly one fold.
    everything = np.concatenate([train, validation, test])
    assert everything.size == labels.size
    assert np.unique(everything).size == labels.size

    # No leakage: no realization appears in two folds.
    train_realizations = set(realizations[train].tolist())
    val_realizations = set(realizations[validation].tolist())
    test_realizations = set(realizations[test].tolist())
    assert not (train_realizations & val_realizations)
    assert not (train_realizations & test_realizations)
    assert not (val_realizations & test_realizations)

    # A realization is never split across folds.
    for name, index in folds.items():
        grouped: dict[int, set[int]] = {}
        for realization, label in zip(
            realizations[index].tolist(), labels[index].tolist()
        ):
            grouped.setdefault(realization, set()).add(label)
        assert all(len(labels_) == 1 for labels_ in grouped.values()), name

    # Stratified: every class appears in every fold.
    for name, index in folds.items():
        assert set(labels[index].tolist()) == set(range(6)), name


# --------------------------------------------------------------
# training report
# --------------------------------------------------------------

def test_training_report_has_confusion_matrix_and_per_snr():
    from prototype.ml.train import train

    _model, summary = train(
        frames_per_class=4,
        epochs=2,
        batch_size=16,
        seed=5,
        val_fraction=0.25,
        test_fraction=0.25,
        dtype="float32",
        channels=(8, 16, 32),
        dense_units=16,
        log=lambda *_: None,
    )

    assert summary["train_realizations"] > 0
    assert summary["best_epoch"] >= 1
    assert "test_metrics" in summary

    for key in ("val_metrics", "test_metrics"):
        report = summary[key]
        assert report is not None, key
        matrix = np.asarray(report["confusion_matrix"])
        assert matrix.shape == (16, 16)
        assert matrix.sum() == report["samples"]
        assert set(report["per_class_accuracy"]) == set(
            report["confusion_matrix_labels"]
        )
        assert report["per_snr_accuracy"]


def test_saved_artifact_declares_architecture_and_provenance(tmp_path):
    from prototype.ml.cnn import ModulationCNN
    from prototype.ml.train import save_artifact, train

    model, summary = train(
        frames_per_class=2,
        epochs=1,
        batch_size=16,
        seed=1,
        dtype="float32",
        channels=(8, 16, 32),
        dense_units=16,
        log=lambda *_: None,
    )
    artifact = tmp_path / "candidate.npz"
    save_artifact(model, summary, artifact)

    engine = ModulationCNN(artifact)
    architecture = engine.config["architecture"]
    assert architecture["blocks"][0]["conv_filters"] == 8
    assert architecture["blocks"][2]["conv_filters"] == 32
    assert architecture["head"][-1]["units"] == 16

    provenance = engine.config["provenance"]
    assert provenance["external_dataset_used"] is False
    assert "stratified" in provenance["split"]
    assert len(provenance["dataset_class_mapping"]) == 16
    assert engine.config["training"]["seed"] == 1


# --------------------------------------------------------------
# NumPy runtime parity
# --------------------------------------------------------------

def test_numpy_runtime_matches_training_forward_pass(tmp_path):
    """The exported artifact must reproduce training-time predictions."""

    from prototype.ml.train import save_artifact, train
    from prototype.ml.validation import parity_check

    model, summary = train(
        frames_per_class=2,
        epochs=1,
        batch_size=16,
        seed=2,
        dtype="float32",
        channels=(8, 16, 32),
        dense_units=16,
        log=lambda *_: None,
    )
    artifact = tmp_path / "parity.npz"
    save_artifact(model, summary, artifact)

    rng = np.random.default_rng(0)
    frames = rng.standard_normal((12, 512, 2)).astype(np.float32) * 0.3

    report = parity_check(artifact, frames)
    assert report["frames_compared"] == 12
    assert report["argmax_matches"] == 12
    assert report["max_abs_softmax_difference"] < 1e-4
    assert report["passed"] is True


def test_validate_artifact_reports_honesty_fields(tmp_path):
    from prototype.ml.cnn import ML_VALIDATION_FLOOR
    from prototype.ml.train import save_artifact, train
    from prototype.ml.validation import (
        render_validation_summary,
        validate_artifact,
    )

    model, summary = train(
        frames_per_class=3,
        epochs=2,
        batch_size=16,
        seed=4,
        dtype="float32",
        channels=(8, 16, 32),
        dense_units=16,
        log=lambda *_: None,
    )
    artifact = tmp_path / "validate.npz"
    save_artifact(model, summary, artifact)

    report = validate_artifact(
        artifact, frames_per_class=2, seed=99, parity_frames=8
    )

    assert report["parity"]["passed"] is True
    honesty = report["honesty"]
    assert honesty["validation_floor"] == ML_VALIDATION_FLOOR
    # A hopelessly tiny model must never be presented as a prediction.
    assert honesty["presented_as"] in ("prediction", "evidence only")
    if not honesty["held_out_accuracy_passes_floor"]:
        assert honesty["presented_as"] == "evidence only"
    assert report["evaluation"]["samples"] > 0
    assert render_validation_summary(report)


def test_honesty_gate_is_unchanged_by_this_work():
    """The floor itself must not have been lowered to pass the new model."""

    from prototype.ml.cnn import ML_VALIDATION_FLOOR

    assert ML_VALIDATION_FLOOR == 0.60
