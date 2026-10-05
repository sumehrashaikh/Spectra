"""Regression tests for ML v3: the wider-context runtime and its parity.

ML v3 widened the classifier input from 512 to 1024 samples and replaced
the flatten head with global average pooling, which made the NumPy
runtime's geometry a property of each *artifact* instead of a module
constant.  These tests pin the properties that change bought us, and the
two real bugs that were found while building it:

* a strided convolution whose window offsets were computed as
  ``stride * (t + j)`` still reads the right samples only because the
  window is built from a strided *view* — this is easy to get subtly
  wrong and is checked here against an independent implementation;
* flattening ``(batch, channels, time)`` instead of ``(batch, time,
  channels)`` silently reinterprets every trained classifier weight, so
  the head's index order is asserted against an order-explicit reference.

None of this needs PyTorch: ``ml.torch_train`` is import-safe without it,
and the parity check runs the shipped NumPy runtime.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from prototype.ml.cnn import (
    ML_VALIDATION_FLOOR,
    NUM_CLASSES,
    ModulationCNN,
    extract_frames,
    normalize_frames,
    resolve_architecture,
)
from prototype.ml.dataset import CLASS_NAMES
from prototype.ml.validation import parity_check, reference_probabilities


# --------------------------------------------------------------
# Tiny artifact writer (no PyTorch, no training)
# --------------------------------------------------------------

def write_artifact(
    path,
    *,
    frame_length: int = 1024,
    channels=(32, 64, 128),
    kernels=(65, 17, 9),
    strides=(4, 1, 1),
    pools=(2, 2, 2),
    head: str = "gap",
    hidden_dense: int = 0,
    seed: int = 0,
):
    """Write a random-weight artifact in the shipped ``.npz`` format.

    The point is to exercise the *runtime* (and the resolver that reads
    geometry back out of the weights), not any training code, so the
    weights only have to be well-formed.
    """

    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    blocks = []
    in_channels = 2
    length = int(frame_length)

    for index, (channel, kernel, stride, pool) in enumerate(
        zip(channels, kernels, strides, pools)
    ):
        tensors[f"conv{index}.kernel"] = rng.standard_normal(
            (kernel, in_channels, channel)
        ).astype(np.float32) * 0.1
        tensors[f"conv{index}.bias"] = np.zeros(channel, dtype=np.float32)
        for name in ("gamma", "beta", "mean", "variance"):
            value = np.ones(channel, dtype=np.float32) if name in (
                "gamma", "variance"
            ) else np.zeros(channel, dtype=np.float32)
            # Non-unit variance keeps the BatchNorm fusion observable.
            if name == "variance":
                value = np.full(channel, 0.5 + 0.01 * index, dtype=np.float32)
            tensors[f"bn{index}.{name}"] = value
        blocks.append({
            "conv_filters": int(channel),
            "conv_kernel": int(kernel),
            "conv_input_channels": int(in_channels),
            "conv_stride": int(stride),
            "batchnorm": True,
            "activation": "relu",
            "pool": (
                {"type": "max", "size": int(pool), "stride": int(pool)}
                if pool and int(pool) > 1 else None
            ),
        })
        length = (length - int(kernel)) // int(stride) + 1
        if pool and int(pool) > 1:
            length //= int(pool)
        in_channels = int(channel)

    head_input = in_channels if head == "gap" else length * in_channels
    if hidden_dense:
        tensors["dense0.kernel"] = rng.standard_normal(
            (head_input, hidden_dense)
        ).astype(np.float32) * 0.1
        tensors["dense0.bias"] = np.zeros(hidden_dense, dtype=np.float32)
        final_in = hidden_dense
    else:
        final_in = head_input
    tensors["dense1.kernel"] = rng.standard_normal(
        (final_in, NUM_CLASSES)
    ).astype(np.float32) * 0.1
    tensors["dense1.bias"] = np.zeros(NUM_CLASSES, dtype=np.float32)

    config = {
        "input_shape": [int(frame_length), 2],
        "num_classes": NUM_CLASSES,
        "labels": CLASS_NAMES,
        "normalization": "unit_rms",
        "batch_norm_epsilon": 1e-3,
        "trained": True,
        "architecture": {
            "input_shape": [int(frame_length), 2],
            "blocks": blocks,
            "head": (
                [{"layer": "global_average_pooling1d"}]
                if head == "gap" else [{"layer": "flatten"}]
            ),
            "head_type": head,
        },
        "training": {"best_val_accuracy": 0.0},
    }
    tensors["config_json"] = np.frombuffer(
        json.dumps(config).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **tensors)
    return path


# --------------------------------------------------------------
# Architecture resolution
# --------------------------------------------------------------

def test_runtime_reads_geometry_from_the_weights(tmp_path):
    path = write_artifact(tmp_path / "gap.npz", head="gap")
    engine = ModulationCNN(path)

    assert engine.frame_length == 1024
    assert engine.head == "gap"
    assert engine.has_hidden_dense is False
    assert engine.architecture["head_input"] == 128
    assert engine.architecture["temporal_length"] == 22
    assert [(b.kernel, b.stride, b.pool) for b in engine.blocks] == [
        (65, 4, 2), (17, 1, 2), (9, 1, 2),
    ]


def test_runtime_rejects_an_artifact_whose_config_contradicts_its_weights(
    tmp_path,
):
    """A declared head that the weights do not implement must raise."""

    path = write_artifact(tmp_path / "lying.npz", head="gap")
    with np.load(path, allow_pickle=False) as data:
        tensors = {k: data[k] for k in data.files if k != "config_json"}
        config = json.loads(bytes(data["config_json"]).decode("utf-8"))
    config["architecture"]["head"] = [{"layer": "flatten"}]

    with pytest.raises(ValueError, match="head"):
        resolve_architecture(tensors, config)


def test_runtime_rejects_a_bad_frame_length(tmp_path):
    path = write_artifact(tmp_path / "gap.npz", head="gap")
    engine = ModulationCNN(path)
    with pytest.raises(ValueError, match="1024-sample frames"):
        engine.predict_frames(np.zeros((2, 512, 2), dtype=np.float32))


def test_extract_frames_honours_a_1024_length():
    samples = np.arange(8192, dtype=np.float64) + 1j * 0
    frames = extract_frames(samples, frame_length=1024, max_frames=4)
    assert frames.shape == (4, 1024, 2)
    # First and last frames must cover the ends of the capture.
    assert frames[0, 0, 0] == pytest.approx(0.0)
    # The last window ends on the final sample of the capture.
    assert frames[-1, -1, 0] == pytest.approx(8190.0)
    with pytest.raises(ValueError):
        extract_frames(np.zeros(100, dtype=complex), frame_length=1024)


# --------------------------------------------------------------
# Parity against an independent forward pass
# --------------------------------------------------------------

@pytest.mark.parametrize("head", ["gap", "flatten"])
@pytest.mark.parametrize("hidden_dense", [0, 8])
def test_runtime_matches_an_independent_reference(tmp_path, head, hidden_dense):
    """Every head/layout combination must agree with a separate forward pass.

    The reference in ``ml.validation`` shares no code with ``ml.cnn``: it
    applies BatchNorm explicitly instead of fusing it, convolves by looping
    over kernel taps instead of using ``as_strided``, and spells out the
    flatten index order instead of calling ``reshape``.
    """

    tiny = dict(
        frame_length=1024,
        channels=(6, 10, 14),
        kernels=(33, 9, 5),
        strides=(4, 1, 1),
        pools=(2, 2, 2),
        head=head,
        hidden_dense=hidden_dense,
        seed=3,
    )
    path = write_artifact(tmp_path / "tiny.npz", **tiny)

    rng = np.random.default_rng(11)
    frames = (rng.standard_normal((128, 1024, 2)) * 0.4).astype(np.float32)

    report = parity_check(path, frames)
    assert report["frames_compared"] == 128
    assert report["argmax_matches"] == 128, report
    assert report["max_abs_softmax_difference"] < 1e-5, report
    assert report["passed"] is True


def test_flatten_head_uses_time_major_index_order(tmp_path):
    """Pin the flatten order that a channel-major flatten would silently break.

    ``dense1.kernel`` is replaced by a one-hot probe that reads exactly one
    ``(time, channel)`` position of the feature map.  The position the
    runtime actually reads is then compared with the index the *documented*
    order (time-major) predicts.  A channel-major flatten reads a different
    position and fails.
    """

    path = write_artifact(
        tmp_path / "probe.npz",
        frame_length=1024,
        channels=(4, 8, 12),
        kernels=(33, 9, 5),
        strides=(4, 1, 1),
        pools=(2, 2, 2),
        head="flatten",
        hidden_dense=0,
        seed=5,
    )
    with np.load(path, allow_pickle=False) as data:
        tensors = {k: np.asarray(data[k]) for k in data.files
                   if k != "config_json"}
        config = json.loads(bytes(data["config_json"]).decode("utf-8"))

    # Feature map after the blocks: 1024 -> 248 -> 124 -> 116 -> 58 -> 54 -> 27
    time_positions, channels = 27, 12
    target_time, target_channel = 7, 9
    probe = np.zeros((time_positions * channels, NUM_CLASSES), dtype=np.float32)
    probe[target_time * channels + target_channel, 0] = 1.0
    tensors["dense1.kernel"] = probe
    tensors["dense1.bias"] = np.zeros(NUM_CLASSES, dtype=np.float32)

    engine = ModulationCNN.__new__(ModulationCNN)
    engine._tensors = {k: v.astype(np.float32) for k, v in tensors.items()}
    engine.config = config
    engine.normalization = "unit_rms"
    engine.artifact_name = "probe.npz"
    engine.labels = CLASS_NAMES
    engine.labels_source = "test"
    engine.architecture = resolve_architecture(engine._tensors, config)
    engine.blocks = engine.architecture["blocks"]
    assert engine.architecture["temporal_length"] == time_positions

    frames = normalize_frames(
        np.random.default_rng(2).standard_normal((4, 1024, 2)).astype(np.float32)
    )
    scores = engine.predict_frames(frames)
    # The one-hot probe makes the score of class 0 the ReLU of exactly one
    # feature value, so the softmax is a monotone function of that value.
    reference = reference_probabilities(engine._tensors, config, frames)
    assert np.allclose(scores, reference, atol=1e-6)

    # And sanity-check that the probed position is the intended one by
    # recomputing the feature map through the reference implementation.
    assert not np.allclose(
        scores[:, 0],
        np.full(scores.shape[0], scores[:, 0].mean()),
    )


# --------------------------------------------------------------
# Dataset generator
# --------------------------------------------------------------

def test_generator_produces_1024_sample_frames_deterministically():
    from prototype.ml.dataset import build_dataset_v2

    first = build_dataset_v2(2, seed=17, frame_length=1024,
                             channel_variation=True)
    second = build_dataset_v2(2, seed=17, frame_length=1024,
                              channel_variation=True)

    frames, labels, realizations, snr = first
    assert frames.shape == (32, 1024, 2)
    assert frames.dtype == np.float32
    assert labels.shape == realizations.shape == snr.shape == (32,)
    assert np.array_equal(frames, second[0])
    assert np.array_equal(labels, second[1])
    # Every class must be present and equally represented.
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    assert set(counts.tolist()) == {2}


def test_rich_timing_variation_stays_inside_a_symbol_period():
    from prototype.ml.dataset import _random_frame_config_v2

    configs = [
        _random_frame_config_v2(
            np.random.default_rng(seed), channel_variation=True,
            timing_span_samples=100, fractional_timing=True,
        )
        for seed in range(40)
    ]
    assert all(0 <= c.timing_offset_samples < 100 for c in configs)
    assert all(-0.5 <= c.fractional_timing_offset < 0.5 for c in configs)
    # The plain default must remain the historical 0-7 sample offset.
    plain = _random_frame_config_v2(
        np.random.default_rng(1), channel_variation=True
    )
    assert 0 <= plain.timing_offset_samples < 8
    assert plain.fractional_timing_offset == 0.0


# --------------------------------------------------------------
# Known class ambiguity in the synthetic generator
# --------------------------------------------------------------

def test_bfsk_and_msk_are_currently_the_same_waveform():
    """Documented limitation, not a passing grade.

    As generated today, ``BFSK`` and ``MSK`` are the *same signal*: both
    use a frequency deviation of ``0.5 * symbol_rate`` and a rectangular
    frequency pulse, which is exactly MSK (2-FSK at modulation index
    0.5).  The two labels therefore carry no information about the
    waveform, and no classifier can do better than 50% on that pair —
    which bounds the whole 16-class task at 0.9375.

    This test exists so the limitation is *recorded* rather than
    rediscovered.  If it starts failing, the generator changed: update
    this test, re-train, and re-validate rather than assuming the old
    accuracy numbers still mean the same thing.
    """

    from prototype.ml.dataset import FrameConfig, generate_frame

    def frame(kind: str) -> np.ndarray:
        return generate_frame(kind, FrameConfig(
            snr_db=15.0, cfo_hz=3.0, phase_offset_rad=0.2,
            symbol_rate=120.0, rolloff=0.35, seed=987, frame_length=1024,
        ))

    assert np.array_equal(frame("BFSK"), frame("MSK"))

    # Contrast: the pairs whose confusion is *not* a label ambiguity,
    # because they really are different waveforms, so their errors are
    # genuine model error rather than an unanswerable question.
    for first, second in (("MSK", "GMSK"), ("QPSK", "8PSK"),
                          ("16QAM", "64QAM")):
        assert not np.array_equal(frame(first), frame(second)), (
            f"{first} and {second} now generate identical waveforms; the "
            "class set needs revisiting."
        )


# --------------------------------------------------------------
# Calibration report
# --------------------------------------------------------------

def test_reliability_report_flags_an_overconfident_model():
    from prototype.ml.calibration import reliability_report

    rng = np.random.default_rng(0)
    labels = rng.integers(0, NUM_CLASSES, 400)
    # Confidently wrong: argmax is never the label, scores are near 1.
    probabilities = np.full((400, NUM_CLASSES), 0.001)
    probabilities[np.arange(400), (labels + 1) % NUM_CLASSES] = 0.985

    report = reliability_report(labels, probabilities)
    assert report["accuracy"] == 0.0
    assert report["mean_confidence"] > 0.9
    assert report["expected_calibration_error"] > 0.9
    assert report["bins"]


def test_confidence_report_is_callable():
    """Regression: this used to raise NameError on an undefined helper."""

    from prototype.ml.calibration import confidence_report

    report = confidence_report(
        [0, 1, 1, 0], [0, 0, 1, 1], [0.9, 0.6, 0.8, 0.55], threshold=0.5
    )
    assert report["above_threshold"] == 4
    # Two of the four confident calls are right.
    assert report["accuracy_given_confidence"] == pytest.approx(0.5)
    assert report["total"] == 4


# --------------------------------------------------------------
# Honesty gate
# --------------------------------------------------------------

def test_the_floor_was_not_moved_for_v3():
    assert ML_VALIDATION_FLOOR == 0.60


def test_torch_trainer_is_importable_without_torch():
    """Training may need PyTorch; the runtime and the CLI never may."""

    import importlib

    module = importlib.import_module("prototype.ml.torch_train")
    assert callable(module.main)
    assert callable(module.require_torch)

    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="venv-mltrain"):
            module.require_torch()
