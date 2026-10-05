"""Artifact-level validation for the modulation CNN.

The training loop reports the metrics of the model it holds in memory.
This module answers the questions that actually matter for the *shipped*
artifact:

* **Parity** — does the NumPy inference runtime
  (``ml.cnn.ModulationCNN``) produce the same predictions as the
  training-time model on identical input?  Training deliberately keeps a
  separate forward implementation from the runtime, so this is the
  regression test that keeps the two from drifting.
* **Independent hold-out** — how accurate is the artifact on frames
  generated with a *different* seed from the training run, scored through
  the runtime rather than the training code?
* Per-class accuracy, confusion matrix and per-SNR accuracy for that
  hold-out.

Nothing here changes the pipeline, the DSP classifier or any default; it
reads an artifact and prints evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from prototype.ml.calibration import reliability_report
from prototype.ml.cnn import (
    BN_EPSILON,
    ML_VALIDATION_FLOOR,
    NUM_CLASSES,
    ModulationCNN,
    normalize_frames,
)
from prototype.ml.train import Model, evaluate_predictions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def load_training_model(
    artifact_path: str | Path,
    dtype: str | np.dtype = np.float32,
) -> Model:
    """Rebuild the NumPy-training model from an exported artifact.

    Only meaningful for artifacts produced by ``ml.train`` (flatten head,
    ``dense0`` present).  ML v3 artifacts are trained by
    ``ml.torch_train`` and have no NumPy training-time twin, which is why
    ``parity_check`` falls back to an independent reference implementation.
    """

    with np.load(artifact_path, allow_pickle=False) as data:
        tensors = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "config_json"
        }
    return Model(tensors, dtype=dtype)


def reference_probabilities(
    tensors: dict[str, np.ndarray],
    config: dict,
    frames: np.ndarray,
) -> np.ndarray:
    """An independent, deliberately naive forward pass over an artifact.

    Written without ``as_strided``, weight fusion or any shared helper, so
    it shares no code path with ``ml.cnn`` and therefore actually tests
    it.  It is slow (a Python loop over kernel taps per layer) and is only
    used on the handful of frames a parity check needs.

    It follows the same geometry, which it re-derives from the tensors
    itself rather than trusting ``ml.cnn``'s resolver.
    """

    frames = np.asarray(frames, dtype=np.float64)

    conv_ids = sorted(
        int(name[len("conv"):].split(".")[0])
        for name in tensors
        if name.startswith("conv") and name.endswith(".kernel")
    )
    architecture = config.get("architecture")
    declared = list(architecture.get("blocks") or []) \
        if isinstance(architecture, dict) else []

    h = frames
    for index, _ in enumerate(conv_ids):
        kernel = np.asarray(tensors[f"conv{index}.kernel"], dtype=np.float64)
        bias = np.asarray(tensors[f"conv{index}.bias"], dtype=np.float64)
        spec = declared[index] if index < len(declared) else {}
        stride = int(spec.get("conv_stride", 1))
        pool_spec = spec.get("pool", {"size": 2})
        pool = None if not pool_spec else int(pool_spec.get("size", 2))

        # BatchNorm, applied explicitly rather than fused into the kernel.
        scale = np.asarray(tensors[f"bn{index}.gamma"], dtype=np.float64) / \
            np.sqrt(np.asarray(tensors[f"bn{index}.variance"], dtype=np.float64)
                    + BN_EPSILON)
        shift = np.asarray(tensors[f"bn{index}.beta"], dtype=np.float64)
        mean = np.asarray(tensors[f"bn{index}.mean"], dtype=np.float64)

        taps, _in_channels, out_channels = kernel.shape
        length = (h.shape[1] - taps) // stride + 1
        out = np.zeros((h.shape[0], length, out_channels), dtype=np.float64)
        for tap in range(taps):
            window = h[:, tap : tap + stride * length : stride, :]
            out += np.einsum("blc,cn->bln", window, kernel[tap])
        out = (out + bias - mean) * scale + shift
        h = np.maximum(out, 0.0)

        if pool and pool > 1:
            keep = (h.shape[1] // pool) * pool
            h = h[:, :keep, :].reshape(h.shape[0], -1, pool, h.shape[2]).max(axis=2)

    # Head reduction happens before any hidden layer, for both head shapes.
    expected = (
        int(tensors["dense0.kernel"].shape[0]) if "dense0.kernel" in tensors
        else int(tensors["dense1.kernel"].shape[0])
    )
    # Written out index by index (time-major, then channel) rather than as
    # a ``reshape``: the runtime flattens ``(batch, time, channel)``, and a
    # reference that merely reshapes the same array the same way would
    # agree with it even if the runtime had the axes the wrong way round.
    # Spelling out the order is what makes this an actual check.
    flattened = np.stack(
        [h[:, t, c]
         for t in range(h.shape[1])
         for c in range(h.shape[2])],
        axis=1,
    )
    if expected == h.shape[2]:
        features = h.mean(axis=1)
    elif expected == flattened.shape[1]:
        features = flattened
    else:
        raise ValueError(
            "Cannot tell how this head consumes the feature map: the head "
            f"wants {expected} inputs, flatten gives {flattened.shape[1]} "
            f"and pooling gives {h.shape[2]}."
        )

    if "dense0.kernel" in tensors:
        features = np.maximum(
            features @ np.asarray(tensors["dense0.kernel"], dtype=np.float64)
            + np.asarray(tensors["dense0.bias"], dtype=np.float64),
            0.0,
        )
    logits = features @ np.asarray(tensors["dense1.kernel"], dtype=np.float64) \
        + np.asarray(tensors["dense1.bias"], dtype=np.float64)
    return _softmax(logits)


def parity_check(
    artifact_path: str | Path,
    frames: np.ndarray,
    *,
    dtype: str | np.dtype = np.float32,
    argmax_tolerance: float = 0.0,
    max_abs_tolerance: float = 1e-4,
    reference_scores: np.ndarray | None = None,
    reference_name: str = "independent NumPy reference",
) -> dict[str, Any]:
    """Compare the NumPy runtime against a second forward pass.

    ``frames`` must be raw ``(N, L, 2)`` frames; normalization is applied
    by both sides exactly as the pipeline applies it.

    ``reference_scores`` lets the caller supply the scores of the model
    that was actually trained (e.g. the PyTorch network) instead of using
    the built-in naive reference.  Either way the comparison is between
    two independent implementations, which is the point: the runtime
    fuses BatchNorm into the convolution weights and reorders axes, and
    neither is checked by anything else.

    Returns the fraction of matching argmax predictions and the largest
    absolute softmax difference.  The check fails when any prediction
    disagrees or the score difference exceeds the tolerance.
    """

    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[2] != 2:
        raise ValueError(
            f"frames must have shape (N, L, 2), got {frames.shape}"
        )

    normalized = normalize_frames(frames)

    runtime = ModulationCNN(artifact_path)
    if normalized.shape[1] != runtime.frame_length:
        raise ValueError(
            f"Artifact {runtime.artifact_name} expects "
            f"{runtime.frame_length}-sample frames, got "
            f"{normalized.shape[1]}."
        )
    runtime_scores = runtime.predict_frames(normalized)

    if reference_scores is None:
        with np.load(artifact_path, allow_pickle=False) as data:
            tensors = {
                key: np.asarray(data[key])
                for key in data.files
                if key != "config_json"
            }
        reference_scores = reference_probabilities(
            tensors, runtime.config, normalized
        )
    else:
        reference_scores = np.asarray(reference_scores, dtype=np.float64)
        if reference_scores.shape != runtime_scores.shape:
            raise ValueError(
                f"reference scores have shape {reference_scores.shape}, "
                f"runtime produced {runtime_scores.shape}"
            )

    runtime_argmax = runtime_scores.argmax(axis=1)
    reference_argmax = reference_scores.argmax(axis=1)

    matches = int((runtime_argmax == reference_argmax).sum())
    total = int(runtime_argmax.size)
    max_abs = float(np.abs(
        runtime_scores.astype(np.float64) - reference_scores
    ).max()) if total else 0.0

    return {
        "artifact": Path(artifact_path).name,
        "reference": reference_name,
        "frames_compared": total,
        "argmax_matches": matches,
        "argmax_match_rate": (matches / total) if total else None,
        "max_abs_softmax_difference": max_abs,
        "tolerance": {
            "argmax_disagreements_allowed": argmax_tolerance,
            "max_abs_softmax": max_abs_tolerance,
        },
        "passed": bool(
            total > 0
            and (total - matches) <= argmax_tolerance
            and max_abs <= max_abs_tolerance
        ),
        "note": (
            "The runtime fuses BatchNorm into the convolution weights and "
            "uses (batch, time, channel) with a time-major flatten; a match "
            "here means it agrees with an independent forward pass "
            "numerically, not merely in spirit."
        ),
    }


def validate_artifact(
    artifact_path: str | Path,
    *,
    frames_per_class: int = 60,
    seed: int = 20260930,
    channel_variation: bool = True,
    snr_range: tuple[float, float] = (0.0, 18.0),
    parity_frames: int = 128,
    include_artifact_config: bool = True,
    timing_span_samples: int | None = None,
    fractional_timing: bool | None = None,
) -> dict[str, Any]:
    """Score an artifact on held-out frames using the NumPy runtime only.

    The evaluation set is generated with ``seed`` (a *different* seed from
    the training run by construction of the caller's choice), so it is
    independent of training data.

    The frame length and the timing/training distribution come from the
    artifact itself when the artifact records them, so a 1024-sample model
    is not silently scored on 512-sample frames.  Both can be overridden.
    """

    from prototype.ml.dataset import build_dataset_v2

    runtime = ModulationCNN(artifact_path)
    training = runtime.config.get("training") or {}
    if timing_span_samples is None:
        timing_span_samples = int(training.get("timing_span_samples", 8))
    if fractional_timing is None:
        fractional_timing = bool(training.get("fractional_timing", False))

    frames, labels, _realizations, snr_db = build_dataset_v2(
        frames_per_class, seed=seed, snr_range=snr_range,
        channel_variation=channel_variation,
        frame_length=runtime.frame_length,
        timing_span_samples=timing_span_samples,
        fractional_timing=fractional_timing,
    )

    scores = runtime.predict_frames(normalize_frames(frames))
    predictions = scores.argmax(axis=1)

    metrics = evaluate_predictions(labels, predictions, snr_db,
                                   snr_bin_width=3.0)

    parity = parity_check(
        artifact_path, frames[: min(parity_frames, frames.shape[0])]
    )

    declared = runtime.validation_accuracy
    payload: dict[str, Any] = {
        "artifact": str(artifact_path),
        "artifact_name": Path(artifact_path).name,
        "evaluation": {
            "generator": (
                "ml.dataset.build_dataset_v2 (held-out seed, "
                f"L={runtime.frame_length}, "
                f"snr {snr_range[0]:g}-{snr_range[1]:g} dB"
                f", timing span {timing_span_samples}"
                f"{' + fractional' if fractional_timing else ''})"
            ),
            "seed": seed,
            "frame_length": runtime.frame_length,
            "head": runtime.head,
            "per_snr_bin_width_db": 3.0,
            "frames_per_class": frames_per_class,
            "samples": metrics["samples"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_accuracy": metrics["per_class_accuracy"],
            "per_class_f1": metrics["per_class_f1"],
            "support": metrics["support"],
            "confusion_matrix_labels": metrics["confusion_matrix_labels"],
            "confusion_matrix": metrics["confusion_matrix"],
            "per_snr_accuracy": metrics.get("per_snr_accuracy"),
            "calibration": reliability_report(labels, scores),
        },
        "parity": parity,
        "honesty": {
            "declared_validation_accuracy_in_artifact": declared,
            "validation_floor": ML_VALIDATION_FLOOR,
            "declared_passes_floor": bool(
                declared is not None and declared >= ML_VALIDATION_FLOOR
            ),
            "runtime_validated": runtime.validated,
            "presented_as": (
                "prediction" if runtime.validated else "evidence only"
            ),
            "held_out_accuracy_passes_floor": bool(
                metrics["accuracy"] >= ML_VALIDATION_FLOOR
            ),
            "note": (
                "This is synthetic held-out evidence. It is NOT real-world "
                "or hardware validation, and passing the floor only means "
                "the artifact may be presented as a prediction within the "
                "synthetic domain it was trained on."
            ),
        },
    }

    if include_artifact_config:
        payload["artifact_metadata"] = {
            "labels": list(runtime.labels),
            "labels_source": runtime.labels_source,
            "trained": runtime.trained,
            "status": runtime.status,
            "architecture": runtime.config.get("architecture"),
            "provenance": runtime.config.get("provenance"),
            "training": {
                key: value
                for key, value in (runtime.config.get("training") or {}).items()
                if key not in ("history", "train_metrics", "val_metrics",
                               "test_metrics")
            },
        }
    return payload


def render_validation_summary(payload: dict[str, Any]) -> list[str]:
    """Compact human-readable lines for CLI output."""

    evaluation = payload["evaluation"]
    honesty = payload["honesty"]
    parity = payload["parity"]

    lines = [
        f"artifact: {payload['artifact_name']} "
        f"(L={evaluation['frame_length']}, head {evaluation['head']})",
        f"held-out synthetic accuracy: {evaluation['accuracy']:.3f} "
        f"({evaluation['samples']} frames, macro-F1 "
        f"{evaluation['macro_f1']:.3f})",
        f"parity: {parity['argmax_matches']}/{parity['frames_compared']} "
        f"argmax match, max |dsoftmax| "
        f"{parity['max_abs_softmax_difference']:.2e} "
        f"-> {'PASS' if parity['passed'] else 'FAIL'}",
        f"floor {honesty['validation_floor']:.2f}: "
        f"held-out {'PASSES' if honesty['held_out_accuracy_passes_floor'] else 'does NOT pass'}"
        f"; runtime presents results as {honesty['presented_as']}",
    ]

    calibration = evaluation.get("calibration") or {}
    if calibration:
        lines.append(
            f"confidence: mean {calibration['mean_confidence']:.3f}, "
            f"correct {calibration['mean_correct_confidence']:.3f}, "
            f"wrong {calibration['mean_wrong_confidence']:.3f}, "
            f"ECE {calibration['expected_calibration_error']:.3f} "
            "(uncalibrated softmax)"
        )

    lines.append("")
    lines.append(
        f"{'class':<10}{'accuracy':>10}{'support':>9}"
    )
    for name, accuracy in evaluation["per_class_accuracy"].items():
        shown = "n/a" if accuracy is None else f"{accuracy:.3f}"
        lines.append(
            f"{name:<10}{shown:>10}{evaluation['support'].get(name, 0):>9}"
        )

    per_snr = evaluation.get("per_snr_accuracy") or {}
    if per_snr:
        lines.append("")
        lines.append(f"{'snr dB':<10}{'accuracy':>10}{'n':>9}")
        for snr, values in per_snr.items():
            lines.append(
                f"{snr:<10}{values['accuracy']:>10.3f}"
                f"{values['sample_count']:>9}"
            )
    return lines
