"""Evaluation metrics for the Spectra ML v1 classifier.

ML v1 must be honest about its validation scope.  Reporting only a
single overall accuracy number is not enough, so this module exposes:

- per-class precision / recall / F1,
- a confusion matrix,
- sample counts,
- per-SNR performance (where practical),
- macro / weighted averages.

Everything is computed from integer label arrays (no deep-learning
dependency), so the metrics can be embedded in smoke tests and CI.

Confidence handling
-------------------
This module deals with *measured* class predictions.  It does not turn
model logits into calibrated probabilities.  A model score is exposed
by the inference runtime and labeled explicitly as a model score; this
module only distinguishes ``predicted_class`` from ``confidence`` at
the decision level and leaves calibration as an open ML v1 limitation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _as_int_arrays(
    y_true: Sequence[int], y_pred: Sequence[int], sample_weight: Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Coerce predictions and references to 1-D integer arrays."""

    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"{y_true_arr.size} != {y_pred_arr.size}"
        )

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != y_true_arr.size:
            raise ValueError(
                f"sample_weight length mismatch: {w.size} != {y_true_arr.size}"
            )
        return y_true_arr, y_pred_arr, w
    return y_true_arr, y_pred_arr, None


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], *, labels: Sequence[int] | None = None
) -> np.ndarray:
    """Compute a confusion matrix C where ``C[i, j]`` counts true-class i
    predicted as class j."""

    y_true_arr, y_pred_arr, _ = _as_int_arrays(y_true, y_pred)
    classes = labels if labels is not None else np.unique(
        np.concatenate([y_true_arr, y_pred_arr])
    )
    classes = np.asarray(classes, dtype=np.int64)
    index = {int(clas): pos for pos, clas in enumerate(classes)}

    matrix = np.zeros((classes.size, classes.size), dtype=np.int64)
    for true, pred in zip(y_true_arr.tolist(), y_pred_arr.tolist()):
        if true in index and pred in index:
            matrix[index[true], index[pred]] += 1

    return matrix, classes


def per_class_stats(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Return per-class precision, recall, F1 and support."""

    y_true_arr, y_pred_arr, weights = _as_int_arrays(y_true, y_pred, sample_weight)
    matrix, classes = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    stats: dict[str, dict[str, float]] = {}
    for pos, clas in enumerate(classes):
        true_pos = int(matrix[pos, pos])
        predicted_pos = int(matrix[:, pos].sum())
        actual_pos = int(matrix[pos, :].sum())

        precision = float(true_pos / predicted_pos) if predicted_pos else 0.0
        recall = float(true_pos / actual_pos) if actual_pos else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        support = float(actual_pos if weights is None else float(np.sum(weights[y_true_arr == clas])))

        stats[str(clas)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return stats


def summary_stats(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    sample_weight: Sequence[float] | None = None,
    average: str = "macro",
) -> dict[str, float]:
    """Overall accuracy and the requested macro/weighted averages."""

    y_true_arr, y_pred_arr, weights = _as_int_arrays(y_true, y_pred, sample_weight)
    matrix, classes = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    accuracy = float(
        np.trace(matrix) / max(matrix.sum(), 1)
    )

    per_class = per_class_stats(y_true_arr, y_pred_arr, labels=classes, sample_weight=weights)
    supports = np.array([per_class[str(int(clas))]["support"] for clas in classes], dtype=np.float64)

    if average == "macro":
        accuracy = accuracy
        return {
            "accuracy": accuracy,
            "precision": float(np.mean([per_class[str(int(c))]["precision"] for c in classes])),
            "recall": float(np.mean([per_class[str(int(c))]["recall"] for c in classes])),
            "f1": float(np.mean([per_class[str(int(c))]["f1"] for c in classes])),
        }

    if average == "weighted":
        total = max(supports.sum(), 1.0)
        return {
            "accuracy": accuracy,
            "precision": float(np.sum([per_class[str(int(c))]["precision"] * supports[i] for i, c in enumerate(classes)]) / total),
            "recall": float(np.sum([per_class[str(int(c))]["recall"] * supports[i] for i, c in enumerate(classes)]) / total),
            "f1": float(np.sum([per_class[str(int(c))]["f1"] * supports[i] for i, c in enumerate(classes)]) / total),
        }

    raise ValueError(f"Unknown average '{average}'; expected 'macro' or 'weighted'.")


def per_snr_stats(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    snr_db: Sequence[float],
    *,
    labels: Sequence[int] | None = None,
) -> dict[float, dict[str, float]]:
    """Per-SNR accuracy / F1, useful for the SNR sweep that ML v1 is
    required to measure."""

    y_true_arr, y_pred_arr, _ = _as_int_arrays(y_true, y_pred)
    snr_arr = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if snr_arr.size != y_true_arr.size:
        raise ValueError("snr_db length must match y_true/y_pred.")

    unique_snr = np.unique(snr_arr)
    summary: dict[float, dict[str, float]] = {}
    for snr in unique_snr:
        mask = snr_arr == snr
        if not np.any(mask):
            continue
        mask = mask & (y_true_arr.size > 0)
        mask_idx = np.where(mask)[0]
        if mask_idx.size == 0:
            summary[float(snr)] = {"accuracy": 0.0, "f1": 0.0, "sample_count": 0}
            continue
        stats = summary_stats(
            y_true_arr[mask],
            y_pred_arr[mask],
            labels=labels,
            average="macro",
        )
        summary[float(snr)] = {
            "accuracy": stats["accuracy"],
            "f1": stats["f1"],
            "sample_count": float(mask_idx.size),
        }
    return summary


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    sample_weight: Sequence[float] | None = None,
    digits: int = 4,
) -> str:
    """Render a human-readable per-class report plus weighted averages."""

    stats = per_class_stats(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    ordered = [int(clas) for clas in (labels if labels is not None else sorted({int(x) for x in stats}))]
    rows = []
    header = f"{'class':>8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"
    rows.append(header)
    rows.append("-" * len(header))

    for clas in ordered:
        key = str(clas)
        if key not in stats:
            continue
        s = stats[key]
        rows.append(
            f"{clas:>8} {s['precision']:>10.{digits}f} {s['recall']:>10.{digits}f} "
            f"{s['f1']:>10.{digits}f} {int(s['support']):>10}"
        )

    weighted = summary_stats(y_true, y_pred, labels=labels, sample_weight=sample_weight, average="weighted")
    rows.append("-" * len(header))
    rows.append(
        f"{'weighted':>8} {weighted['precision']:>10.{digits}f} {weighted['recall']:>10.{digits}f} "
        f"{weighted['f1']:>10.{digits}f} {sum(int(stats.get(str(int(c)), {'support': 0})['support']) for c in ordered):>10}"
    )
    return "\n".join(rows)
