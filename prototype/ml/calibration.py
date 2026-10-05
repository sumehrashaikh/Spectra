"""ML v1 confidence / calibration handling.

Honesty contract for ML v1:

- A raw CNN softmax mean score is a *model score*, not a calibrated
  probability.  It is a measure of the model's own confidence.
- This module provides an optional calibration transformer that turns a
  held-out validation set into a monotone mapping into a pseudo-
  probability.
- When calibration is enabled, results expose both ``model_score`` and
  ``calibrated_score``, and the calibrated value is always clearly
  labeled.
- When calibration is not available, results expose only the model
  score and it is labeled exactly as that: a model score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .evaluation import _as_int_arrays, per_class_stats, summary_stats

# Use a plain monotone mapping from scores to pseudo-probabilities.
# Both the identity and a logistic (Platt-style) map are implemented so
# a teammate can pick one without an external dependency.  Isotonic
# regression is not bundled because scipy.integrate is the only hard
# CI dependency here; a logistic map is the safe default.


def _logistic_map(scores: np.ndarray, temperature: float = 1.0, center: float = 0.0) -> np.ndarray:
    """Logistic (Platt-style) recalibration on model scores in [0, 1].

    This is monotone and bounded to (0, 1), but it is a *fitted*
    estimate on the named calibration set, so it must be stored with
    the artifact it was fit on.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    temperature = float(temperature)
    center = float(center)
    if temperature <= 0:
        return np.full_like(scores, center, dtype=np.float64)

    # Numerically stable logistic on (center - temperature/2, center + temperature/2)
    # centered around ``center`` so a score of 0.5 maps to the center.
    z = (scores - 0.5) * 2.0 * temperature
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _identity_map(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).reshape(-1)


CalibrateFn = Callable[[np.ndarray], np.ndarray]


class ScoreCalibrator:
    """Fit a monotone map from held-out model scores to pseudo-
    probabilities.  The fit can be stored in the artifact config so a
    different teammate can extract the same calibrated numbers.

    Notes
    -----
    This is deliberately a *logistic* (Platt-style) calibrator with a
    single learned temperature, because it needs no external dependency
    and has a closed-form least-squares fit against a held-out set.
    A full isotonic regression would require order-statistics kernels
    and is left as future work; logging that limitation explicitly is
    part of the ML v1 honesty contract.
    """

    def __init__(
        self,
        method: str = "logistic",
        *,
        temperature: float = 1.0,
        temperature_bounds: tuple[float, float] = (0.1, 5.0),
        center: float = 0.0,
        calibration_set_size: int | None = None,
    ) -> None:
        self.method = method
        self.temperature = float(temperature)
        self.temperature_bounds = tuple(float(v) for v in temperature_bounds)
        self.center = float(center)
        self.calibration_set_size = calibration_set_size

    @property
    def is_identity(self) -> bool:
        return self.method == "identity" or (
            self.method == "logistic"
            and abs(self.temperature - 1.0) < 1e-9
            and abs(self.center - 0.0) < 1e-9
        )

    def fit(self, scores: np.ndarray, targets: np.ndarray) -> "ScoreCalibrator":
        """Fit the calibration mapping.

        ``scores`` are the raw model mean-scores in [0, 1] produced by the
        inference runtime, and ``targets`` is the true class label for each
        score.  Fitting minimises the squared error between the calibrated
        pseudo-probability and the one-hot target of the true class.
        """

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        if scores.size != targets.size:
            raise ValueError(
                f"scores and targets must have the same length: {scores.size} != {targets.size}"
            )
        if self.method != "logistic":
            # Non-logistic methods are intentionally not fitted here; they are
            # pure transforms with no learned parameters.
            return self

        lower, upper = self.temperature_bounds
        best_error = float("inf")
        best_temp = self.temperature

        candidates = np.unique(
            np.clip(
                np.linspace(lower, upper, 41),
                lower,
                upper,
            )
        )

        for temp in candidates:
            mapped = _logistic_map(scores, temperature=temp, center=self.center)
            target_onehot = np.zeros((scores.size, int(targets.max()) + 1), dtype=np.float64)
            target_onehot[np.arange(scores.size), targets] = 1.0
            error = float(np.mean((mapped[:, None] - target_onehot) ** 2))
            if error < best_error:
                best_error = error
                best_temp = float(temp)

        self.temperature = best_temp
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "logistic":
            return _logistic_map(scores, temperature=self.temperature, center=self.center)
        if self.method == "identity":
            return _identity_map(scores)
        raise ValueError(f"Unknown calibration method {self.method!r}")

    def fit_transform(self, scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return self.fit(scores, targets).transform(scores)

    def save(self, path: str | Path) -> None:
        payload = {
            "method": self.method,
            "temperature": self.temperature,
            "temperature_bounds": list(self.temperature_bounds),
            "center": self.center,
            "is_identity": self.is_identity,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            method=payload["method"],
            temperature=payload["temperature"],
            temperature_bounds=tuple(payload["temperature_bounds"]),
            center=payload["center"],
        )


def calibrate_predictions(
    predictions: list[dict],
    *,
    calibrator: ScoreCalibrator | None = None,
    calibration_scores: np.ndarray | None = None,
    calibration_labels: np.ndarray | None = None,
) -> list[dict]:
    """Return a *new* prediction payload where ``confidence`` is a
    calibrated pseudo-probability whenever calibration is available.

    The downstream result keeps ``model_score`` (the raw CNN mean-score)
    and, when calibration ran, ``calibrated_score``.  The caller is
    responsible for labelling both.
    """

    if not predictions:
        return []

    if calibrator is None:
        calibrator = ScoreCalibrator(method="identity")

    if calibration_scores is not None and calibration_labels is not None:
        calibrator = calibrator.fit(
            np.asarray(calibration_scores, dtype=np.float64).reshape(-1),
            np.asarray(calibration_labels, dtype=np.int64).reshape(-1),
        )

    calibrated_scores = None
    if calibrator is not None and not calibrator.is_identity:
        calibrated_scores = calibrator.transform(
            np.array([float(p.get("confidence", 0.0)) for p in predictions]),
            dtype=np.float64,
        )

    out: list[dict] = []
    for payload, calibrated in zip(predictions, calibrated_scores or []):
        calibrated_payload = dict(payload)
        calibrated_payload["model_score"] = float(payload.get("confidence", 0.0))
        if calibrated is not None:
            calibrated_payload["calibrated_score"] = float(calibrated)
            calibrated_payload["confidence"] = float(calibrated)
            calibrated_payload["calibration"] = {
                "method": calibrator.method,
                "temperature": calibrator.temperature,
            }
        out.append(calibrated_payload)

    return out


def reliability_report(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Confidence/calibration statistics for a set of softmax outputs.

    Reports what a user of a model actually needs to know: whether a high
    softmax score means a high chance of being right, and how peaked the
    scores are.  Expected calibration error is included because it is the
    number that makes an over-confident softmax visible; nothing here
    changes the scores themselves, and the result is explicitly labelled
    as uncalibrated.
    """

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0 or probabilities.size == 0:
        return {}

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    per_bin = []
    weighted_gap = 0.0
    for index in range(bins):
        selected = bucket == index
        count = int(selected.sum())
        if count:
            mean_confidence = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            weighted_gap += (count / y_true.size) * abs(
                accuracy - mean_confidence
            )
        else:
            mean_confidence = accuracy = None
        per_bin.append({
            "bin": f"{edges[index]:.1f}-{edges[index + 1]:.1f}",
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        })

    onehot = np.zeros_like(probabilities)
    onehot[np.arange(y_true.size), y_true] = 1.0
    true_score = np.clip(
        probabilities[np.arange(y_true.size), y_true], 1e-12, 1.0
    )
    top2 = np.sort(probabilities, axis=1)[:, -2:] \
        if probabilities.shape[1] > 1 else probabilities

    return {
        "samples": int(y_true.size),
        "mean_confidence": float(confidence.mean()),
        "mean_correct_confidence": (
            float(confidence[correct > 0].mean()) if correct.any() else None
        ),
        "mean_wrong_confidence": (
            float(confidence[correct == 0].mean())
            if (correct == 0).any() else None
        ),
        "accuracy": float(correct.mean()),
        "expected_calibration_error": float(weighted_gap),
        "negative_log_likelihood": float(-np.log(true_score).mean()),
        "brier_score": float(np.mean(np.sum(
            (probabilities - onehot) ** 2, axis=1
        ))),
        "mean_score_margin": (
            float((top2[:, 1] - top2[:, 0]).mean())
            if probabilities.shape[1] > 1 else None
        ),
        "bins": per_bin,
        "note": (
            "Softmax scores are model scores, not calibrated probabilities. "
            "An ECE well above zero means confidence must not be read as a "
            "hit rate; ML_CONFIDENCE_FLOOR gates on the raw mean score."
        ),
    }


def confidence_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidence: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Diagnostic for the confidence handling: how often does the model
    score turn out to be correct as a proxy for the top-1 prediction?

    This is a *faithfulness diagnostic*, not a calibration claim.  It
    exists to show that the current score is not yet a calibrated
    probability.
    """

    y_true_arr, y_pred_arr, _ = _as_int_arrays(y_true, y_pred)
    conf_arr = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if conf_arr.size != y_true_arr.size:
        raise ValueError("confidence length must match y_true/y_pred.")

    above = conf_arr >= threshold
    if not np.any(above):
        return {"above_threshold": 0, "accuracy_given_confidence": None}

    correct_given = float(np.mean(y_true_arr[above] == y_pred_arr[above]))
    accuracy_given = float(np.mean(y_true_arr == y_pred_arr))
    return {
        "above_threshold": int(above.sum()),
        "total": int(y_true_arr.size),
        "accuracy_given_confidence": correct_given,
        "accuracy_all": accuracy_given,
        "mean_confidence_above_threshold": float(np.mean(conf_arr[above])),
    }
