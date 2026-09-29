"""NumPy-only inference for the converted modulation CNN.

The model (Keras 3 Sequential, 3x Conv1D/BN/MaxPool + Dense head,
input ``(512, 2)`` IQ pairs, 16 softmax outputs) ships as a NumPy
``.npz`` artifact produced by ``tools.convert_keras_pickle``. No deep
learning framework is required at runtime, and the original pickle is
never loaded — pickles are untrusted input.

Honesty rules (matching the rest of Spectra):

- The artifact carries no training-dataset or label-map metadata, so
  class names default to ``class_00`` .. ``class_15``. A real label
  map can be dropped in via ``labels.json`` (see ``ModulationCNN``)
  without touching the weights.
- The training preprocessing is unknown; the default per-frame RMS
  normalization is a configuration choice, not a fact. Confidence
  outputs are treated as *scores* until the model is validated against
  labeled data, and ``predict_frames`` reports them as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Architecture constants of the converted modulation CNN.
FRAME_LENGTH = 512
NUM_CLASSES = 16
BN_EPSILON = 1e-3


# --------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------

def extract_frames(
    samples: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    max_frames: int = 8,
) -> np.ndarray:
    """Split a complex IQ stream into ``(num_frames, 512, 2)`` windows.

    Windows are taken at the largest stride that still yields
    ``max_frames`` non-overlapping frames, so a long capture is
    represented across its whole duration instead of only its start.
    Short captures yield fewer frames; a capture shorter than one
    frame raises ``ValueError`` (the caller should skip the ML stage,
    not fabricate input).
    """

    samples = np.asarray(samples, dtype=np.complex128).reshape(-1)

    if samples.size < frame_length:
        raise ValueError(
            f"ML classifier needs at least {frame_length} samples "
            f"for one frame; capture has {samples.size}."
        )

    num_available = samples.size // frame_length
    num_frames = min(max_frames, num_available)

    stride = (
        (samples.size - frame_length) // max(num_frames - 1, 1)
        if num_frames > 1
        else 0
    )

    iq = np.empty((num_frames, frame_length, 2), dtype=np.float32)
    for k in range(num_frames):
        start = k * stride
        chunk = samples[start : start + frame_length]
        iq[k, :, 0] = chunk.real
        iq[k, :, 1] = chunk.imag

    return iq


def normalize_frames(
    frames: np.ndarray,
    mode: str = "unit_rms",
) -> np.ndarray:
    """Per-frame input normalization (a config choice, not a fact).

    ``unit_rms`` scales each frame so its total RMS magnitude is 1,
    making predictions amplitude-invariant — the usual convention for
    amplitude-agnostic modulation recognition.
    """

    if mode != "unit_rms":
        raise ValueError(
            f"Unknown normalization mode {mode!r} (supported: unit_rms)."
        )

    power = np.mean(frames.real ** 2 + frames.imag ** 2, axis=(1, 2),
                    keepdims=True)
    scale = 1.0 / np.sqrt(np.maximum(power, 1e-30))
    return frames * scale.astype(np.float32)


# --------------------------------------------------------------
# NumPy forward pass (weights fused with BatchNormalization)
# --------------------------------------------------------------

def _conv1d_valid(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Valid convolution along time; x is (B, T, Cin)."""

    batch, steps, _ = x.shape
    k, _, channels = kernel.shape
    length = steps - k + 1
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(batch, length, k, x.shape[2]),
        strides=(s[0], s[1], s[1], s[2]),
    )
    return np.tensordot(windows, kernel, axes=([2, 3], [0, 1])) + bias


def _maxpool2(x: np.ndarray) -> np.ndarray:
    """MaxPool1D(pool=2, stride=2) with Keras floor semantics."""

    even = x[:, : (x.shape[1] // 2) * 2, :]
    return np.maximum(even[:, 0::2, :], even[:, 1::2, :])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------

@dataclass
class MLPrediction:
    """One model prediction over a capture (scores, not facts)."""

    scores: np.ndarray                      # (num_frames, 16) softmax
    mean_scores: np.ndarray                 # (16,) averaged over frames
    predicted_class: str
    confidence: float                       # mean score of the winner
    frame_agreement: float                  # fraction of frames agreeing
    num_frames: int
    labels_source: str
    metadata: dict = field(default_factory=dict)


class ModulationCNN:
    """Load and run the converted CNN with NumPy only.

    Every artifact carries a ``config_json`` that records its own
    provenance: how it was converted, who trained it, on what data and
    under what assumptions. ``train.py`` fills in those fields; the
    conversion tool leaves them honest defaults.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        labels_json: str | Path | None = None,
        normalization: str = "unit_rms",
    ):
        artifact_path = Path(artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"ML artifact not found: {artifact_path}")

        with np.load(artifact_path, allow_pickle=False) as data:
            self._tensors = {
                key: np.asarray(data[key], dtype=np.float32)
                for key in data.files
                if key != "config_json"
            }
            self.config = json.loads(
                bytes(data["config_json"]).decode("utf-8")
            )

        self.labels = list(self.config.get("labels", []))
        self.labels_source = self.config.get("labels_source", "unknown")
        self.normalization = normalization
        self.artifact_name = Path(artifact_path).name

        # A labels.json beside the artifact (or passed explicitly)
        # overrides the neutral map without touching weights.
        override = Path(labels_json) if labels_json else (
            artifact_path.with_suffix(".labels.json")
        )
        if override.is_file():
            self.labels = [str(x) for x in json.loads(
                override.read_text(encoding="utf-8")
            )]
            self.labels_source = f"labels.json ({override.name})"

        if len(self.labels) != NUM_CLASSES:
            raise ValueError(
                f"Label map has {len(self.labels)} entries; "
                f"model expects {NUM_CLASSES}."
            )

    # -- properties -------------------------------------------------

    @property
    def is_available(self) -> bool:
        return True

    @property
    def trained(self) -> bool:
        """True only if the artifact records a completed training run."""
        return bool(self.config.get("trained", False))

    @property
    def training_summary(self) -> str:
        """Human-readable description of how/when the model was trained."""
        tr = self.config.get("training") or {}
        if not tr:
            return "not trained — this artifact is a weight conversion only"
        return (
            f"{tr.get('frames_per_class', '?')} frames/class x "
            f"{tr.get('epochs', '?')} epochs, best holdout "
            f"{tr.get('best_val_accuracy', '?'):.3f}"
        )

    @property
    def status(self) -> str:
        """One line humans can read instead of silent near-uniform scores."""
        if not self.trained:
            return (
                f"UNTRAINED ({self.training_summary}) — scores are "
                "near-random by design; treat as evidence only"
            )
        return f"trained: {self.training_summary}"

    def status_message(self) -> str:
        """First sentence suitable for a dialog/provenance entry."""
        if self.trained:
            return f"ML CNN ({self.artifact_name}): {self.status}"
        return (
            f"ML CNN ({self.artifact_name}): {self.status}. "
            "Disable the ML toggle or train the model before relying on it."
        )

    # -- inference --------------------------------------------------

    def predict_frames(self, frames: np.ndarray) -> np.ndarray:
        """Softmax scores for normalized ``(num_frames, 512, 2)`` input."""

        h = np.ascontiguousarray(frames, dtype=np.float32)

        for idx in range(3):
            scale = self._tensors[f"bn{idx}.gamma"] / np.sqrt(
                self._tensors[f"bn{idx}.variance"] + BN_EPSILON
            )
            kernel = self._tensors[f"conv{idx}.kernel"] * scale[None, None, :]
            bias = (
                (self._tensors[f"conv{idx}.bias"]
                 - self._tensors[f"bn{idx}.mean"]) * scale
                + self._tensors[f"bn{idx}.beta"]
            )
            h = np.maximum(_conv1d_valid(h, kernel, bias), 0.0)
            h = _maxpool2(h)

        h = h.reshape(h.shape[0], -1)
        h = np.maximum(
            h @ self._tensors["dense0.kernel"]
            + self._tensors["dense0.bias"],
            0.0,
        )
        logits = (
            h @ self._tensors["dense1.kernel"]
            + self._tensors["dense1.bias"]
        )
        return _softmax(logits)

    def predict_signal(
        self,
        samples: np.ndarray,
        max_frames: int = 8,
    ) -> MLPrediction:
        """Full pipeline: frame -> normalize -> predict -> summarize."""

        frames = extract_frames(samples, max_frames=max_frames)
        frames = normalize_frames(frames, mode=self.normalization)
        scores = self.predict_frames(frames)

        mean_scores = scores.mean(axis=0)
        winner = int(mean_scores.argmax())
        agreement = float(
            np.mean(scores.argmax(axis=1) == winner)
        )

        return MLPrediction(
            scores=scores,
            mean_scores=mean_scores,
            predicted_class=self.labels[winner],
            confidence=float(mean_scores[winner]),
            frame_agreement=agreement,
            num_frames=int(scores.shape[0]),
            labels_source=self.labels_source,
            metadata={
                "normalization": self.normalization,
                "labels_source": self.labels_source,
                "num_classes": NUM_CLASSES,
                "input_shape": [FRAME_LENGTH, 2],
                "note": (
                    "CNN scores are uncalibrated until the model is "
                    "validated against labeled captures; treat them "
                    "as evidence, not ground truth."
                ),
            },
        )


# --------------------------------------------------------------
# Module-level singleton (loaded lazily, once per process)
# --------------------------------------------------------------

_DEFAULT = Path(__file__).with_name("modulation_cnn.npz")
_TRAINED = Path(__file__).with_name("modulation_cnn_trained.npz")
_engine: ModulationCNN | None = None


def get_engine(labels_json: str | Path | None = None) -> ModulationCNN | None:
    """Return the shared engine (or None if nothing is available).

    A ``labels_json`` override builds a one-off engine carrying that label
    map; it is deliberately *not* cached in the process singleton so the
    default (artifact-defined) labels are never mutated by a caller.
    """

    if labels_json is not None:
        return _select_engine(labels_json)

    global _engine
    if _engine is None:
        _engine = _select_engine()
    return _engine


def _select_engine(labels_json: str | Path | None = None) -> ModulationCNN | None:
    """Choose the artifact priority and report why."""
    for candidate in (_TRAINED, _DEFAULT):
        if candidate.is_file():
            try:
                engine = ModulationCNN(candidate, labels_json=labels_json)
                engine.artifact_name = candidate.name
                return engine
            except Exception:
                # Corrupt artifact: move on, the next candidate is tried.
                continue
def predict_modulation(
    samples: np.ndarray, labels_json: str | Path | None = None
) -> dict | None:
    """Convenience API used by the pipeline's ML stage.

    Returns ``None`` only when nothing is available at all (no artifact,
    corrupt file, or capture too short). A *trained* artifact plus a
    short capture is a real signal that should be distinguished from a
    missing artifact, so the caller can warn the user differently.

    ``labels_json`` optionally overrides the artifact's class labels for
    this call only (used by the CLI ``--labels`` flag).
    """

    engine = get_engine(labels_json)
    if engine is None:
        return None

    try:
        prediction = engine.predict_signal(samples)
    except ValueError:
        return {
            "predicted_class": None,
            "confidence": 0.0,
            "frame_agreement": 0.0,
            "num_frames": 0,
            "labels_source": engine.labels_source,
            "artifact": engine.artifact_name,
            "model": "modulation_cnn (NumPy runtime)",
            "trained": engine.trained,
            "status": engine.status,
            "note": "Capture shorter than one 512-sample frame",
        }

    top = np.argsort(prediction.mean_scores)[::-1][:3]
    return {
        "model": "modulation_cnn (NumPy runtime)",
        "artifact": engine.artifact_name,
        "labels_source": engine.labels_source,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "frame_agreement": prediction.frame_agreement,
        "num_frames": prediction.num_frames,
        "top3": [
            {
                "class": engine.labels[int(idx)],
                "score": float(prediction.mean_scores[int(idx)]),
            }
            for idx in top
        ],
        "trained": engine.trained,
        "status": engine.status,
        "labels_source": prediction.labels_source,
        "normalization": prediction.metadata["normalization"],
        "note": prediction.metadata["note"],
    }
