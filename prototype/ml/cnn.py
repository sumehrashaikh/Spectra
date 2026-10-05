"""NumPy-only inference for the converted modulation CNN.

The model (3x Conv1D/BN/MaxPool + a global head, input ``(L, 2)`` IQ
pairs, 16 softmax outputs) ships as a NumPy ``.npz`` artifact. No deep
learning framework is required at runtime, and the original pickle is
never loaded — pickles are untrusted input.

The runtime reads its own geometry from the weights instead of
assuming one architecture, so it executes both the historical
512-sample flatten-head model and the ML v3 1024-sample
global-average-pooled model from the same code path (see
``resolve_architecture``).

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

# Honesty gates.  An artifact declares its own holdout accuracy; below
# this the model is reported as *unvalidated evidence* instead of a
# prediction, because presenting a ~1-in-3 classifier as a label is how
# confident-but-wrong "predictions" reach the GUI.
ML_VALIDATION_FLOOR = 0.60
# A single capture also has to clear these before its class is presented
# as a prediction rather than evidence.
ML_CONFIDENCE_FLOOR = 0.55
ML_AGREEMENT_FLOOR = 0.50


# --------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------

def extract_frames(
    samples: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    max_frames: int = 8,
) -> np.ndarray:
    """Split a complex IQ stream into ``(num_frames, frame_length, 2)``.

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
    stride: int = 1,
) -> np.ndarray:
    """Valid convolution along time; x is (B, T, Cin).

    ``stride`` defaults to 1 (the historical behaviour).  The input is
    forced contiguous first: the windowed view below is built with
    ``as_strided``, which would read the wrong elements from a
    non-contiguous array.
    """

    stride = int(stride)
    if stride < 1:
        raise ValueError(f"conv stride must be >= 1, got {stride}")
    x = np.ascontiguousarray(x)
    batch, steps, _ = x.shape
    k, _, _channels = kernel.shape
    length = (steps - k) // stride + 1
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(batch, length, k, x.shape[2]),
        strides=(s[0], s[1] * stride, s[1], s[2]),
    )
    return np.tensordot(windows, kernel, axes=([2, 3], [0, 1])) + bias


def _maxpool(x: np.ndarray, size: int = 2) -> np.ndarray:
    """Non-overlapping max pooling along time, floor semantics.

    Only ``size=2`` is executed as a strided maximum; other sizes go
    through ``maximum.reduceat`` so an architecture can declare a
    different window without new code.
    """

    size = int(size)
    if size <= 1:
        return x
    keep = (x.shape[1] // size) * size
    even = x[:, :keep, :]
    if size == 2:
        return np.maximum(even[:, 0::2, :], even[:, 1::2, :])
    return even.reshape(even.shape[0], -1, size, even.shape[2]).max(axis=2)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    """MaxPool1D(pool=2, stride=2) with Keras floor semantics."""

    return _maxpool(x, 2)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------
# Architecture resolution
# --------------------------------------------------------------

@dataclass(frozen=True)
class BlockSpec:
    """One Conv1D -> BN -> ReLU -> (MaxPool) block, read from weights."""

    kernel: int
    in_channels: int
    out_channels: int
    stride: int = 1
    pool: int | None = 2


def resolve_architecture(
    tensors: dict[str, np.ndarray],
    config: dict,
) -> dict:
    """Work out how to run these weights, and check the config agrees.

    The *tensor shapes* are the source of truth for the layer geometry;
    the config only supplies what the weights cannot express (the conv
    stride, the pooling window, the input length and which global head
    to apply).  Two head layouts are supported:

    * ``flatten`` — the historical architecture: the whole temporal
      feature map goes into ``dense0``;
    * ``gap`` — global average pooling over time, then ``dense0``.  This
      is what ML v3 uses: it keeps the parameter count independent of
      the input length instead of scaling ``dense0`` with it.

    The head is *inferred* from ``dense0.kernel`` and then compared with
    the declared head; a disagreement raises instead of silently
    running a different network than the artifact claims to be.
    """

    architecture = config.get("architecture")
    declared_blocks: list[dict] = []
    declared_head: str | None = None
    if isinstance(architecture, dict):
        declared_blocks = list(architecture.get("blocks") or [])
        head = architecture.get("head") or []
        if head and isinstance(head[0], dict):
            first = str(head[0].get("layer", ""))
            if first in ("global_average_pooling1d", "global_average_pooling"):
                declared_head = "gap"
            elif first == "flatten":
                declared_head = "flatten"

    frame_length = int((config.get("input_shape") or [FRAME_LENGTH, 2])[0])
    if frame_length <= 0:
        raise ValueError(f"Artifact declares a bad input length {frame_length}")

    conv_ids = sorted(
        int(name[len("conv"):].split(".")[0])
        for name in tensors
        if name.startswith("conv") and name.endswith(".kernel")
    )
    if not conv_ids:
        raise ValueError("Artifact contains no conv kernels.")

    blocks: list[BlockSpec] = []
    for index, conv_id in enumerate(conv_ids):
        kernel, in_channels, out_channels = tensors[
            f"conv{conv_id}.kernel"
        ].shape
        spec = declared_blocks[index] if index < len(declared_blocks) else {}
        pool_spec = spec.get("pool", {"size": 2})
        pool = None if not pool_spec else int(pool_spec.get("size", 2))
        blocks.append(BlockSpec(
            kernel=int(kernel),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            stride=int(spec.get("conv_stride", 1)),
            pool=pool,
        ))

    length = frame_length
    for block in blocks:
        if length < block.kernel:
            raise ValueError(
                f"Input length {frame_length} is too short for a "
                f"{block.kernel}-tap convolution."
            )
        length = (length - block.kernel) // block.stride + 1
        if block.pool and block.pool > 1:
            length //= block.pool
    if length < 1:
        raise ValueError(
            f"Architecture collapses the input: {frame_length} samples "
            "leave no temporal positions."
        )

    last_channels = blocks[-1].out_channels
    # The head layout is identified by *key presence*: a hidden dense layer
    # exists exactly when the artifact carries ``dense0``.  An ML v3
    # ``GAP -> Dense -> softmax`` head has none, and the runtime must not
    # apply a ReLU to a layer that was never trained.
    hidden = "dense0.kernel" in tensors
    final_in = int(tensors["dense1.kernel"].shape[0])
    if hidden:
        dense_in = int(tensors["dense0.kernel"].shape[0])
        hidden_out = int(tensors["dense0.kernel"].shape[1])
    else:
        dense_in = final_in
        hidden_out = 0

    if dense_in == length * last_channels:
        head = "flatten"
    elif dense_in == last_channels:
        head = "gap"
    else:
        raise ValueError(
            f"The head expects {dense_in} inputs, which matches neither "
            f"flatten ({length} x {last_channels} = "
            f"{length * last_channels}) nor global average pooling "
            f"({last_channels}).  The artifact's input_shape/stride/pool "
            "metadata is inconsistent with its weights."
        )
    if declared_head is not None and declared_head != head:
        raise ValueError(
            f"Artifact declares a {declared_head!r} head but its weights "
            f"are shaped for {head!r}."
        )
    if hidden and hidden_out != final_in:
        raise ValueError(
            f"dense0 produces {hidden_out} features but dense1 expects "
            f"{final_in}."
        )
    if int(tensors["dense1.kernel"].shape[1]) != NUM_CLASSES:
        raise ValueError(
            f"dense1 must emit {NUM_CLASSES} classes, got "
            f"{tensors['dense1.kernel'].shape[1]}."
        )

    return {
        "frame_length": frame_length,
        "blocks": blocks,
        "head": head,
        "temporal_length": length,
        "hidden_dense": hidden,
        "head_input": dense_in,
    }


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

        # Geometry comes from the weights themselves, cross-checked
        # against the artifact's own declaration (see resolve_architecture).
        self.architecture = resolve_architecture(self._tensors, self.config)
        self.blocks: list[BlockSpec] = self.architecture["blocks"]

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
    def frame_length(self) -> int:
        """Samples per frame this artifact was built for (512 or 1024)."""

        return int(self.architecture["frame_length"])

    @property
    def head(self) -> str:
        """``"flatten"`` (historical) or ``"gap"`` (ML v3)."""

        return str(self.architecture["head"])

    @property
    def has_hidden_dense(self) -> bool:
        """True when ``dense0`` exists between the pooled features and the
        classifier — the historical topology but not the ML v3 one."""

        return bool(self.architecture["hidden_dense"])

    @property
    def trained(self) -> bool:
        """True only if the artifact records a completed training run."""
        return bool(self.config.get("trained", False))

    @property
    def validation_accuracy(self) -> float | None:
        """Holdout accuracy the artifact declares for itself (or None)."""
        training = self.config.get("training") or {}
        value = training.get("best_val_accuracy")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def validated(self) -> bool:
        """True only when the model is accurate enough to be *relied on*.

        Below ``ML_VALIDATION_FLOOR`` the runtime still runs the network
        (the scores are real evidence) but callers must not present its
        argmax as a classification result.
        """

        accuracy = self.validation_accuracy
        return bool(accuracy is not None and accuracy >= ML_VALIDATION_FLOOR)

    @property
    def training_summary(self) -> str:
        """Human-readable description of how/when the model was trained.

        Reads whichever epoch count the trainer recorded: ``ml.train``
        writes ``epochs``, ``ml.torch_train`` writes ``epochs_completed``
        (it also records what was requested, and a run that stopped early
        must not be described as if it had finished).
        """

        tr = self.config.get("training") or {}
        if not tr:
            return "not trained — this artifact is a weight conversion only"
        epochs = tr.get("epochs", tr.get("epochs_completed"))
        accuracy = tr.get("best_val_accuracy")
        accuracy_text = (
            f"{float(accuracy):.3f}" if accuracy is not None else "?"
        )
        return (
            f"{tr.get('frames_per_class', '?')} frames/class x "
            f"{epochs if epochs is not None else '?'} epochs, "
            f"best holdout {accuracy_text}, input {self.frame_length} samples"
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
        """Softmax scores for normalized ``(num_frames, frame_length, 2)``.

        Dimensions are read from the artifact, so the same runtime serves
        the 512-sample flatten-head model and the 1024-sample
        global-average-pooled model.
        """

        h = np.ascontiguousarray(frames, dtype=np.float32)
        if h.ndim != 3 or h.shape[2] != 2:
            raise ValueError(
                f"frames must be (N, {self.frame_length}, 2), got "
                f"{h.shape}"
            )
        if h.shape[1] != self.frame_length:
            raise ValueError(
                f"Artifact {self.artifact_name} expects "
                f"{self.frame_length}-sample frames; got {h.shape[1]}."
            )

        for idx, block in enumerate(self.blocks):
            scale = self._tensors[f"bn{idx}.gamma"] / np.sqrt(
                self._tensors[f"bn{idx}.variance"] + BN_EPSILON
            )
            kernel = self._tensors[f"conv{idx}.kernel"] * scale[None, None, :]
            bias = (
                (self._tensors[f"conv{idx}.bias"]
                 - self._tensors[f"bn{idx}.mean"]) * scale
                + self._tensors[f"bn{idx}.beta"]
            )
            h = np.maximum(
                _conv1d_valid(h, kernel, bias, stride=block.stride), 0.0
            )
            if block.pool and block.pool > 1:
                h = _maxpool(h, block.pool)

        if self.head == "gap":
            # Global average pooling: one descriptor per filter, so the
            # head's size does not grow with the input length.
            h = h.mean(axis=1)
        else:
            h = h.reshape(h.shape[0], -1)

        if self.has_hidden_dense:
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

        frames = extract_frames(
            samples, frame_length=self.frame_length, max_frames=max_frames
        )
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
                "input_shape": [self.frame_length, 2],
                "head": self.head,
                # The note has to match the gate this artifact actually
                # passed, or the GUI repeats a disclaimer about a model
                # that no longer applies to it.
                "note": (
                    "CNN scores are model scores, not calibrated "
                    "probabilities; this artifact's holdout accuracy "
                    "clears ML_VALIDATION_FLOOR, so a capture whose scores "
                    "also clear the confidence/agreement floors may be "
                    "presented as a prediction."
                    if self.validated
                    else (
                        "CNN scores are uncalibrated and this artifact's "
                        "holdout accuracy is below ML_VALIDATION_FLOOR; "
                        "treat them as evidence, not ground truth."
                    )
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

    Every payload carries ``presented_as``:

    * ``"prediction"`` - the artifact is validated and this capture's
      scores clear the confidence and frame-agreement floors;
    * ``"evidence"`` - the argmax is reported as a raw model score only.
      That covers an unvalidated artifact, a capture whose scores miss
      the floors, and a capture too short to form a frame (where
      ``num_frames`` is 0 and ``predicted_class`` is ``None``).

    The short-capture payload is deliberately *not* an error: it lets the
    caller distinguish "nothing available" from "capture too short", and
    the pipeline turns it into ``status="skipped"`` rather than "ok".
    """

    from prototype.ml.fusion import canonical_modulation, is_comparable

    engine = get_engine(labels_json)
    if engine is None:
        return None

    try:
        prediction = engine.predict_signal(samples)
    except ValueError:
        return {
            "predicted_class": None,
            "predicted_class_canonical": None,
            "confidence": 0.0,
            "frame_agreement": 0.0,
            "num_frames": 0,
            "frame_length": engine.frame_length,
            "labels_source": engine.labels_source,
            "artifact": engine.artifact_name,
            "model": "modulation_cnn (NumPy runtime)",
            "trained": engine.trained,
            "validated": engine.validated,
            "validation_accuracy": engine.validation_accuracy,
            "presented_as": "evidence",
            "low_confidence": True,
            "comparable": False,
            "status": engine.status,
            # This is the single source of the reason.  The pipeline and
            # the GUI compose their user-facing note from it, so the
            # wording must already say that the stage was skipped rather
            # than implying the model produced a verdict.
            "note": (
                "ML stage skipped inference: the capture is shorter than "
                f"one {engine.frame_length}-sample frame (the artifact's "
                "input length)."
            ),
            "inference_ran": False,
            "skip_reason": "insufficient_input",
        }

    top = np.argsort(prediction.mean_scores)[::-1][:3]
    canonical = canonical_modulation(prediction.predicted_class)
    confident = bool(
        prediction.confidence >= ML_CONFIDENCE_FLOOR
        and prediction.frame_agreement >= ML_AGREEMENT_FLOOR
    )
    presented_as = (
        "prediction" if (engine.validated and confident) else "evidence"
    )

    return {
        "model": "modulation_cnn (NumPy runtime)",
        "artifact": engine.artifact_name,
        "labels_source": engine.labels_source,
        "predicted_class": prediction.predicted_class,
        "predicted_class_canonical": canonical,
        "comparable": bool(is_comparable(canonical)),
        "confidence": prediction.confidence,
        "frame_agreement": prediction.frame_agreement,
        "num_frames": prediction.num_frames,
        "frame_length": engine.frame_length,
        "top3": [
            {
                "class": engine.labels[int(idx)],
                "class_canonical": canonical_modulation(
                    engine.labels[int(idx)]
                ),
                "score": float(prediction.mean_scores[int(idx)]),
            }
            for idx in top
        ],
        "trained": engine.trained,
        "validated": engine.validated,
        "validation_accuracy": engine.validation_accuracy,
        "presented_as": presented_as,
        "low_confidence": not confident,
        "status": engine.status,
        "labels_source": prediction.labels_source,
        "normalization": prediction.metadata["normalization"],
        "note": prediction.metadata["note"],
    }
