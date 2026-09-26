"""ML v1 reproducible dataset generation.

Extends the existing synthetic dataset with:

- deterministic per-file seeds so training/validation/test splits can
  be reproduced,
- an SNR-conditioned frame generator (train on a band of SNRs, eval on a
  held-out band),
- a clean serialization of a dataset to disk (NPZ) so training is
  reproducible from disk alone, and
- a leaky-safe splitter that operates on *realizations* rather than
  individual frames.

Nothing here touches the inference runtime (``ml/cnn.py``).  Inference
remains NumPy-only and is driven by a loaded artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from prototype.ml.dataset import (
    CLASS_NAMES,
    CLASS_INDEX,
    FrameConfig,
    build_dataset,
    generate_frame,
)

SAMPLE_RATE = 8000.0
FRAME_LENGTH = 512


@dataclass(frozen=True)
class DatasetFile:
    """Metadata describing one materialized dataset file on disk."""

    path: Path
    frames_per_class: int
    modulation_snr_db: float
    channel_snr_db: float
    cfo_hz: float
    phase_offset_rad: float
    symbol_rate: float
    rolloff: float
    seed: int
    n_frames: int
    n_classes: int


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class MLDataset:
    """A materialized, leak-safe, reproducible modulation dataset."""

    frames: np.ndarray
    labels: np.ndarray
    metadata: dict

    def save(self, path: str | Path) -> DatasetFile:
        """Write the frames/labels plus a human-readable metadata JSON."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            frames=self.frames.astype(np.float32),
            labels=self.labels.astype(np.int64),
        )
        meta = {
            **{k: v for k, v in self.metadata.items()},
            "n_frames": int(self.frames.shape[0]),
            "n_classes": int(self.labels.max()) + 1,
            "frame_length": int(self.frames.shape[1]),
            "n_q": int(self.frames.shape[2]),
            "dtype": str(self.frames.dtype),
            "labels": [str(c) for c in self.labels],
        }
        (out.with_suffix(".json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return DatasetFile(
            path=out,
            frames_per_class=self.metadata.get("frames_per_class", 0),
            modulation_snr_db=self.metadata.get("modulation_snr_db", 0.0),
            channel_snr_db=self.metadata.get("channel_snr_db", 0.0),
            cfo_hz=self.metadata.get("cfo_hz", 0.0),
            phase_offset_rad=self.metadata.get("phase_offset_rad", 0.0),
            symbol_rate=self.metadata.get("symbol_rate", 100.0),
            rolloff=self.metadata.get("rolloff", 0.35),
            seed=self.metadata.get("seed", 0),
            n_frames=meta["n_frames"],
            n_classes=meta["n_classes"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLDataset":
        """Load a materialized dataset (with its metadata)."""

        z = np.load(Path(path), allow_pickle=False)
        frames = z["frames"]
        labels = z["labels"]
        if "meta" not in z:
            raise ValueError(f"Dataset file {path} has no metadata.")
        meta = json.loads(z["meta"].item())
        return cls(frames=frames, labels=labels, metadata=meta)


def build_ml_dataset(
    *,
    frames_per_class: int = 40,
    modulation_snr_db: float = 12.0,
    channel_snr_db: float = 12.0,
    cfo_hz: float = 0.0,
    phase_offset_rad: float = 0.0,
    symbol_rate: float = 100.0,
    rolloff: float = 0.35,
    rng_seed: int = 0,
    classes: Sequence[str] | None = None,
) -> MLDataset:
    """Build a single-materialized ML dataset with fixed channel settings.

    This is the convenience entry point for a full training run.  Every
    stochastic parameter (SNR, CFO, phase, symbol rate, rolloff) is
    fixed per file so a second teammate can regenerate the exact same
    bytes by passing the same arguments.
    """

    cfg = FrameConfig(
        snr_db=modulation_snr_db,
        cfo_hz=cfo_hz,
        phase_offset_rad=phase_offset_rad,
        symbol_rate=symbol_rate,
        rolloff=rolloff,
        seed=rng_seed,
    )
    frames, labels = build_dataset(
        frames_per_class=frames_per_class, seed=rng_seed, classes=classes
    )
    return MLDataset(
        frames=frames.astype(np.float32),
        labels=labels.astype(np.int64),
        metadata={
            "frames_per_class": frames_per_class,
            "modulation_snr_db": modulation_snr_db,
            "channel_snr_db": channel_snr_db,
            "cfo_hz": cfo_hz,
            "phase_offset_rad": phase_offset_rad,
            "symbol_rate": symbol_rate,
            "rolloff": rolloff,
            "seed": rng_seed,
            "classes": list(classes or CLASS_NAMES),
            "sample_rate": SAMPLE_RATE,
            "frame_length": FRAME_LENGTH,
        },
    )


def snr_bands() -> list[float]:
    """The SNR sweep bands used for ML v1 evaluation.

    Train and validate on the clean-ish band; evaluate on the weak band
    too, so a confident-but-wrong model on low SNR is surfaced rather
    than hidden.
    """

    return [0.0, 6.0, 12.0, 18.0, 24.0]


def split_by_snr(
    frames: Sequence[MLDataset],
    *,
    train_snr_max: float = 18.0,
    eval_snr_min: float = 0.0,
    eval_snr_max: float = 24.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split multiple materialized datasets by SNR into train / eval.

    Frames with ``channel_snr_db <= train_snr_max`` go to training;
    frames with SNR inside the eval band go to eval.  This is the
    leakage-safe analogue of holding out a weak-signal band.
    """

    rng = np.random.default_rng(seed)

    train_frames: list[MLDataset] = []
    eval_frames: list[MLDataset] = []

    for ds in frames:
        snr = float(ds.metadata.get("channel_snr_db", ds.metadata.get("modulation_snr_db", 12.0)))
        if snr <= train_snr_max:
            train_frames.append(ds)
        if eval_snr_min <= snr <= eval_snr_max:
            eval_frames.append(ds)

    # Shuffle each fold deterministically before materializing.
    rng.shuffle(train_frames)
    rng.shuffle(eval_frames)

    return {
        "train": np.array([f.path for f in train_frames], dtype=object),
        "eval": np.array([f.path for f in eval_frames], dtype=object),
    }
