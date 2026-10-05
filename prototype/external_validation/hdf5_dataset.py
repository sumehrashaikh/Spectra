"""Loader and inspector for the optional external real-world IQ dataset.

The dataset is a third-party, off-air capture set (7 modulations, clean and
multipath channels, 20-30 dB SNR) that SPECTRA uses only as an *external
validation input*. This module reads it; it never converts it into training
data and never mutates anything in the repository.

``h5py`` is imported lazily so a checkout without it keeps working: the
functions here report the dependency as unavailable instead of raising at
import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Candidate locations, relative to the project root. The folder originally
# lived at ``correlation/dataset``; it may still be found there.
DEFAULT_DATASET_DIRS: tuple[str, ...] = ("dataset", "correlation/dataset")

CONFIG_NAME = "dataset.config.json"

# Fallbacks used when neither the config file nor the HDF5 attributes are
# available (never guessed silently: the source of each value is reported).
FALLBACK_FRAME_LENGTH = 1024
FALLBACK_SAMPLE_RATE_HZ = 2.0e6
FALLBACK_CLASSES: tuple[str, ...] = (
    "BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM",
)
CHANNEL_NAMES: tuple[str, ...] = ("clean", "multipath")


def h5py_available() -> bool:
    """True when the optional ``h5py`` dependency can be imported."""

    import importlib.util

    return importlib.util.find_spec("h5py") is not None


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415 - deliberately lazy
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The external dataset feature needs the optional 'h5py' "
            "dependency: pip install h5py"
        ) from exc
    return h5py


def resolve_dataset_dir(explicit: str | Path | None = None) -> Path | None:
    """Locate the dataset directory (explicit path wins, then defaults)."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        project_root = Path(__file__).resolve().parent.parent
        candidates.extend(project_root / name for name in DEFAULT_DATASET_DIRS)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


@dataclass
class SampledFrame:
    """One dataset frame selected by the sampling plan."""

    subset: str
    index: int
    modulation: str
    modulation_id: int
    channel: str
    channel_id: int
    snr_db: int
    iq: np.ndarray  # (frame_length, 2) float32

    @property
    def samples(self) -> np.ndarray:
        """Frame as complex64 IQ samples."""

        return (self.iq[:, 0].astype(np.float32)
                + 1j * self.iq[:, 1].astype(np.float32))


@dataclass
class DatasetSubset:
    """Metadata for one HDF5 subset file."""

    name: str
    path: Path
    num_frames: int
    frame_length: int
    x_dtype: str
    chunks: Any = None
    compression: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "exists": self.path.is_file(),
            "num_frames": int(self.num_frames),
            "frame_length": int(self.frame_length),
            "x_dtype": self.x_dtype,
            "chunks": self.chunks,
            "compression": self.compression,
        }


@dataclass
class RealWorldIqDataset:
    """Read-only view over the external real-world IQ dataset.

    Parameters
    ----------
    dataset_dir:
        Folder holding the ``subset_*.h5`` files. When omitted the default
        locations are searched (``dataset/`` then ``correlation/dataset/``).
    """

    dataset_dir: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    subsets: dict[str, DatasetSubset] = field(default_factory=dict)
    class_names: tuple[str, ...] = FALLBACK_CLASSES
    channel_names: tuple[str, ...] = CHANNEL_NAMES
    frame_length: int = FALLBACK_FRAME_LENGTH
    sample_rate_hz: float = FALLBACK_SAMPLE_RATE_HZ
    metadata_sources: dict[str, str] = field(default_factory=dict)

    # -- construction ------------------------------------------------

    def __post_init__(self) -> None:
        if self.dataset_dir is not None:
            self.dataset_dir = Path(self.dataset_dir)
            self._load_config()
            self._discover_subsets()
            self._read_hdf5_attributes()

    @classmethod
    def load(cls, dataset_dir: str | Path | None = None) -> "RealWorldIqDataset":
        return cls(dataset_dir=resolve_dataset_dir(dataset_dir))

    # -- discovery ---------------------------------------------------

    def _load_config(self) -> None:
        config_path = self.dataset_dir / CONFIG_NAME if self.dataset_dir else None
        if config_path and config_path.is_file():
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
            self.metadata_sources["config"] = str(config_path)
        else:
            self.config = {}

        frame_length = self.config.get("frame_length")
        if frame_length:
            self.frame_length = int(frame_length)
            self.metadata_sources["frame_length"] = "dataset.config.json"

        rate = self.config.get("sample_rate_hz")
        if rate:
            self.sample_rate_hz = float(rate)
            self.metadata_sources["sample_rate_hz"] = "dataset.config.json"

        classes = self.config.get("classes")
        if isinstance(classes, dict):
            mapped = [None] * len(classes)
            for key, value in classes.items():
                try:
                    mapped[int(key)] = str(value)
                except (TypeError, ValueError):
                    mapped = None
                    break
            if mapped and all(mapped):
                self.class_names = tuple(mapped)
                self.metadata_sources["class_names"] = "dataset.config.json"

        channels = self.config.get("channels")
        if isinstance(channels, dict):
            ordered = [None] * len(channels)
            for key, value in channels.items():
                try:
                    ordered[int(key)] = str(value)
                except (TypeError, ValueError):
                    ordered = None
                    break
            if ordered and all(ordered):
                self.channel_names = tuple(ordered)

    def _discover_subsets(self) -> None:
        declared = (self.config.get("files") or {})
        names = list(declared) or ["train", "val", "test"]
        for name in names:
            filename = declared.get(name, f"subset_{name}.h5")
            path = self.dataset_dir / filename
            if not path.is_file():
                continue
            self.subsets[name] = DatasetSubset(
                name=name, path=path, num_frames=0,
                frame_length=self.frame_length, x_dtype="unknown",
            )

    def _read_hdf5_attributes(self) -> None:
        """Fill in shape/attrs from the HDF5 files when available."""

        if not self.subsets or not h5py_available():
            return
        h5py = _require_h5py()

        for name, subset in self.subsets.items():
            with h5py.File(subset.path, "r") as handle:
                dataset = handle["X"]
                subset.num_frames = int(dataset.shape[0])
                subset.frame_length = int(dataset.shape[1])
                subset.x_dtype = str(dataset.dtype)
                subset.chunks = dataset.chunks
                subset.compression = dataset.compression
                attrs = dict(handle.attrs)

            if "frame_length" not in self.metadata_sources:
                value = attrs.get("frame_len")
                if value is not None:
                    self.frame_length = int(value)
                    self.metadata_sources["frame_length"] = "hdf5 attrs"

            label_map = attrs.get("mod2id_json")
            if label_map and "class_names" not in self.metadata_sources:
                try:
                    self.class_names = _ordered_class_names(json.loads(label_map))
                    self.metadata_sources["class_names"] = "hdf5 attrs"
                except Exception:
                    pass

    # -- properties --------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self.subsets) and h5py_available()

    @property
    def unavailable_reason(self) -> str | None:
        if not h5py_available():
            return ("optional dependency 'h5py' is not installed "
                    "(pip install h5py)")
        if not self.subsets:
            return (f"no subset_*.h5 files found in {self.dataset_dir!s}; see "
                    "dataset/README.md")
        return None

    def label_for(self, modulation_id: int) -> str:
        if 0 <= modulation_id < len(self.class_names):
            return self.class_names[modulation_id]
        return f"class_{modulation_id}"

    def channel_for(self, channel_id: int) -> str:
        if 0 <= channel_id < len(self.channel_names):
            return self.channel_names[channel_id]
        return f"channel_{channel_id}"

    # -- inspection --------------------------------------------------

    def inspect(self, subsets: list[str] | None = None) -> dict[str, Any]:
        """Shape/dtype/label/balance report for the requested subsets."""

        selected = subsets or list(self.subsets)
        report: dict[str, Any] = {
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else None,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "frame_length": self.frame_length,
            "sample_rate_hz": self.sample_rate_hz,
            "class_names": list(self.class_names),
            "channel_names": list(self.channel_names),
            "metadata_sources": dict(self.metadata_sources),
            "subsets": {},
            "note": (
                "External real-world dataset (7 modulations, 2 channels, "
                "20-30 dB). Used as a validation INPUT only."
            ),
        }

        if not self.available:
            return report

        for name in selected:
            subset = self.subsets.get(name)
            if subset is None:
                report["subsets"][name] = {"available": False}
                continue
            report["subsets"][name] = self._inspect_subset(subset)
        return report

    def _inspect_subset(self, subset: DatasetSubset) -> dict[str, Any]:
        h5py = _require_h5py()
        with h5py.File(subset.path, "r") as handle:
            modulation_ids = np.asarray(handle["y_mod"][:])
            channel_ids = np.asarray(handle["y_chan"][:])
            snr_values = np.asarray(handle["y_snr"][:])
            x_shape = tuple(int(dim) for dim in handle["X"].shape)
            x_dtype = str(handle["X"].dtype)

        mod_counts = _value_counts(modulation_ids)
        channel_counts = _value_counts(channel_ids)
        snr_counts = _value_counts(snr_values)

        cells: dict[str, int] = {}
        for (mid, cid, snr), count in zip(
            zip(modulation_ids.tolist(), channel_ids.tolist(),
                snr_values.tolist()),
            [1] * modulation_ids.size,
        ):
            key = f"{self.label_for(mid)}|{self.channel_for(cid)}|{int(snr)}"
            cells[key] = cells.get(key, 0) + count

        counts = list(mod_counts.values())
        balance = (min(counts) / max(counts)) if counts and max(counts) else 0.0

        return {
            "available": True,
            "path": str(subset.path),
            "X_shape": list(x_shape),
            "X_dtype": x_dtype,
            "num_frames": int(modulation_ids.size),
            "chunks": subset.chunks,
            "compression": subset.compression,
            "modulation_labels": {
                self.label_for(int(mid)): int(count)
                for mid, count in sorted(mod_counts.items())
            },
            "channel_labels": {
                self.channel_for(int(cid)): int(count)
                for cid, count in sorted(channel_counts.items())
            },
            "snr_values_db": {
                int(snr): int(count) for snr, count in sorted(snr_counts.items())
            },
            "class_balance_ratio": round(float(balance), 4),
            "cells": dict(sorted(cells.items())),
        }

    # -- sampling ----------------------------------------------------

    def sample_frames(
        self,
        subsets: list[str] | tuple[str, ...] = ("test",),
        per_cell: int = 3,
        seed: int = 7,
        max_frames: int | None = None,
    ) -> list[SampledFrame]:
        """Sample up to ``per_cell`` frames per (modulation, channel, SNR) cell.

        Sampling is stratified so every modulation / channel / SNR
        combination is represented, which is what makes the per-group
        breakdowns readable. Frames are returned in a deterministic order
        for a given ``seed``.
        """

        if not self.available:
            raise RuntimeError(self.unavailable_reason or "dataset unavailable")

        h5py = _require_h5py()
        rng = np.random.default_rng(seed)
        selected: list[dict[str, Any]] = []

        for name in subsets:
            subset = self.subsets.get(name)
            if subset is None:
                continue
            with h5py.File(subset.path, "r") as handle:
                modulation_ids = np.asarray(handle["y_mod"][:])
                channel_ids = np.asarray(handle["y_chan"][:])
                snr_values = np.asarray(handle["y_snr"][:])

                for mid in np.unique(modulation_ids):
                    for cid in np.unique(channel_ids):
                        for snr in np.unique(snr_values):
                            mask = (
                                (modulation_ids == mid)
                                & (channel_ids == cid)
                                & (snr_values == snr)
                            )
                            available = np.flatnonzero(mask)
                            if available.size == 0:
                                continue
                            take = min(per_cell, int(available.size))
                            picks = rng.choice(available, size=take,
                                               replace=False)
                            for index in picks:
                                selected.append({
                                    "subset": name,
                                    "index": int(index),
                                    "modulation_id": int(mid),
                                    "channel_id": int(cid),
                                    "snr_db": int(snr),
                                })

                if max_frames is not None and len(selected) >= max_frames:
                    selected = selected[:max_frames]
                    break

            # Read the frames of this subset in one sorted pass: HDF5
            # fancy indexing is much faster (and chunk-friendlier) when
            # the row indices are ordered.
            indices = [item["index"] for item in selected
                       if item["subset"] == name]
            if not indices:
                continue
            order = np.argsort(indices)
            sorted_indices = np.asarray(indices, dtype=np.int64)[order]
            with h5py.File(subset.path, "r") as handle:
                blocks = np.asarray(handle["X"][sorted_indices], dtype=np.float32)
            by_index = {int(idx): k for k, idx in enumerate(sorted_indices)}
            for item in selected:
                if item["subset"] != name:
                    continue
                item["iq"] = blocks[by_index[item["index"]]]

        frames: list[SampledFrame] = []
        for item in selected:
            if "iq" not in item:
                continue
            frames.append(SampledFrame(
                subset=item["subset"],
                index=item["index"],
                modulation=self.label_for(item["modulation_id"]),
                modulation_id=item["modulation_id"],
                channel=self.channel_for(item["channel_id"]),
                channel_id=item["channel_id"],
                snr_db=item["snr_db"],
                iq=item["iq"],
            ))
        return frames

    def iter_frames(self, **kwargs: Any) -> Iterator[SampledFrame]:
        yield from self.sample_frames(**kwargs)


def load_dataset(dataset_dir: str | Path | None = None) -> RealWorldIqDataset:
    """Convenience constructor used by the CLI and tests."""

    return RealWorldIqDataset.load(dataset_dir)


# --------------------------------------------------------------
# helpers
# --------------------------------------------------------------

def _ordered_class_names(mapping: dict[str, Any]) -> tuple[str, ...]:
    """Turn ``{"BPSK": 0, ...}`` into an index-ordered tuple."""

    ordered = [None] * len(mapping)
    for label, index in mapping.items():
        ordered[int(index)] = str(label)
    if any(name is None for name in ordered):
        raise ValueError("label map has gaps")
    return tuple(ordered)


def _value_counts(values: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}
