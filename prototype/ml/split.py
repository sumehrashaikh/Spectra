"""Leakage-safe dataset splitting for the modulation CNN.

The core rule of ML v1 evaluation is that the *signal realization*
(underlying symbol sequence, underlying random channel draw) must not
appear in more than one of train / val / test.  Splitting by individual
frames is not leakage-safe, because two adjacent frames of the same
symbol stream share the same underlying symbols and the same channel
instance.

This module therefore splits by a *realization id* attached to every
frame.  A realization is defined as one call to
``generate_frame(class_name, FrameConfig(...))``, i.e. one raw
baseband waveform across which the channel is drawn once.  Adjacent
frames taken from the same realization are kept whole in a single fold.

Design
------
- ``generate_frame`` returns a ``LabeledFrame`` containing
  ``samples`` plus ``realization_id``.
- ``split_realizations`` groups frames by realization id, shuffles the
  distinct realization ids, and assigns whole realizations to train /
  val / test.
- ``frame_split`` gives back the per-frame indices for each fold, so
  dataset-level metrics can be computed exactly as before when the
  consumer wants a flat array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class LabeledFrame:
    """One labeled frame plus the realization it came from."""

    samples: np.ndarray
    label: int
    realization_id: int
    metadata: dict = field(default_factory=dict)

    def to_flat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.samples, np.array([self.label]), np.array([self.realization_id])


def _attach_realization_id(frames: Iterable[LabeledFrame], start_id: int = 0) -> list[LabeledFrame]:
    """Attach a unique realization id to every frame in order.

    Frames that shared a ``realization_id`` before this call are
    guaranteed to receive the same id after it (grouping by the input
    ordering).  In practice the caller owns the realization grouping.
    """
    frames_list = list(frames)
    for index, frame in enumerate(frames_list):
        frame.realization_id = start_id + index
    return frames_list


def stratified_realization_split(
    labels: np.ndarray,
    realization_ids: np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Stratified, leakage-safe train/validation/test split.

    Two properties are enforced at once, which are easy to lose with a
    naive per-frame split:

    1. **No leakage** — every frame of a realization lands in exactly one
       fold.  The unit of assignment is the realization, never the frame,
       so two frames sharing a symbol stream or a channel draw can never
       straddle the train/test boundary.
    2. **Stratification** — each fold keeps the class distribution of the
       whole set, so a balanced dataset stays balanced in every fold.

    Returns ``{"train": idx, "validation": idx, "test": idx}`` with
    ``np.ndarray`` of frame indices.  Folds are disjoint and their union
    is every frame in the input.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    realization_ids = np.asarray(realization_ids, dtype=np.int64).reshape(-1)
    if labels.size != realization_ids.size:
        raise ValueError(
            "labels and realization_ids must have the same length: "
            f"{labels.size} != {realization_ids.size}"
        )
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    rng = np.random.default_rng(seed)
    folds: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    names = ("train", "validation", "test")

    for klass in np.unique(labels):
        mask = labels == klass
        realizations = np.unique(realization_ids[mask])
        rng.shuffle(realizations)

        n = realizations.size
        train_cut = int(np.floor(fractions[0] * n))
        # Guarantee at least one realization per non-empty fold so tiny
        # datasets (and tests) never produce an empty split.
        if n >= 3:
            train_cut = max(1, min(train_cut, n - 2))
            validation_cut = train_cut + max(
                1, min(int(np.floor(fractions[1] * n)), n - train_cut - 1)
            )
        else:
            validation_cut = n

        chunks = (
            realizations[:train_cut],
            realizations[train_cut:validation_cut],
            realizations[validation_cut:],
        )
        for name, chunk in zip(names, chunks):
            for realization in chunk:
                folds[name].extend(
                    np.flatnonzero(realization_ids == realization).tolist()
                )

    return {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in folds.items()
    }


def split_realizations(
    frames: Sequence[LabeledFrame],
    *,
    train_fraction: float = 0.64,
    validation_fraction: float = 0.16,
    test_fraction: float = 0.20,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split frames so that whole signal realizations stay in one fold.

    The returned dict maps fold name to a list of frame indices.

    Raises ``ValueError`` if the requested fractions do not sum to
    1.0.

    The caller is responsible for providing a frame stream in which all
    frames belonging to one underlying signal realization carry the
    *same* ``realization_id``.  This helper does not invent a split;
    it shuffles the distinct realization ids and places whole
    realizations in one fold.

    Example
    -------
    >>> frames = [LabeledFrame(samples=np.zeros((32, 2)), label=0, realization_id=0), ...]
    >>> folds = split_realizations(frames, seed=7)
    >>> set(folds) == {"train", "validation", "test"}
    True
    """

    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-9:
        raise ValueError(
            f"Fractions must sum to 1.0, got {train_fraction=} {validation_fraction=} {test_fraction=}"
        )

    if not frames:
        return {"train": [], "validation": [], "test": []}

    # Group indices by realization id, preserving the order supplied.
    by_realization: dict[int, list[int]] = {}
    for index, frame in enumerate(frames):
        by_realization.setdefault(int(frame.realization_id), []).append(index)

    realization_ids = np.array(
        sorted(by_realization.keys()), dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(realization_ids)

    fold_sizes = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }

    # Build the fold index boundaries on the *realization count* so
    # each whole realization lands in exactly one fold.  Round the cut
    # points with integer arithmetic to avoid the float-rounding bug
    # shown in the regression: for small realization counts, e.g.
    # 4 realizations, 0.5*4 = 2.0 is exact, but for small counts the
    # round-then-cumsum path can produce an empty slice.  We instead
    # compute the cut indices as integers directly.
    n_realizations = len(realization_ids)

    def _cut(fraction: float, n: int) -> int:
        # Largest integer <= fraction*n.
        value = fraction * n
        if value == int(value):
            return int(value)
        return int(np.floor(value))

    train_cut = _cut(train_fraction, n_realizations)
    validation_cut = train_cut + _cut(validation_fraction, n_realizations)
    test_cut = n_realizations
    # When the fractional sum does not tile exactly, the last cut takes
    # the remainder so the folds always tile the realization ids.
    if test_cut < n_realizations and validation_cut >= test_cut:
        test_cut = n_realizations

    fold_names = ["train", "validation", "test"]

    def _cut(fraction: float, n: int) -> int:
        # Largest integer <= fraction*n.
        value = fraction * n
        if value == int(value):
            return int(value)
        return int(np.floor(value))

    train_cut = _cut(train_fraction, n_realizations)
    validation_cut = train_cut + _cut(validation_fraction, n_realizations)
    test_cut = n_realizations
    # When the fractional sum does not tile exactly, the last cut takes
    # the remainder so the folds always tile the realization ids.
    if test_cut < n_realizations and validation_cut >= test_cut:
        test_cut = n_realizations

    boundaries = [0, train_cut, validation_cut, test_cut]

    result: dict[str, list[int]] = {name: [] for name in fold_names}
    for name, start, end in zip(
        fold_names, boundaries[:-1], boundaries[1:]
    ):
        grouped: list[int] = []
        for rid in realization_ids[start:end]:
            grouped.extend(by_realization[int(rid)])
        result[name].extend(grouped)
        result[name].sort()

    # Deterministic ordering: sort each fold's indices so a flat
    # consumer sees reproducible output.
    for name in fold_names:
        result[name].sort()

    return result

