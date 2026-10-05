"""ML v3 training: a PyTorch temporal CNN exported to the NumPy runtime.

Why this module exists
----------------------
``ml.train`` trains the modulation CNN in pure NumPy.  It is honest and
dependency-free, but on CPU it is wall-clock-bound: ML v2 needed 16
epochs to reach train 0.580, and pushing epochs further on that
architecture overfit (train 0.96 / val 0.47) long before it closed the
gap.  The diagnosis was not "train longer" but "train more, on more
data, with a better-suited architecture".

ML v3 therefore trains in PyTorch and exports to the *same* ``.npz``
artifact format that ``ml.cnn.ModulationCNN`` already consumes.  Nothing
about the deployment path changes:

* the shipped runtime stays NumPy-only and dependency-free;
* this module lives behind the optional ``.venv-mltrain`` environment and
  is never imported by the pipeline, the CLI or the GUI;
* the exported artifact is validated against the NumPy runtime by
  ``ml.validation`` before it is ever considered for promotion.

What changed relative to ML v2
------------------------------
* **Input context 512 -> 1024 samples.**  At 8 kHz with an 80-160 sym/s
  task, 512 samples is only 5-10 symbols; the extra context is the single
  most direct fix for the within-family confusion seen in ML v2.
* **Global average pooling instead of flatten.**  Flattening a 250-step
  feature map into a dense layer is what made the old model 4.1M
  parameters, forced the narrow widths, and tied the parameter count to
  the input length.  Pooling over time keeps the head at ~2k parameters
  and lets the same network accept a longer window.
* **Large first kernel (65 taps at stride 4).**  A 3-tap convolution
  stack has a receptive field of ~26 samples here, i.e. well under one
  symbol; it can never see constellation structure.  A 65-tap stride-4
  stem is a learned filterbank at symbol scale, and the three blocks
  together tile the whole 1024-sample window (receptive field ~1021).
* **Best-validation checkpoint + early stopping + LR schedule**, so a
  long run cannot silently end on an overfit epoch.

Honesty
-------
This module does not touch the DSP classifier, the class labels, the
unsupported-class handling, or ``ML_VALIDATION_FLOOR``.  The external
Mendeley dataset is deliberately **not** used here: it is 7 coarse
classes at 2 MSps and 20-30 dB, and mixing it into a 16-class 8 kHz
model would be a domain change disguised as extra data.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from prototype.ml.calibration import reliability_report
from prototype.ml.cnn import BN_EPSILON, NUM_CLASSES, normalize_frames
from prototype.ml.dataset import CLASS_NAMES
from prototype.ml.split import stratified_realization_split

# The NumPy trainer's reporting helpers are pure NumPy, so the torch
# trainer reuses them instead of inventing a second metric definition.
# Two implementations of "accuracy" is how reports start to disagree.
from prototype.ml.train import SNR_BIN_WIDTH, evaluate_predictions

NOT_TRAINED_MESSAGE = (
    "ML v3 training needs the optional CPU-PyTorch environment.  Create it "
    "once with:\n"
    "    python -m venv .venv-mltrain\n"
    "    .venv-mltrain/Scripts/python -m pip install "
    "--index-url https://download.pytorch.org/whl/cpu torch numpy\n"
    "then run training with .venv-mltrain/Scripts/python.  The SPECTRA "
    "runtime itself never needs PyTorch — inference stays NumPy-only."
)


def require_torch():
    """Import torch, or explain exactly how to get it."""

    try:
        import torch  # noqa: PLC0415  (deliberately lazy)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(NOT_TRAINED_MESSAGE) from exc
    return torch


# --------------------------------------------------------------
# Architecture
# --------------------------------------------------------------

@dataclass
class ArchSpec:
    """Geometry of the ML v3 temporal CNN.

    Defaults are the "B" arm of the ML v3 ablation: a 1024-sample input,
    a 65-tap stride-4 stem, two further blocks, global average pooling,
    then a single linear classifier.

    ``head="flatten"`` reproduces the ML v2 topology so the ablation can
    compare architectures under one training loop instead of comparing
    two trainers.
    """

    frame_length: int = 1024
    channels: tuple[int, ...] = (32, 64, 128)
    kernels: tuple[int, ...] = (65, 17, 9)
    strides: tuple[int, ...] = (4, 1, 1)
    pools: tuple[int, ...] = (2, 2, 2)
    head: str = "gap"
    hidden_dense: int = 0
    dropout: float = 0.0
    bn_epsilon: float = BN_EPSILON

    def __post_init__(self) -> None:
        lengths = {len(self.channels), len(self.kernels),
                   len(self.strides), len(self.pools)}
        if len(lengths) != 1:
            raise ValueError(
                "channels/kernels/strides/pools must have equal length: "
                f"{lengths}"
            )
        if self.head not in ("gap", "flatten"):
            raise ValueError(f"Unknown head {self.head!r}")

    @property
    def blocks(self) -> int:
        return len(self.channels)

    @property
    def temporal_length(self) -> int:
        """Positions left after every conv and pool (0 if it collapses)."""

        length = int(self.frame_length)
        for kernel, stride, pool in zip(
            self.kernels, self.strides, self.pools
        ):
            if length < kernel:
                return 0
            length = (length - int(kernel)) // int(stride) + 1
            if pool and int(pool) > 1:
                length //= int(pool)
        return length

    @property
    def head_input(self) -> int:
        last = int(self.channels[-1])
        return last if self.head == "gap" else self.temporal_length * last

    @property
    def receptive_field(self) -> int:
        """Rough span, in input samples, that the last block can see."""

        size, stride = 1, 1
        for kernel, step, pool in zip(self.kernels, self.strides, self.pools):
            size = size + (int(kernel) - 1) * stride
            stride *= int(step)
            if pool and int(pool) > 1:
                size += (int(pool) - 1) * stride
                stride *= int(pool)
        return size

    def as_architecture(self) -> dict:
        """The declaration written into the artifact config."""

        blocks = []
        for index, (kernel, channel, stride, pool) in enumerate(zip(
            self.kernels, self.channels, self.strides, self.pools
        )):
            in_channels = 2 if index == 0 else int(self.channels[index - 1])
            blocks.append({
                "conv_filters": int(channel),
                "conv_kernel": int(kernel),
                "conv_input_channels": int(in_channels),
                "conv_stride": int(stride),
                "padding": "valid",
                "batchnorm": True,
                "activation": "relu",
                "pool": (
                    {"type": "max", "size": int(pool), "stride": int(pool)}
                    if pool and int(pool) > 1 else None
                ),
            })

        head: list[dict] = []
        if self.head == "gap":
            head.append({"layer": "global_average_pooling1d"})
        else:
            head.append({"layer": "flatten"})
        if self.hidden_dense:
            head.append({"layer": "dense", "units": int(self.hidden_dense),
                         "activation": "relu"})
            if self.dropout:
                head.append({"layer": "dropout", "rate": float(self.dropout)})
        head.append({"layer": "dense", "units": NUM_CLASSES,
                     "activation": "softmax"})
        return {
            "input_shape": [int(self.frame_length), 2],
            "padding": "valid",
            "blocks": blocks,
            "head": head,
            "head_type": self.head,
            "parameters": None,  # filled in by the exporter
            "runtime": "prototype.ml.cnn.ModulationCNN (NumPy only)",
        }


# --------------------------------------------------------------
# Model
# --------------------------------------------------------------

def build_model(spec: ArchSpec):
    """Construct the network.  Requires torch."""

    torch = require_torch()
    nn = torch.nn

    class TemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = 2
            for channel, kernel, stride, pool in zip(
                spec.channels, spec.kernels, spec.strides, spec.pools
            ):
                layers: list[Any] = [
                    nn.Conv1d(in_channels, int(channel), int(kernel),
                              stride=int(stride)),
                    nn.BatchNorm1d(int(channel), eps=spec.bn_epsilon),
                    nn.ReLU(inplace=True),
                ]
                if pool and int(pool) > 1:
                    layers.append(nn.MaxPool1d(int(pool), int(pool)))
                blocks.append(nn.Sequential(*layers))
                in_channels = int(channel)
            self.blocks = nn.ModuleList(blocks)

            # The head's input width depends on the pooling, not on the
            # last conv width: a flatten head consumes every temporal
            # position, a pooled head consumes one value per filter.
            head: list[Any] = []
            if spec.hidden_dense:
                head.append(nn.Linear(spec.head_input, int(spec.hidden_dense)))
                head.append(nn.ReLU(inplace=True))
                if spec.dropout:
                    head.append(nn.Dropout(float(spec.dropout)))
                head.append(nn.Linear(int(spec.hidden_dense), NUM_CLASSES))
            else:
                head.append(nn.Linear(spec.head_input, NUM_CLASSES))
            self.classifier = nn.Sequential(*head)

        def features(self, x):
            # Inputs arrive as (batch, samples, 2) — the layout the NumPy
            # artifact and the pipeline both use — while torch convolves
            # over (batch, channels, samples).
            x = x.transpose(1, 2)
            for block in self.blocks:
                x = block(x)
            if x.shape[2] != spec.temporal_length:
                raise ValueError(
                    f"Blocks produced {x.shape[2]} temporal positions but "
                    f"the head was sized for {spec.temporal_length}."
                )
            if spec.head == "gap":
                # Global average pooling over time: one descriptor per
                # filter, so the head does not grow with input length.
                x = x.mean(dim=2)
            else:
                # The NumPy runtime holds features as (batch, time,
                # channels) and flattens in that order; torch holds
                # (batch, channels, time).  Flattening torch's layout
                # directly would interleave the two axes differently and
                # silently reinterpret every trained weight, so the
                # transpose is required for the export to mean anything.
                x = x.transpose(1, 2).reshape(x.shape[0], -1)
            return x

        def forward(self, x):
            return self.classifier(self.features(x))

    model = TemporalCNN()
    return model


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------

@dataclass
class DatasetBundle:
    frames: np.ndarray            # (N, frame_length, 2) float32, unit RMS
    labels: np.ndarray
    realization_ids: np.ndarray
    snr_db: np.ndarray
    spec_frame_length: int


def build_bundle(
    frames_per_class: int,
    *,
    seed: int,
    spec: ArchSpec,
    snr_range: tuple[float, float],
    channel_variation: bool,
    timing_span_samples: int,
    fractional_timing: bool,
    multipath_fraction: float,
    cache_dir: Path | None = None,
    tag: str = "",
    log=print,
) -> DatasetBundle:
    """Generate (or reload) the training frames.

    The frames are normalized to unit RMS here, once, exactly as the
    NumPy runtime normalizes them at inference time.  Training on raw
    amplitude-varying frames instead cost ML v2 about 16 accuracy points
    because the runtime normalizes and the trainer did not.
    """

    from prototype.ml.dataset import build_dataset_v2

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"v3_{tag}_fpc{frames_per_class}_seed{seed}_L{spec.frame_length}"
            f"_t{timing_span_samples}_{'frac' if fractional_timing else 'int'}"
        )
        cached = cache_dir / f"{stem}.npz"
        if cached.is_file():
            log(f"Reusing cached frames: {cached.name}")
            with np.load(cached, allow_pickle=False) as data:
                return DatasetBundle(
                    frames=data["frames"],
                    labels=data["labels"],
                    realization_ids=data["realization_ids"],
                    snr_db=data["snr_db"],
                    spec_frame_length=int(data["frame_length"]),
                )

    started = time.time()
    log(
        f"Generating {frames_per_class * len(CLASS_NAMES)} frames "
        f"({frames_per_class}/class, {spec.frame_length} samples, "
        f"snr {snr_range[0]:g}-{snr_range[1]:g} dB, "
        f"timing span {timing_span_samples}"
        f"{', fractional' if fractional_timing else ''})…"
    )
    frames, labels, realization_ids, snr_db = build_dataset_v2(
        frames_per_class,
        seed=seed,
        snr_range=snr_range,
        channel_variation=channel_variation,
        multipath_fraction=multipath_fraction,
        frame_length=spec.frame_length,
        timing_span_samples=timing_span_samples,
        fractional_timing=fractional_timing,
    )
    frames = normalize_frames(frames)
    log(f"  generated in {time.time() - started:.1f}s")

    bundle = DatasetBundle(
        frames=frames, labels=labels, realization_ids=realization_ids,
        snr_db=snr_db, spec_frame_length=spec.frame_length,
    )
    if cache_dir is not None:
        np.savez(
            cache_dir / f"{stem}.npz",
            frames=bundle.frames, labels=bundle.labels,
            realization_ids=bundle.realization_ids, snr_db=bundle.snr_db,
            frame_length=np.int64(spec.frame_length),
        )
    return bundle


def make_folds(bundle: DatasetBundle, *, val_fraction: float, test_fraction: float,
               seed: int, log=print) -> dict[str, np.ndarray]:
    """Stratified, realization-safe split with the disjointness proved."""

    folds = stratified_realization_split(
        bundle.labels,
        bundle.realization_ids,
        fractions=(
            1.0 - val_fraction - test_fraction, val_fraction, test_fraction,
        ),
        seed=seed,
    )
    folded = np.concatenate([folds["train"], folds["validation"], folds["test"]])
    assert np.unique(folded).size == bundle.labels.size, "folds must cover all frames"

    realization_sets = [
        set(bundle.realization_ids[folds[name]].tolist())
        for name in ("train", "validation", "test")
    ]
    for first in range(3):
        for second in range(first + 1, 3):
            shared = realization_sets[first] & realization_sets[second]
            assert not shared, f"realization leaked across folds: {shared}"

    log(
        f"Split: train={folds['train'].size} val={folds['validation'].size} "
        f"test={folds['test'].size} frames "
        f"({len(realization_sets[0])}/{len(realization_sets[1])}/"
        f"{len(realization_sets[2])} disjoint realizations)"
    )
    return folds


# --------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------

def torch_probabilities(model, frames: np.ndarray, *, batch_size: int = 256,
                        device=None) -> np.ndarray:
    """Softmax probabilities in eval mode (running BN statistics)."""

    torch = require_torch()
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(frames[start : start + batch_size])
            )
            if device is not None:
                batch = batch.to(device)
            logits = model(batch)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not chunks:
        return np.zeros((0, NUM_CLASSES), dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


# --------------------------------------------------------------
# Training
# --------------------------------------------------------------

@dataclass
class TrainConfig:
    spec: ArchSpec = field(default_factory=ArchSpec)
    frames_per_class: int = 1500
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 2e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    plateau_patience: int = 3
    early_stopping_patience: int = 10
    grad_clip: float = 5.0
    seed: int = 7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    snr_range: tuple[float, float] = (0.0, 18.0)
    channel_variation: bool = True
    timing_span_samples: int = 8
    fractional_timing: bool = False
    multipath_fraction: float = 0.0
    threads: int = 0
    max_seconds: float = 0.0


def train(config: TrainConfig, *, checkpoint: Path | None = None,
          resume: bool = False, cache_dir: Path | None = None,
          tag: str = "", log=print) -> tuple[Any, dict]:
    """Train and return ``(model, summary)``.

    Stops gracefully when ``max_seconds`` elapses, writing a checkpoint so
    the run can continue in a later invocation (tool timeouts on this
    project are 600 s, while a useful run here takes longer).
    """

    torch = require_torch()
    nn = torch.nn

    if config.threads and config.threads > 0:
        torch.set_num_threads(int(config.threads))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))

    spec = config.spec
    if spec.temporal_length < 1:
        raise ValueError(
            f"Architecture collapses a {spec.frame_length}-sample input."
        )

    bundle = build_bundle(
        config.frames_per_class, seed=config.seed, spec=spec,
        snr_range=config.snr_range, channel_variation=config.channel_variation,
        timing_span_samples=config.timing_span_samples,
        fractional_timing=config.fractional_timing,
        multipath_fraction=config.multipath_fraction,
        cache_dir=cache_dir, tag=tag, log=log,
    )
    folds = make_folds(
        bundle, val_fraction=config.val_fraction,
        test_fraction=config.test_fraction, seed=config.seed, log=log,
    )

    train_idx, val_idx, test_idx = (
        folds["train"], folds["validation"], folds["test"]
    )
    X_train = bundle.frames[train_idx]
    y_train = bundle.labels[train_idx]
    X_val = bundle.frames[val_idx]
    y_val = bundle.labels[val_idx]

    model = build_model(spec)
    parameters = count_parameters(model)
    log(
        f"Model: {spec.blocks} blocks {spec.channels}/{spec.kernels} "
        f"stride {spec.strides} pool {spec.pools} head={spec.head} "
        f"({parameters:,} parameters, head input {spec.head_input}, "
        f"temporal length {spec.temporal_length}, "
        f"receptive field ~{spec.receptive_field} samples)"
    )

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs),
            eta_min=config.min_learning_rate,
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=config.plateau_patience,
        )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    history: list[dict] = []
    best_val = -1.0
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
    stopped_early = False
    time_exhausted = False

    if resume and checkpoint is not None and checkpoint.is_file():
        log(f"Resuming from {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_state"]
        epochs_without_improvement = int(state.get("stale", 0))

    started = time.time()

    def elapsed() -> float:
        return time.time() - started

    for epoch in range(start_epoch, config.epochs):
        model.train()
        # Seeded per epoch, not once per run: a run stopped by the time
        # budget and continued with --resume then sees exactly the same
        # shuffles as an uninterrupted run, so the two cannot be compared
        # unfairly.
        order = np.random.default_rng([config.seed, epoch]).permutation(
            train_idx.size
        )
        running_loss, seen = 0.0, 0
        for start in range(0, order.size, config.batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(X_train[order[start : start + config.batch_size]])
            )
            target = torch.from_numpy(
                np.ascontiguousarray(y_train[order[start : start + config.batch_size]])
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_loss += float(loss.detach()) * batch.shape[0]
            seen += int(batch.shape[0])

            if config.max_seconds and elapsed() > config.max_seconds:
                time_exhausted = True
                break
        if time_exhausted:
            log(
                f"  time budget {config.max_seconds:.0f}s reached mid-epoch "
                f"{epoch + 1}; checkpointing to continue later"
            )
            break

        # Validation in eval mode — running BN statistics, which is what
        # the exported artifact uses in the field.
        val_probabilities = torch_probabilities(model, X_val)
        val_accuracy = float(
            (val_probabilities.argmax(axis=1) == y_val).mean()
        ) if y_val.size else 0.0

        if scheduler is not None:
            if config.scheduler == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_accuracy)

        improved = val_accuracy > best_val
        if improved:
            best_val = val_accuracy
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch + 1,
            "loss": running_loss / max(seen, 1),
            "val_accuracy": val_accuracy,
            "learning_rate": lr_now,
            "seconds": round(elapsed(), 1),
        })
        log(
            f"  epoch {epoch + 1:3d}: loss {running_loss / max(seen, 1):.4f} "
            f"val {val_accuracy:.3f} lr {lr_now:.2e} "
            f"[{elapsed():.0f}s]{' *' if improved else ''}"
        )

        if (config.early_stopping_patience
                and epochs_without_improvement >= config.early_stopping_patience):
            log(
                f"  early stop: no validation improvement for "
                f"{epochs_without_improvement} epochs"
            )
            stopped_early = True
            break

    completed = not time_exhausted and not stopped_early
    if checkpoint is not None and (time_exhausted or not completed):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": len(history),
            "history": history,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "stale": epochs_without_improvement,
            "config": _config_payload(config),
        }, checkpoint)
        log(f"Checkpoint written: {checkpoint}")

    model.load_state_dict(best_state)
    model.eval()

    completed_epochs = len(history)
    summary = {
        "trainer": "prototype.ml.torch_train (PyTorch, CPU)",
        "frames_per_class": int(config.frames_per_class),
        "epochs_requested": int(config.epochs),
        "epochs_completed": int(completed_epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "min_learning_rate": float(config.min_learning_rate),
        "optimizer": config.optimizer,
        "weight_decay": float(config.weight_decay),
        "learning_rate_schedule": config.scheduler,
        "grad_clip": float(config.grad_clip),
        "seed": int(config.seed),
        "dtype": "float32",
        "normalization": "unit_rms",
        "snr_range_db": [float(config.snr_range[0]), float(config.snr_range[1])],
        "channel_variation": bool(config.channel_variation),
        "timing_span_samples": int(config.timing_span_samples),
        "fractional_timing": bool(config.fractional_timing),
        "multipath_fraction": float(config.multipath_fraction),
        "early_stopped": bool(stopped_early),
        "time_exhausted": bool(time_exhausted),
        "architecture": {
            "frame_length": int(spec.frame_length),
            "channels": [int(c) for c in spec.channels],
            "kernels": [int(k) for k in spec.kernels],
            "strides": [int(s) for s in spec.strides],
            "pools": [int(p) for p in spec.pools],
            "head": spec.head,
            "hidden_dense": int(spec.hidden_dense),
            "temporal_length": int(spec.temporal_length),
            "head_input": int(spec.head_input),
            "receptive_field_samples": int(spec.receptive_field),
            "parameters": int(parameters),
        },
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "test_samples": int(test_idx.size),
        "train_seconds": round(elapsed(), 1),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val),
        "history": history,
        "complete": bool(completed),
    }

    if completed:
        # The test fold is touched exactly once, here, with the best
        # validation checkpoint — never during model selection.
        train_probabilities = torch_probabilities(model, X_train)
        val_probabilities = torch_probabilities(model, X_val)
        test_probabilities = (
            torch_probabilities(model, bundle.frames[test_idx])
            if test_idx.size else np.zeros((0, NUM_CLASSES))
        )

        train_report = evaluate_predictions(
            y_train, train_probabilities.argmax(axis=1),
            bundle.snr_db[train_idx], snr_bin_width=SNR_BIN_WIDTH,
        )
        val_report = evaluate_predictions(
            y_val, val_probabilities.argmax(axis=1),
            bundle.snr_db[val_idx], snr_bin_width=SNR_BIN_WIDTH,
        ) if val_idx.size else None
        test_report = evaluate_predictions(
            bundle.labels[test_idx], test_probabilities.argmax(axis=1),
            bundle.snr_db[test_idx], snr_bin_width=SNR_BIN_WIDTH,
        ) if test_idx.size else None

        summary["train_metrics"] = train_report
        summary["val_metrics"] = val_report
        summary["test_metrics"] = test_report
        summary["per_class_val_accuracy"] = (
            val_report or train_report
        )["per_class_accuracy"]
        # Calibration is measured on the validation fold: the test fold is
        # reserved for the single final accuracy number.
        summary["calibration"] = {
            "validation": reliability_report(y_val, val_probabilities),
            "test": reliability_report(
                bundle.labels[test_idx], test_probabilities
            ) if test_idx.size else {},
        }
        summary["_probabilities"] = {
            "X_train": None,  # intentionally not retained
            "test": test_probabilities,
        }
    return model, summary


def _config_payload(config: TrainConfig) -> dict:
    payload = asdict(config)
    payload["spec"] = asdict(config.spec)
    return payload


# --------------------------------------------------------------
# Export
# --------------------------------------------------------------

def save_artifact(model, spec: ArchSpec, summary: dict, output: Path,
                  *, labels_source: str = "ml.dataset classes (training ground truth)",
                  extra_provenance: dict | None = None) -> dict:
    """Write PyTorch weights in the NumPy runtime's ``.npz`` format.

    Layout notes, because getting these wrong is silent:

    * torch stores a Conv1d weight as ``(out, in, k)``; the runtime wants
      ``(k, in, out)`` — hence the permute.
    * torch stores a Linear weight as ``(out, in)``; the runtime wants
      ``(in, out)``.
    * batch-norm parameters transfer directly, and the runtime fuses them
      into the convolution at load time.
    """

    torch = require_torch()

    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    tensors: dict[str, np.ndarray] = {}

    for index, block in enumerate(model.blocks):
        weight_key = f"blocks.{index}.0.weight"
        bias_key = f"blocks.{index}.0.bias"
        # (out, in, k) -> (k, in, out)
        tensors[f"conv{index}.kernel"] = np.transpose(
            state[weight_key], (2, 1, 0)
        ).astype(np.float32)
        tensors[f"conv{index}.bias"] = state[bias_key].astype(np.float32)
        for name, torch_name in (
            ("gamma", "weight"), ("beta", "bias"),
            ("mean", "running_mean"), ("variance", "running_var"),
        ):
            tensors[f"bn{index}.{name}"] = state[
                f"blocks.{index}.1.{torch_name}"
            ].astype(np.float32)

    classifier = [key for key in state if key.startswith("classifier.")]
    linear_keys = sorted(
        {key.split(".")[1] for key in classifier
         if key.endswith(".weight") and state[key].ndim == 2},
        key=int,
    )
    # A spec-faithful ``GAP -> Dense -> softmax`` head has a single linear
    # layer.  It is written as ``dense1`` with no ``dense0`` at all, which
    # is how the runtime tells the two head shapes apart — writing a dummy
    # ``dense0`` would make the runtime apply an untrained ReLU.
    if len(linear_keys) == 1:
        only = linear_keys[0]
        if int(state[f"classifier.{only}.weight"].shape[1]) != spec.head_input:
            raise ValueError(
                "Single-dense head expects "
                f"{spec.head_input} inputs, got "
                f"{state[f'classifier.{only}.weight'].shape[1]}."
            )
        tensors["dense1.kernel"] = np.transpose(
            state[f"classifier.{only}.weight"], (1, 0)
        ).astype(np.float32)
        tensors["dense1.bias"] = state[
            f"classifier.{only}.bias"
        ].astype(np.float32)
    else:
        first, second = linear_keys[0], linear_keys[-1]
        tensors["dense0.kernel"] = np.transpose(
            state[f"classifier.{first}.weight"], (1, 0)
        ).astype(np.float32)
        tensors["dense0.bias"] = state[
            f"classifier.{first}.bias"
        ].astype(np.float32)
        tensors["dense1.kernel"] = np.transpose(
            state[f"classifier.{second}.weight"], (1, 0)
        ).astype(np.float32)
        tensors["dense1.bias"] = state[
            f"classifier.{second}.bias"
        ].astype(np.float32)

    architecture = spec.as_architecture()
    architecture["parameters"] = int(
        sum(value.size for value in tensors.values())
    )

    config = {
        "class_name": "TemporalCNN",
        "input_shape": [int(spec.frame_length), 2],
        "num_classes": NUM_CLASSES,
        "labels": CLASS_NAMES,
        "labels_source": labels_source,
        "normalization": "unit_rms",
        "batch_norm_epsilon": float(spec.bn_epsilon),
        "conv_padding": "valid",
        "pooling": "max, non-overlapping",
        "activation_head": "softmax",
        "trained": True,
        "architecture": architecture,
        "training": {
            key: value for key, value in summary.items()
            if not key.startswith("_")
        },
        "provenance": {
            "dataset": (
                "prototype.ml.dataset.build_dataset_v2 (frame_length="
                f"{spec.frame_length}) — synthetic frames with exact labels, "
                "generated by the repo's own RRC pulse shaper and "
                "simulation.channel impairment model"
            ),
            "dataset_class_mapping": dict(enumerate(CLASS_NAMES)),
            "split": (
                "stratified + realization-leakage-safe "
                "(prototype.ml.split.stratified_realization_split); "
                "disjointness asserted, not assumed"
            ),
            "trainer": (
                "prototype.ml.torch_train — PyTorch "
                f"{_torch_version()} CPU training in the isolated "
                ".venv-mltrain environment; exported to this NumPy artifact"
            ),
            "runtime": (
                "prototype.ml.cnn.ModulationCNN (NumPy only, no PyTorch at "
                "inference time)"
            ),
            "checkpoint": "best validation checkpoint, not the final epoch",
            "external_dataset_used": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **(extra_provenance or {}),
        },
        "note": (
            "Trained in-project on synthetic frames from ml.dataset (repo "
            "RRC + channel simulator); validated on held-out synthetic folds "
            "with no realization leakage. Real-world off-air performance is "
            "UNMEASURED — the external Mendeley dataset is deliberately "
            "excluded from training and its published baseline is not a "
            "SPECTRA result."
        ),
    }

    tensors["config_json"] = np.frombuffer(
        json.dumps(config, indent=2).encode("utf-8"), dtype=np.uint8
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **tensors)
    output.with_suffix(".config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return config


def load_artifact_into_torch(artifact_path: Path | str, spec: ArchSpec | None = None):
    """Rebuild the PyTorch network from an exported artifact.

    This is the exact inverse of ``save_artifact``'s weight mapping, which
    is what makes the training-model/NumPy-runtime parity checkable long
    after the run that produced the artifact: no checkpoint, run state or
    original process is needed, only the artifact itself.

    Comparing ``torch(model from artifact)`` against ``ml.cnn`` then
    proves two separate things at once — that the exported tensors really
    are the trained weights (this round-trip), and that the NumPy runtime
    computes the same function as PyTorch (the forward comparison).
    """

    torch = require_torch()
    with np.load(artifact_path, allow_pickle=False) as data:
        tensors = {
            key: np.asarray(data[key])
            for key in data.files if key != "config_json"
        }
        config = json.loads(bytes(data["config_json"]).decode("utf-8"))

    if spec is None:
        architecture = config["architecture"]
        spec = ArchSpec(
            frame_length=int(config["input_shape"][0]),
            channels=tuple(b["conv_filters"] for b in architecture["blocks"]),
            kernels=tuple(b["conv_kernel"] for b in architecture["blocks"]),
            strides=tuple(b.get("conv_stride", 1) for b in architecture["blocks"]),
            pools=tuple(
                int((b.get("pool") or {}).get("size", 1))
                for b in architecture["blocks"]
            ),
            head=architecture.get("head_type", "gap"),
            hidden_dense=(
                int(tensors["dense0.kernel"].shape[1])
                if "dense0.kernel" in tensors else 0
            ),
        )

    model = build_model(spec)
    state = model.state_dict()
    for index in range(spec.blocks):
        # (k, in, out) -> (out, in, k)
        state[f"blocks.{index}.0.weight"] = torch.from_numpy(
            np.ascontiguousarray(
                np.transpose(tensors[f"conv{index}.kernel"], (2, 1, 0))
            )
        )
        state[f"blocks.{index}.0.bias"] = torch.from_numpy(
            np.ascontiguousarray(tensors[f"conv{index}.bias"])
        )
        for name, artifact_suffix in (
            ("weight", "gamma"), ("bias", "beta"),
            ("running_mean", "mean"), ("running_var", "variance"),
        ):
            state[f"blocks.{index}.1.{name}"] = torch.from_numpy(
                np.ascontiguousarray(tensors[f"bn{index}.{artifact_suffix}"])
            )

    # Locate the linear layers by index rather than assuming 0 and 2: a
    # dropout in the head would shift the classifier's position.
    linear_keys = sorted(
        {int(key.split(".")[1]) for key in state
         if key.startswith("classifier.") and key.endswith(".weight")}
    )
    if "dense0.kernel" in tensors:
        first, last = linear_keys[0], linear_keys[-1]
        state[f"classifier.{first}.weight"] = torch.from_numpy(
            np.ascontiguousarray(np.transpose(tensors["dense0.kernel"], (1, 0)))
        )
        state[f"classifier.{first}.bias"] = torch.from_numpy(
            np.ascontiguousarray(tensors["dense0.bias"])
        )
    else:
        last = linear_keys[-1]
    state[f"classifier.{last}.weight"] = torch.from_numpy(
        np.ascontiguousarray(np.transpose(tensors["dense1.kernel"], (1, 0)))
    )
    state[f"classifier.{last}.bias"] = torch.from_numpy(
        np.ascontiguousarray(tensors["dense1.bias"])
    )

    model.load_state_dict(state)
    model.eval()
    return model, spec


def _torch_version() -> str:
    try:
        import torch  # noqa: PLC0415
        return str(torch.__version__)
    except Exception:  # pragma: no cover
        return "unknown"


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def _spec_from_args(args) -> ArchSpec:
    return ArchSpec(
        frame_length=int(args.frame_length),
        channels=tuple(int(x) for x in args.channels),
        kernels=tuple(int(x) for x in args.kernels),
        strides=tuple(int(x) for x in args.strides),
        pools=tuple(int(x) for x in args.pools),
        head=args.head,
        hidden_dense=int(args.hidden_dense),
        dropout=float(args.dropout),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ML v3: train the temporal CNN in PyTorch and export a "
                    "NumPy-runtime artifact."
    )
    parser.add_argument("--frames-per-class", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"],
                        default="cosine")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--snr-range", type=float, nargs=2, default=(0.0, 18.0),
                        metavar=("LOW", "HIGH"))
    parser.add_argument("--no-channel-variation", action="store_true")
    parser.add_argument("--timing-span-samples", type=int, default=8,
                        help="Integer timing offset drawn from [0, N) samples.")
    parser.add_argument("--fractional-timing", action="store_true",
                        help="Also draw a sub-sample (fractional) delay.")
    parser.add_argument("--multipath-fraction", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=0,
                        help="Torch CPU threads (0 = torch default).")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Stop gracefully after this many seconds and "
                             "checkpoint for a later --resume.")

    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--kernels", type=int, nargs="+", default=[65, 17, 9])
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 1, 1])
    parser.add_argument("--pools", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--head", choices=["gap", "flatten"], default="gap")
    parser.add_argument("--hidden-dense", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--output", type=Path, default=Path("ml/_candidate_v3.npz"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("data/output/_mlv3_cache"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the model and time one batch, then stop.")
    args = parser.parse_args(argv)

    try:
        require_torch()
    except RuntimeError as error:
        print(error)
        return 3

    spec = _spec_from_args(args)
    config = TrainConfig(
        spec=spec,
        frames_per_class=args.frames_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        grad_clip=args.grad_clip,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        snr_range=tuple(args.snr_range),
        channel_variation=not args.no_channel_variation,
        timing_span_samples=args.timing_span_samples,
        fractional_timing=args.fractional_timing,
        multipath_fraction=args.multipath_fraction,
        threads=args.threads,
        max_seconds=args.max_seconds,
    )

    if args.dry_run:
        return _dry_run(config)

    model, summary = train(
        config, checkpoint=args.checkpoint, resume=args.resume,
        cache_dir=args.cache_dir, tag=args.tag,
    )

    if not summary["complete"]:
        print(
            f"Training did NOT finish ({summary['epochs_completed']}/"
            f"{summary['epochs_requested']} epochs). Re-run with --resume to "
            "continue; no artifact was written."
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        return 4

    save_artifact(model, spec, summary, args.output)
    # Keep the trained weights next to the artifact for traceability.  The
    # artifact alone is enough to re-derive this model
    # (load_artifact_into_torch), so this file is a convenience, not a
    # dependency of the runtime or of the parity check.
    import torch  # noqa: PLC0415  (already required above)
    torch.save(
        {"state_dict": model.state_dict(), "spec": asdict(spec),
         "architecture": summary["architecture"]},
        args.output.with_suffix(".torch.pt"),
    )
    print(f"best validation accuracy: {summary['best_val_accuracy']:.4f} "
          f"(epoch {summary['best_epoch']})")
    if summary.get("test_metrics"):
        print(f"held-out test accuracy:   "
              f"{summary['test_metrics']['accuracy']:.4f} "
              f"(macro-F1 {summary['test_metrics']['macro_f1']:.4f})")
    print(f"artifact written: {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
        )
        print(f"report written:   {args.report}")
    return 0


def _jsonable(summary: dict) -> dict:
    return {
        key: value for key, value in summary.items()
        if not key.startswith("_")
    }


def _dry_run(config: TrainConfig) -> int:
    """Time one forward/backward batch so a long run can be sized."""

    torch = require_torch()
    if config.threads:
        torch.set_num_threads(int(config.threads))
    model = build_model(config.spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(config.batch_size, config.spec.frame_length, 2)
    y = torch.randint(0, NUM_CLASSES, (config.batch_size,))

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimizer.step()

    steps = 10
    started = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimizer.step()
    per_step = (time.time() - started) / steps

    total = config.frames_per_class * NUM_CLASSES
    steps_per_epoch = total * (1.0 - config.val_fraction - config.test_fraction) / config.batch_size
    print(json.dumps({
        "torch": _torch_version(),
        "threads": int(torch.get_num_threads()),
        "parameters": count_parameters(model),
        "frame_length": config.spec.frame_length,
        "temporal_length": config.spec.temporal_length,
        "head_input": config.spec.head_input,
        "receptive_field_samples": config.spec.receptive_field,
        "seconds_per_step": round(per_step, 4),
        "seconds_per_sample": round(per_step / config.batch_size, 5),
        "steps_per_epoch": round(steps_per_epoch, 1),
        "estimated_seconds_per_epoch": round(per_step * steps_per_epoch, 1),
        "estimated_seconds_total": round(
            per_step * steps_per_epoch * config.epochs, 1
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
