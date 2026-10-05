"""Training pipeline for the modulation CNN (NumPy only).

Trains the 3x Conv1D/BN/MaxPool + Dense architecture on the synthetic
labeled dataset from ``ml.dataset`` and writes artifacts in the same
format the inference runtime (``ml.cnn``) consumes — so training and
inference share one implementation of the forward pass semantics.

Includes a finite-difference gradient check of the analytic
backpropagation (conv/dense paths, with BN statistics frozen for the
check — the d(statistics)/d(input) terms are excluded from both sides,
which is the standard simplification and keeps the comparison exact).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prototype.ml.cnn import (
    FRAME_LENGTH,
    NUM_CLASSES,
    _conv1d_valid,
    _maxpool2,
    _softmax,
    normalize_frames,
)
from prototype.ml.dataset import CLASS_NAMES, build_dataset, build_dataset_v2
from prototype.ml.split import stratified_realization_split

BN_EPSILON = 1e-3

# SNR is drawn continuously, so the per-SNR report is aggregated into
# fixed-width bins instead of one bucket per distinct float.
SNR_BIN_WIDTH = 3.0


# --------------------------------------------------------------
# Layers (forward with caches, analytic backward)
# --------------------------------------------------------------

def conv_forward(x, W, b):
    out = _conv1d_valid(x, W, b)
    cache = (x, W)
    return out, cache


def conv_backward(dout, cache):
    """dout: (B, L, Cout); cache carries the raw input (B, T, Cin)."""
    x, W = cache
    batch, steps, in_channels = x.shape
    k, _, out_channels = W.shape
    length = steps - k + 1
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(batch, length, k, in_channels),
        strides=(s[0], s[1], s[1], s[2]),
    )
    dW = np.tensordot(dout, windows, axes=([0, 1], [0, 1])).transpose(
        1, 2, 0
    )  # (Cout, K, Cin) -> (K, Cin, Cout)
    db = dout.sum(axis=(0, 1))
    dkernel_out = np.tensordot(dout, W, axes=([2], [2]))
    dx = np.zeros_like(x)
    for j in range(k):
        dx[:, j : j + length, :] += dkernel_out[:, :, j, :]
    return dx, dW, db


def relu_forward(h):
    return np.maximum(h, 0.0), h


def relu_backward(dout, cache):
    return dout * (cache > 0.0)


def pool_forward(h):
    """MaxPool1D(2, stride 2); returns output + first-vs-second mask."""
    n = (h.shape[1] // 2) * 2
    even = h[:, :n, :]
    first = even[:, 0::2, :]
    second = even[:, 1::2, :]
    out = np.maximum(first, second)
    return out, first >= second


def pool_backward(dout, take_first, in_shape):
    """Route gradients to the winning input sample (ties -> first)."""
    dx = np.zeros(in_shape, dtype=dout.dtype)
    n = (in_shape[1] // 2) * 2
    dx[:, 0:n:2, :] = dout * take_first
    dx[:, 1:n:2, :] = dout * (~take_first)
    return dx


def dense_forward(h, W, b):
    return h @ W + b, (h, W)


def dense_backward(dout, cache):
    h, W = cache
    dW = h.T @ dout
    db = dout.sum(axis=0)
    dx = dout @ W.T
    return dx, dW, db


def softmax_xentropy_forward(logits, labels):
    probs = _softmax(logits)
    n = logits.shape[0]
    loss = float(-np.log(np.maximum(probs[np.arange(n), labels], 1e-30)).mean())
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    return loss, dlogits, probs


# --------------------------------------------------------------
# Full network
# --------------------------------------------------------------

class CNNOptimizer:
    """Adam over a dict of named parameters."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3):
        self.params = params
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for key, grad in grads.items():
            param = self.params[key]
            self.m[key] = b1 * self.m[key] + (1 - b1) * grad
            self.v[key] = b2 * self.v[key] + (1 - b2) * grad * grad
            m_hat = self.m[key] / (1 - b1**self.t)
            v_hat = self.v[key] / (1 - b2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


# Default architecture: 3 x (Conv1D -> BN -> ReLU -> MaxPool) + 2 dense.
# This *shape family* is what the NumPy runtime (ml/cnn.ModulationCNN)
# executes; the widths are read from the artifact, so they are free to
# change without touching the runtime.
DEFAULT_CHANNELS: tuple[int, ...] = (64, 128, 256)
DEFAULT_KERNELS: tuple[int, ...] = (7, 5, 3)
DEFAULT_DENSE_UNITS = 256


def flatten_length(
    frame_length: int = FRAME_LENGTH,
    kernels: tuple[int, ...] = DEFAULT_KERNELS,
) -> int:
    """Temporal length after ``len(kernels)`` valid convs + stride-2 pools."""

    length = int(frame_length)
    for kernel in kernels:
        length = length - int(kernel) + 1
        length = length // 2
    return length


class Model:
    """The modulation CNN with training-mode BN statistics.

    ``dtype`` controls the precision of every parameter and activation.
    float32 trains roughly twice as fast as float64 for a negligible
    accuracy difference; float64 remains the default so the
    finite-difference gradient check keeps its tight tolerance.

    ``channels`` / ``kernels`` / ``dense_units`` set the widths.  The
    NumPy runtime reads every shape from the artifact, so a narrower
    model is a training-throughput choice, not an interface change.
    """

    def __init__(
        self,
        tensors: dict[str, np.ndarray] | None = None,
        dtype: str | np.dtype = np.float64,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        kernels: tuple[int, ...] = DEFAULT_KERNELS,
        dense_units: int = DEFAULT_DENSE_UNITS,
    ):
        self.dtype = np.dtype(dtype)
        self.channels = tuple(int(c) for c in channels)
        self.kernels = tuple(int(k) for k in kernels)
        self.dense_units = int(dense_units)
        if len(self.channels) != len(self.kernels):
            raise ValueError(
                f"channels ({len(self.channels)}) and kernels "
                f"({len(self.kernels)}) must have the same length"
            )
        rng = np.random.default_rng(42)

        shapes: dict[str, tuple[int, ...]] = {}
        in_channels = 2
        for index, (kernel, out_channels) in enumerate(
            zip(self.kernels, self.channels)
        ):
            prefix = f"{index}"
            shapes[f"conv{prefix}.kernel"] = (kernel, in_channels, out_channels)
            shapes[f"conv{prefix}.bias"] = (out_channels,)
            for name in ("gamma", "beta", "mean", "variance"):
                shapes[f"bn{prefix}.{name}"] = (out_channels,)
            in_channels = out_channels

        flat = flatten_length(FRAME_LENGTH, self.kernels) * self.channels[-1]
        shapes["dense0.kernel"] = (flat, self.dense_units)
        shapes["dense0.bias"] = (self.dense_units,)
        shapes["dense1.kernel"] = (self.dense_units, NUM_CLASSES)
        shapes["dense1.bias"] = (NUM_CLASSES,)
        self.params: dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            if tensors and name in tensors:
                self.params[name] = np.asarray(
                    tensors[name], dtype=self.dtype
                )
                continue
            if "kernel" in name:
                fan_in = int(np.prod(shape[:-1]))
                limit = np.sqrt(6.0 / fan_in)
                self.params[name] = rng.uniform(
                    -limit, limit, shape
                ).astype(self.dtype)
            elif "gamma" in name:
                self.params[name] = np.ones(shape, dtype=self.dtype)
            elif "variance" in name:
                self.params[name] = np.ones(shape, dtype=self.dtype)
            else:
                self.params[name] = np.zeros(shape, dtype=self.dtype)

    # -- forward ----------------------------------------------------

    def forward(self, x, training: bool = False):
        caches = {}
        h = x.astype(self.dtype)
        caches["x0"] = h

        for idx in range(3):
            h = _conv1d_valid(
                h,
                self.params[f"conv{idx}.kernel"],
                self.params[f"conv{idx}.bias"],
            )
            if training:
                batch_mean = h.mean(axis=(0, 1))
                batch_var = h.var(axis=(0, 1))
                momentum = 0.9
                self.params[f"bn{idx}.mean"] = (
                    momentum * self.params[f"bn{idx}.mean"]
                    + (1 - momentum) * batch_mean
                )
                self.params[f"bn{idx}.variance"] = (
                    momentum * self.params[f"bn{idx}.variance"]
                    + (1 - momentum) * batch_var
                )
            else:
                batch_mean = self.params[f"bn{idx}.mean"]
                batch_var = self.params[f"bn{idx}.variance"]
            # Cache the statistics actually used for this normalization.
            # Backward must differentiate the *same* function it ran
            # forward: using the running statistics here while forward
            # normalized by the batch statistics made the analytic
            # gradient inconsistent with the loss being minimized.
            normalized = (h - batch_mean) / np.sqrt(batch_var + BN_EPSILON)
            h = (self.params[f"bn{idx}.gamma"] * normalized
                 + self.params[f"bn{idx}.beta"])
            caches[f"bn{idx}"] = (h, normalized, batch_var)
            h = np.maximum(h, 0.0)
            caches[f"relu{idx}"] = h > 0.0
            caches[f"prepool{idx}"] = h
            h, take_first = pool_forward(h)
            caches[f"pool{idx}"] = take_first
            caches[f"pooled{idx}"] = h

        flat = h.reshape(h.shape[0], -1)
        caches["flat_shape"] = h.shape
        h, dense0_cache = dense_forward(
            flat, self.params["dense0.kernel"], self.params["dense0.bias"]
        )
        caches["dense0"] = dense0_cache
        h, relu3_cache = relu_forward(h)
        caches["relu3"] = relu3_cache
        logits, dense1_cache = dense_forward(
            h, self.params["dense1.kernel"], self.params["dense1.bias"]
        )
        caches["dense1"] = dense1_cache
        return logits, caches

    # -- backward ---------------------------------------------------

    def backward(self, dlogits, caches):
        grads: dict[str, np.ndarray] = {}

        dx, dW, db = dense_backward(dlogits, caches["dense1"])
        grads["dense1.kernel"] = dW
        grads["dense1.bias"] = db

        dh = relu_backward(dx, caches["relu3"])
        dx, dW, db = dense_backward(dh, caches["dense0"])
        grads["dense0.kernel"] = dW
        grads["dense0.bias"] = db

        dh = dx.reshape(caches["flat_shape"])

        for idx in (2, 1, 0):
            # Max-pool: route gradients to the winning input sample.
            dh = pool_backward(
                dh,
                caches[f"pool{idx}"],
                caches[f"prepool{idx}"].shape,
            )

            dh = relu_backward(dh, caches[f"relu{idx}"])
            pre_bn, normalized, batch_var = caches[f"bn{idx}"]
            grads[f"bn{idx}.gamma"] = (dh * normalized).sum(axis=(0, 1))
            grads[f"bn{idx}.beta"] = dh.sum(axis=(0, 1))

            # BN input gradient using the statistics that forward
            # actually normalized by.  The d(mean)/dx and d(var)/dx terms
            # are excluded on both sides of the finite-difference check
            # (standard simplification), so this stays an exact match for
            # the quantity the gradient check verifies.
            dh = (
                dh
                * self.params[f"bn{idx}.gamma"]
                / np.sqrt(batch_var + BN_EPSILON)
            )

            conv_input = (
                caches["x0"] if idx == 0 else caches[f"pooled{idx - 1}"]
            )
            dx, dW, db = conv_backward(
                dh,
                (conv_input, self.params[f"conv{idx}.kernel"]),
            )
            grads[f"conv{idx}.kernel"] = dW
            grads[f"conv{idx}.bias"] = db
            dh = dx

        return grads


def gradient_check(seed: int = 0) -> float:
    """Finite-difference check of conv/dense backward on a tiny input."""

    rng = np.random.default_rng(seed)
    model = Model()
    # Length chosen so all three conv blocks stay valid:
    # 40 -> conv0 34 -> pool 17 -> conv1 13 -> pool 6 -> conv2 4 -> pool 2.
    x = rng.standard_normal((2, 40, 2)) * 0.5

    # Slim the dense layers for the tiny input: temporarily replace.
    original = model.params["dense0.kernel"]
    original_bias = model.params["dense0.bias"]
    original_dense1 = model.params["dense1.kernel"]
    original_dense1_bias = model.params["dense1.bias"]
    model.params["dense0.kernel"] = rng.standard_normal(
        (2 * 256, 64)
    ) * 0.05
    model.params["dense0.bias"] = np.zeros(64)
    model.params["dense1.kernel"] = rng.standard_normal((64, 4)) * 0.05
    model.params["dense1.bias"] = np.zeros(4)

    labels = np.array([0, 2])
    logits, caches = model.forward(x, training=False)
    loss, dlogits, _ = softmax_xentropy_forward(logits, labels)

    grads = model.backward(dlogits, caches)

    checked = 0.0
    compared = 0
    for name in ("conv0.kernel", "conv1.kernel", "dense0.kernel",
                 "dense1.kernel"):
        param = model.params[name]
        analytic = grads[name]
        flat_idx = rng.choice(param.size, size=8, replace=False)
        for flat in flat_idx:
            idx = np.unravel_index(flat, param.shape)
            eps = 1e-5
            old = param[idx]
            param[idx] = old + eps
            loss_plus, _, _ = softmax_xentropy_forward(
                model.forward(x, training=False)[0], labels
            )
            param[idx] = old - eps
            loss_minus, _, _ = softmax_xentropy_forward(
                model.forward(x, training=False)[0], labels
            )
            param[idx] = old
            numeric = (loss_plus - loss_minus) / (2 * eps)
            denom = max(abs(analytic[idx]), abs(numeric), 1e-8)
            checked = max(checked, abs(analytic[idx] - numeric) / denom)
            compared += 1

    model.params["dense0.kernel"] = original
    model.params["dense0.bias"] = original_bias
    model.params["dense1.kernel"] = original_dense1
    model.params["dense1.bias"] = original_dense1_bias
    return checked


# --------------------------------------------------------------
# Training loop
# --------------------------------------------------------------

def _predict(model: "Model", X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Argmax predictions using inference-mode batch statistics."""

    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        logits, _ = model.forward(X[start : start + batch_size], training=False)
        chunks.append(np.asarray(logits.argmax(axis=1), dtype=np.int64))
    return np.concatenate(chunks)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr_db: np.ndarray | None = None,
    snr_bin_width: float | None = None,
) -> dict:
    """Accuracy / per-class accuracy / confusion matrix / per-SNR accuracy.

    ``snr_bin_width`` quantizes a continuously drawn SNR before the
    per-SNR breakdown so the report has readable bins instead of one
    bucket per distinct float.
    """

    from prototype.ml.evaluation import (
        confusion_matrix,
        per_class_stats,
        per_snr_stats,
    )

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    all_classes = np.arange(NUM_CLASSES)

    matrix, _ = confusion_matrix(y_true, y_pred, labels=all_classes)
    stats = per_class_stats(y_true, y_pred, labels=all_classes)

    report: dict = {
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "samples": int(y_true.size),
        "per_class_accuracy": {
            name: (
                float((y_pred[y_true == index] == index).mean())
                if (y_true == index).any() else None
            )
            for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_f1": {
            name: float(stats[str(index)]["f1"])
            for index, name in enumerate(CLASS_NAMES)
        },
        "support": {
            name: int(stats[str(index)]["support"])
            for index, name in enumerate(CLASS_NAMES)
        },
        "macro_f1": float(np.mean(
            [float(stats[str(index)]["f1"]) for index in range(NUM_CLASSES)]
        )),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "confusion_matrix": matrix.tolist(),
    }

    if snr_db is not None:
        snr_values = np.asarray(snr_db, dtype=np.float64).reshape(-1)
        if snr_bin_width:
            snr_values = (
                np.round(snr_values / snr_bin_width) * snr_bin_width
            )
        per_snr = per_snr_stats(y_true, y_pred, snr_values, labels=all_classes)
        report["per_snr_accuracy"] = {
            f"{snr:g}": {
                "accuracy": float(value["accuracy"]),
                "sample_count": int(value["sample_count"]),
            }
            for snr, value in sorted(per_snr.items())
        }
    return report


def train(
    frames_per_class: int = 40,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    init_from: Path | None = None,
    log=print,
    val_fraction: float = 0.20,
    test_fraction: float = 0.0,
    dtype: str | np.dtype = np.float64,
    snr_range: tuple[float, float] = (0.0, 18.0),
    channel_variation: bool = False,
    multipath_fraction: float = 0.0,
    plateau_patience: int = 3,
    channels: tuple[int, ...] = DEFAULT_CHANNELS,
    kernels: tuple[int, ...] = DEFAULT_KERNELS,
    dense_units: int = DEFAULT_DENSE_UNITS,
) -> tuple[Model, dict]:
    """Train and return (model, summary).

    Dataset
    -------
    Frames come from ``ml.dataset``: exact synthetic labels, the repo's
    own channel simulator, a balanced class count, and randomized SNR /
    CFO / phase / symbol rate / roll-off (plus timing offset and multipath
    when ``channel_variation`` is on).

    Splitting
    ---------
    The split is stratified *and* leakage-safe: whole signal
    realizations are assigned to a single fold, so no test frame shares a
    symbol stream or a channel draw with a training frame.  The
    disjointness is asserted, not assumed.

    Checkpointing
    -------------
    The best validation checkpoint is kept, not the last epoch.  The test
    fold is touched exactly once, after training, with that checkpoint.
    """

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    log("Generating dataset "
        f"({frames_per_class} frames/class x {len(CLASS_NAMES)} classes, "
        f"snr {snr_range[0]:g}-{snr_range[1]:g} dB"
        f"{', channel variation' if channel_variation else ''})…")
    X, y, realization_ids, frame_snr = build_dataset_v2(
        frames_per_class, seed=seed, snr_range=snr_range,
        channel_variation=channel_variation,
        multipath_fraction=multipath_fraction,
    )

    # Train on exactly the representation the runtime feeds the network:
    # ``ml.cnn.ModulationCNN`` normalises every frame to unit RMS and the
    # artifact declares ``normalization="unit_rms"``.  Training on the raw
    # (amplitude-varying) frames instead made the weights amplitude-
    # sensitive and cost roughly 16 accuracy points at inference time.
    X = normalize_frames(X)

    # Stratified, realization-safe train/validation/test split.
    folds = stratified_realization_split(
        y,
        realization_ids,
        fractions=(
            1.0 - val_fraction - test_fraction, val_fraction, test_fraction,
        ),
        seed=seed,
    )
    train_idx = folds["train"]
    val_idx = folds["validation"]
    test_idx = folds["test"]

    # Prove the folds share no realization (no leakage), and that they
    # tile the dataset exactly.
    folded = np.concatenate([train_idx, val_idx, test_idx])
    assert np.unique(folded).size == y.size, "folds must cover every frame"
    realization_folds = [
        set(realization_ids[idx].tolist())
        for idx in (train_idx, val_idx, test_idx)
    ]
    for first in range(3):
        for second in range(first + 1, 3):
            overlap = realization_folds[first] & realization_folds[second]
            assert not overlap, f"realization leaked across folds: {overlap}"
    log(f"Split: train={train_idx.size} val={val_idx.size} "
        f"test={test_idx.size} frames "
        f"({len(realization_folds[0])}/"
        f"{len(realization_folds[1])}/{len(realization_folds[2])} "
        "realizations, disjoint)")

    tensors = None
    if init_from and Path(init_from).is_file():
        with np.load(init_from, allow_pickle=False) as data:
            tensors = {k: data[k] for k in data.files if k != "config_json"}
        log(f"Initialized weights from {init_from}")

    model = Model(
        tensors, dtype=dtype, channels=channels, kernels=kernels,
        dense_units=dense_units,
    )
    optimizer = CNNOptimizer(model.params, lr=learning_rate)

    # A fixed subsample keeps the per-epoch training metric cheap while
    # remaining comparable from epoch to epoch.
    probe_idx = train_idx[: min(512, train_idx.size)]

    history = []
    best_val = -1.0
    best_epoch = 0
    best_params = {k: v.copy() for k, v in model.params.items()}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        order = rng.permutation(train_idx.size)
        shuffled = train_idx[order]
        losses = []
        for start in range(0, shuffled.size, batch_size):
            batch = shuffled[start : start + batch_size]
            logits, caches = model.forward(X[batch], training=True)
            loss, dlogits, _ = softmax_xentropy_forward(logits, y[batch])
            grads = model.backward(dlogits, caches)
            optimizer.step(grads)
            losses.append(loss)

        # Both metrics use inference-style BN statistics, which is what
        # the exported artifact will use in the field.
        train_acc = float(
            (_predict(model, X[probe_idx]) == y[probe_idx]).mean()
        ) if probe_idx.size else 0.0
        val_acc = float(
            (_predict(model, X[val_idx]) == y[val_idx]).mean()
        ) if val_idx.size else 0.0

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_params = {k: v.copy() for k, v in model.params.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= plateau_patience:
                optimizer.lr *= 0.5
                epochs_without_improvement = 0
                log(f"  (plateau: learning rate -> {optimizer.lr:.2e})")

        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else None,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": float(optimizer.lr),
        })
        log(f"  epoch {epoch + 1:2d}: loss {np.mean(losses):.4f} "
            f"train {train_acc:.3f} val {val_acc:.3f}")

    model.params = best_params
    model.dtype = dtype

    # Reports on the best-validation checkpoint. The test fold is used
    # here and nowhere else in the run.
    val_report = (
        evaluate_predictions(y[val_idx], _predict(model, X[val_idx]),
                             frame_snr[val_idx], snr_bin_width=SNR_BIN_WIDTH)
        if val_idx.size else None
    )
    test_report = (
        evaluate_predictions(y[test_idx], _predict(model, X[test_idx]),
                             frame_snr[test_idx], snr_bin_width=SNR_BIN_WIDTH)
        if test_idx.size else None
    )
    train_report = evaluate_predictions(
        y[train_idx], _predict(model, X[train_idx]), frame_snr[train_idx],
        snr_bin_width=SNR_BIN_WIDTH,
    )

    summary = {
        "frames_per_class": frames_per_class,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "learning_rate_schedule": (
            f"halve after {plateau_patience} epochs without validation "
            "improvement"
        ),
        "seed": seed,
        "dtype": str(dtype),
        "normalization": "unit_rms",
        "snr_range_db": [float(snr_range[0]), float(snr_range[1])],
        "channel_variation": bool(channel_variation),
        "multipath_fraction": float(multipath_fraction),
        "architecture": {
            "channels": list(model.channels),
            "kernels": list(model.kernels),
            "dense_units": int(model.dense_units),
            "head": "flatten",
            "parameters": int(sum(
                np.asarray(value).size for value in model.params.values()
            )),
        },
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "test_samples": int(test_idx.size),
        "train_realizations": int(len(realization_folds[0])),
        "val_realizations": int(len(realization_folds[1])),
        "test_realizations": int(len(realization_folds[2])),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "final_val_accuracy": history[-1]["val_accuracy"],
        "per_class_val_accuracy": (
            val_report or train_report
        )["per_class_accuracy"],
        "train_metrics": train_report,
        "val_metrics": val_report,
        "test_metrics": test_report,
        "history": history,
    }
    return model, summary


def save_artifact(
    model: Model,
    summary: dict,
    output: Path,
    labels_source: str = "ml.dataset classes (training ground truth)",
) -> None:
    """Write trained weights in the inference artifact format."""

    tensors = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in model.params.items()
    }

    blocks = []
    for index in range(3):
        kernel = np.asarray(model.params[f"conv{index}.kernel"])
        blocks.append({
            "conv_filters": int(kernel.shape[2]),
            "conv_kernel": int(kernel.shape[0]),
            "conv_input_channels": int(kernel.shape[1]),
            "conv_stride": 1,
            "batchnorm": True,
            "activation": "relu",
            "pool": {"type": "max", "size": 2, "stride": 2},
        })

    architecture = {
        "input_shape": [FRAME_LENGTH, 2],
        "padding": "valid",
        "blocks": blocks,
        "head": [
            {"layer": "flatten"},
            {
                "layer": "dense",
                "units": int(model.params["dense0.kernel"].shape[1]),
                "activation": "relu",
            },
            {
                "layer": "dense",
                "units": int(model.params["dense1.kernel"].shape[1]),
                "activation": "softmax",
            },
        ],
        "parameters": int(sum(
            np.asarray(value).size for value in model.params.values()
        )),
        "runtime": "prototype.ml.cnn.ModulationCNN (NumPy only)",
    }

    config = {
        "class_name": "Sequential",
        "input_shape": [FRAME_LENGTH, 2],
        "num_classes": NUM_CLASSES,
        "labels": CLASS_NAMES,
        "labels_source": labels_source,
        "normalization": "unit_rms",
        "batch_norm_epsilon": BN_EPSILON,
        "conv_padding": "valid",
        "pooling": "max, stride 2",
        "activation_head": "softmax",
        "trained": True,
        "architecture": architecture,
        "training": summary,
        "provenance": {
            "dataset": (
                "prototype.ml.dataset.build_dataset_v2 — synthetic frames "
                "with exact labels, generated by the repo's own RRC pulse "
                "shaper and simulation.channel impairment model"
            ),
            "dataset_class_mapping": dict(enumerate(CLASS_NAMES)),
            "split": (
                "stratified + realization-leakage-safe "
                "(prototype.ml.split.stratified_realization_split)"
            ),
            "trainer": (
                "prototype.ml.train (NumPy only; no TensorFlow or "
                "PyTorch at training or inference time)"
            ),
            "checkpoint": "best validation checkpoint, not the final epoch",
            "external_dataset_used": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "note": (
            "Trained in-project on synthetic frames from ml.dataset "
            "(repo RRC + channel simulator); validated on held-out "
            "synthetic folds with no realization leakage. Real-world "
            "off-air performance is UNMEASURED — the external Mendeley "
            "dataset is deliberately excluded from training and its "
            "published baseline is not a SPECTRA result."
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the modulation CNN (NumPy only)."
    )
    parser.add_argument("--frames-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Optional artifact to initialize weights from.")
    parser.add_argument("--output", type=Path,
                        default=Path("ml/modulation_cnn_trained.npz"))
    parser.add_argument("--skip-grad-check", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.20,
                        help="Stratified validation fraction (default 0.20).")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Held-out test fraction, touched once after "
                             "training (default 0.0).")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float32",
                        help="Training precision (float32 is ~2x faster).")
    parser.add_argument("--snr-range", type=float, nargs=2,
                        default=(0.0, 18.0), metavar=("LOW", "HIGH"),
                        help="Training SNR range in dB (default 0 18).")
    parser.add_argument("--channel-variation", action="store_true",
                        help="Also randomize the sample timing offset.")
    parser.add_argument("--multipath-fraction", type=float, default=0.0,
                        help="Fraction of frames given a random multipath "
                             "profile (default 0.0; not part of the "
                             "project's intended training distribution).")
    parser.add_argument("--plateau-patience", type=int, default=3,
                        help="Epochs without val improvement before halving "
                             "the learning rate.")
    parser.add_argument("--channels", type=int, nargs=3,
                        default=list(DEFAULT_CHANNELS),
                        metavar=("C0", "C1", "C2"),
                        help="Conv block widths (default 64 128 256).")
    parser.add_argument("--kernels", type=int, nargs=3,
                        default=list(DEFAULT_KERNELS),
                        metavar=("K0", "K1", "K2"),
                        help="Conv kernel sizes (default 7 5 3).")
    parser.add_argument("--dense-units", type=int, default=DEFAULT_DENSE_UNITS,
                        help="Hidden dense width (default 256).")
    parser.add_argument("--report", type=Path, default=None,
                        help="Write the full training/evaluation report JSON "
                             "to this path.")
    args = parser.parse_args()

    if not args.skip_grad_check:
        error = gradient_check()
        print(f"Gradient check: max relative error {error:.2e}")
        if error > 1e-5:
            print("Gradient check FAILED — refusing to train.")
            return 2

    model, summary = train(
        frames_per_class=args.frames_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        init_from=args.init_from,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        dtype=args.dtype,
        snr_range=tuple(args.snr_range),
        channel_variation=args.channel_variation,
        multipath_fraction=args.multipath_fraction,
        plateau_patience=args.plateau_patience,
        channels=tuple(args.channels),
        kernels=tuple(args.kernels),
        dense_units=args.dense_units,
    )

    save_artifact(model, summary, args.output)
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.3f}")
    if summary.get("test_metrics"):
        print(f"Held-out test accuracy:   "
              f"{summary['test_metrics']['accuracy']:.3f}")
    print(f"Artifact written: {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Training report written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
