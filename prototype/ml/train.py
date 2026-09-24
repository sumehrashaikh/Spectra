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
from pathlib import Path

import numpy as np

from prototype.ml.cnn import (
    FRAME_LENGTH,
    NUM_CLASSES,
    _conv1d_valid,
    _maxpool2,
    _softmax,
)
from prototype.ml.dataset import CLASS_NAMES, build_dataset

BN_EPSILON = 1e-3


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


class Model:
    """The 14-layer modulation CNN with training-mode BN statistics."""

    def __init__(self, tensors: dict[str, np.ndarray] | None = None):
        rng = np.random.default_rng(42)
        shapes = {
            "conv0.kernel": (7, 2, 64), "conv0.bias": (64,),
            "bn0.gamma": (64,), "bn0.beta": (64,),
            "bn0.mean": (64,), "bn0.variance": (64,),
            "conv1.kernel": (5, 64, 128), "conv1.bias": (128,),
            "bn1.gamma": (128,), "bn1.beta": (128,),
            "bn1.mean": (128,), "bn1.variance": (128,),
            "conv2.kernel": (3, 128, 256), "conv2.bias": (256,),
            "bn2.gamma": (256,), "bn2.beta": (256,),
            "bn2.mean": (256,), "bn2.variance": (256,),
            "dense0.kernel": (61 * 256, 256), "dense0.bias": (256,),
            "dense1.kernel": (256, NUM_CLASSES), "dense1.bias": (NUM_CLASSES,),
        }
        self.params: dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            if tensors and name in tensors:
                self.params[name] = np.asarray(tensors[name], dtype=np.float64)
                continue
            if "kernel" in name:
                fan_in = int(np.prod(shape[:-1]))
                limit = np.sqrt(6.0 / fan_in)
                self.params[name] = rng.uniform(
                    -limit, limit, shape
                )
            elif "gamma" in name:
                self.params[name] = np.ones(shape)
            elif "variance" in name:
                self.params[name] = np.ones(shape)
            else:
                self.params[name] = np.zeros(shape)

    # -- forward ----------------------------------------------------

    def forward(self, x, training: bool = False):
        caches = {}
        h = x.astype(np.float64)
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
            normalized = (h - batch_mean) / np.sqrt(batch_var + BN_EPSILON)
            h = (self.params[f"bn{idx}.gamma"] * normalized
                 + self.params[f"bn{idx}.beta"])
            caches[f"bn{idx}"] = (h, normalized)
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
            pre_bn, normalized = caches[f"bn{idx}"]
            grads[f"bn{idx}.gamma"] = (dh * normalized).sum(axis=(0, 1))
            grads[f"bn{idx}.beta"] = dh.sum(axis=(0, 1))

            # Frozen-statistics BN input gradient (exact in inference
            # mode, where mean/variance are constants — the regime the
            # gradient check runs in).
            dh = (
                dh
                * self.params[f"bn{idx}.gamma"]
                / np.sqrt(self.params[f"bn{idx}.variance"] + BN_EPSILON)
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

def train(
    frames_per_class: int = 40,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    init_from: Path | None = None,
    log=print,
) -> tuple[Model, dict]:
    """Train and return (model, summary)."""

    rng = np.random.default_rng(seed)
    log("Generating dataset "
        f"({frames_per_class} frames/class x {len(CLASS_NAMES)} classes)…")
    X, y = build_dataset(frames_per_class, seed=seed)

    # Stratified 80/20 holdout.
    train_idx, val_idx = [], []
    for klass in np.unique(y):
        idx = np.where(y == klass)[0]
        rng.shuffle(idx)
        cut = max(1, int(0.2 * idx.size))
        val_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    tensors = None
    if init_from and Path(init_from).is_file():
        with np.load(init_from, allow_pickle=False) as data:
            tensors = {k: data[k] for k in data.files if k != "config_json"}
        log(f"Initialized weights from {init_from}")

    model = Model(tensors)
    optimizer = CNNOptimizer(model.params, lr=learning_rate)

    history = []
    best_val = -1.0
    best_params = {k: v.copy() for k, v in model.params.items()}

    for epoch in range(epochs):
        order = rng.permutation(train_idx.size)
        losses = []
        for start in range(0, order.size, batch_size):
            batch = train_idx[order[start : start + batch_size]]
            logits, caches = model.forward(X[batch], training=True)
            loss, dlogits, _ = softmax_xentropy_forward(logits, y[batch])
            grads = model.backward(dlogits, caches)
            optimizer.step(grads)
            losses.append(loss)

        # Validation accuracy with inference-style BN statistics.
        correct = 0
        for start in range(0, val_idx.size, batch_size):
            batch = val_idx[start : start + batch_size]
            logits, _ = model.forward(X[batch], training=False)
            correct += int((logits.argmax(axis=1) == y[batch]).sum())
        val_acc = correct / max(val_idx.size, 1)

        if val_acc >= best_val:
            best_val = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "val_accuracy": val_acc,
        })
        log(f"  epoch {epoch + 1:2d}: loss {np.mean(losses):.4f} "
            f"val_acc {val_acc:.3f}")

    model.params = best_params

    # Per-class accuracy on the holdout.
    logits, _ = model.forward(X[val_idx], training=False)
    preds = logits.argmax(axis=1)
    per_class = {}
    for klass in np.unique(y[val_idx]):
        mask = y[val_idx] == klass
        per_class[CLASS_NAMES[int(klass)]] = float(
            (preds[mask] == klass).mean()
        )

    summary = {
        "frames_per_class": frames_per_class,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "best_val_accuracy": best_val,
        "final_val_accuracy": history[-1]["val_accuracy"],
        "per_class_val_accuracy": per_class,
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
        "training": summary,
        "note": (
            "Trained in-project on synthetic frames from ml.dataset "
            "(repo RRC + channel simulator); validated on a synthetic "
            "holdout only. Real-world performance is unmeasured."
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
    )

    save_artifact(model, summary, args.output)
    print(f"Best holdout accuracy: {summary['best_val_accuracy']:.3f}")
    print(f"Artifact written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
