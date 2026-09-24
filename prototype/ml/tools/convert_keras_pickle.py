"""Convert a pickled Keras 3 model into a NumPy artifact.

Safety model
------------
The pickle is **never unpickled**. Pickles can execute arbitrary code
when loaded; this tool instead walks the pickle stream with
``pickletools`` (opcodes only, no execution), extracts the embedded
``.keras`` payload (a plain ZIP containing ``config.json`` and
``model.weights.h5``), and reads it with ``h5py``. Only the layer
weights and architecture metadata cross over into the output.

Usage
-----
    python -m prototype.ml.tools.convert_keras_pickle \
        <model.pkl> <output.npz> [--config-json <output.json>]

The resulting ``.npz`` is consumed by
``prototype.ml.cnn.ModulationCNN`` (NumPy-only inference; TensorFlow
is not required at runtime). h5py is required only for conversion.
"""

from __future__ import annotations

import argparse
import io
import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------
# Pickle stream: locate the embedded .keras ZIP without executing
# --------------------------------------------------------------

def extract_embedded_keras_zip(pickle_path: Path) -> tuple[bytes, str]:
    """Return (zip_bytes, class_name) from a pickled Keras 3 model."""

    data = pickle_path.read_bytes()

    # Static opcode audit first: refuse files referencing anything
    # beyond the known Keras unpickling machinery.
    import pickletools as _pt

    allowed = {
        ("builtins", "getattr"),
        ("_io", "BytesIO"),
    }
    pairs: list[tuple[str, str]] = []
    strings: list[str] = []
    for op, arg, _pos in _pt.genops(io.BytesIO(data)):
        if op.name == "STOP":
            break
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":
            if len(strings) >= 2:
                pairs.append((strings[-2], strings[-1]))
            strings.clear()
        elif op.name in ("MEMOIZE", "FRAME", "PROTO", "BINGET",
                         "LONG_BINGET", "EMPTY_BINGET"):
            continue
        else:
            strings.clear()

    for module, name in pairs:
        if (module, name) not in allowed and not (
            module.startswith("keras.")
        ):
            raise ValueError(
                f"Pickle references unexpected global {module}.{name}; "
                "refusing to process (only Keras models are allowed)."
            )

    # The .keras payload is the largest bytes literal in the stream.
    blobs = []
    for op, arg, _pos in _pt.genops(io.BytesIO(data)):
        if isinstance(arg, (bytes, bytearray)) and len(arg) > 1000:
            blobs.append(bytes(arg))
    if not blobs:
        raise ValueError("No embedded model payload found in the pickle.")

    blob = max(blobs, key=len)
    if blob[:4] != b"PK\x03\x04":
        raise ValueError(
            "Embedded payload is not a .keras ZIP (legacy HDF5 models "
            "need a different converter)."
        )

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        config = json.loads(zf.read("config.json"))
        return blob, config.get("class_name", "Sequential")


# --------------------------------------------------------------
# Weight extraction from the .keras ZIP
# --------------------------------------------------------------

def load_model_definition(keras_zip: bytes):
    """Return (config, metadata, datasets) from a .keras ZIP."""

    with zipfile.ZipFile(io.BytesIO(keras_zip)) as zf:
        config = json.loads(zf.read("config.json"))
        metadata = json.loads(zf.read("metadata.json"))
        with io.BytesIO(zf.read("model.weights.h5")) as weights_io:
            import h5py

            with h5py.File(weights_io, "r") as h5:
                datasets: dict[str, np.ndarray] = {}

                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = np.asarray(obj)

                h5.visititems(_visit)

    return config, metadata, datasets


BN_EPSILON = 1e-3


def extract_weights(config: dict, datasets: dict) -> dict:
    """Map Keras layer configs + weights onto named engine tensors."""

    layers = config["config"]["layers"]

    convs = [L for L in layers if L["class_name"] == "Conv1D"]
    bns = [L for L in layers if L["class_name"] == "BatchNormalization"]
    denses = [L for L in layers if L["class_name"] == "Dense"]

    if len(convs) != 3 or len(denses) != 2:
        raise ValueError(
            "This converter expects the 3x Conv1D + 2x Dense "
            f"modulation CNN (found {len(convs)} conv, "
            f"{len(denses)} dense layers)."
        )

    tensors: dict[str, np.ndarray] = {}

    # Keras 3 stores each layer's variables at
    # ``layers/<layer_name>/vars/<i>``. Variable order per class
    # (verified against this file's shapes): Conv1D/Dense — 0=kernel,
    # 1=bias; BatchNormalization — 0=gamma, 1=beta, 2=moving_mean,
    # 3=moving_variance.
    def _var(layer_name: str, index: int) -> np.ndarray:
        key = f"layers/{layer_name}/vars/{index}"
        if key not in datasets:
            raise KeyError(f"Missing weight dataset {key!r}")
        return datasets[key]

    for idx, conv in enumerate(convs):
        name = conv["config"]["name"]
        tensors[f"conv{idx}.kernel"] = _var(name, 0)
        tensors[f"conv{idx}.bias"] = _var(name, 1)

    for idx, bn in enumerate(bns):
        name = bn["config"]["name"]
        tensors[f"bn{idx}.gamma"] = _var(name, 0)
        tensors[f"bn{idx}.beta"] = _var(name, 1)
        tensors[f"bn{idx}.mean"] = _var(name, 2)
        tensors[f"bn{idx}.variance"] = _var(name, 3)

    for idx, dense in enumerate(denses):
        name = dense["config"]["name"]
        tensors[f"dense{idx}.kernel"] = _var(name, 0)
        tensors[f"dense{idx}.bias"] = _var(name, 1)

    # Sanity: Dense0 input width must be divisible by the final conv
    # channel count (flatten timesteps x channels).
    flat_width = tensors["dense0.kernel"].shape[0]
    channels = tensors["conv2.kernel"].shape[2]
    if flat_width % channels != 0:
        raise ValueError(
            f"Dense input width {flat_width} is not compatible with "
            f"{channels} output channels."
        )

    return tensors


# --------------------------------------------------------------
# Engine self-check: fused-BN forward pass == textbook forward pass
# --------------------------------------------------------------

def _maxpool2(x):
    """MaxPool1D(pool=2, stride=2, valid): floor semantics — Keras
    DROPS a trailing odd-length leftover; naive [::2] keeps it."""
    even = x[:, :(x.shape[1] // 2) * 2, :]
    return np.maximum(even[:, 0::2, :], even[:, 1::2, :])


def _conv1d_valid(x, kernel, bias):
    B, T, Cin = x.shape
    K, _, Cout = kernel.shape
    L = T - K + 1
    s = x.strides
    win = np.lib.stride_tricks.as_strided(
        x, shape=(B, L, K, Cin), strides=(s[0], s[1], s[1], s[2])
    )
    return np.tensordot(win, kernel, axes=([2, 3], [0, 1])) + bias


def _reference_forward(tensors, x, eps):
    """Unfused textbook implementation (conv -> BN -> relu -> pool)."""
    h = x
    for idx in range(3):
        h = np.maximum(
            _conv1d_valid(h, tensors[f"conv{idx}.kernel"],
                          tensors[f"conv{idx}.bias"]),
            0.0,
        )
        gamma = tensors[f"bn{idx}.gamma"]
        beta = tensors[f"bn{idx}.beta"]
        mean = tensors[f"bn{idx}.mean"]
        var = tensors[f"bn{idx}.variance"]
        h = gamma * (h - mean) / np.sqrt(var + eps) + beta
        h = np.maximum(h, 0.0)
        h = _maxpool2(h)
    h = h.reshape(h.shape[0], -1)
    h = np.maximum(h @ tensors["dense0.kernel"] + tensors["dense0.bias"], 0.0)
    logits = h @ tensors["dense1.kernel"] + tensors["dense1.bias"]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _fused_forward(tensors, x):
    """BN folded into the convolutions (the runtime's fast path)."""
    h = x
    for idx in range(3):
        gamma = tensors[f"bn{idx}.gamma"]
        mean = tensors[f"bn{idx}.mean"]
        var = tensors[f"bn{idx}.variance"]
        scale = gamma / np.sqrt(var + 1e-3)
        kernel = tensors[f"conv{idx}.kernel"] * scale[None, None, :]
        bias = (tensors[f"conv{idx}.bias"] - mean) * scale \
            + tensors[f"bn{idx}.beta"]
        h = np.maximum(_conv1d_valid(h, kernel, bias), 0.0)
        h = _maxpool2(h)
    h = h.reshape(h.shape[0], -1)
    h = np.maximum(h @ tensors["dense0.kernel"] + tensors["dense0.bias"], 0.0)
    logits = h @ tensors["dense1.kernel"] + tensors["dense1.bias"]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _load_model(pickle_path: Path, labels_json: Path | None):
    """Return (tensors, engine_config, labels) from a Keras pickle."""
    keras_zip, class_name = extract_embedded_keras_zip(pickle_path)
    config, metadata, datasets = load_model_definition(keras_zip)
    tensors = extract_weights(config, datasets)
    labels = [f"class_{i:02d}" for i in range(16)]
    if labels_json is not None:
        labels = [str(x) for x in json.loads(labels_json.read_bytes())]
        if len(labels) != 16:
            raise ValueError(
                f"Label list has {len(labels)} entries; model expects 16."
            )
    return tensors, config, metadata, datasets, class_name, labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a pickled Keras 3 model to a NumPy artifact."
    )
    parser.add_argument("pickle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument(
        "--labels-json", type=Path, default=None,
        help="Optional JSON file: list of 16 class names, index-ordered.",
    )
    args = parser.parse_args()

    keras_zip, class_name = extract_embedded_keras_zip(args.pickle)
    config, metadata, datasets = load_model_definition(keras_zip)
    tensors = extract_weights(config, datasets)

    labels = [f"class_{i:02d}" for i in range(16)]
    if args.labels_json:
        labels = [str(x) for x in json.loads(args.labels_json.read_bytes())]
        if len(labels) != 16:
            raise ValueError(
                f"Label list has {len(labels)} entries; model expects 16."
            )

    # Self-check: fused and textbook implementations must agree.
    rng = np.random.default_rng(7)
    probe = rng.standard_normal((4, 512, 2)).astype(np.float32)
    ref = _reference_forward(tensors, probe, eps=1e-3)
    fused = _fused_forward(tensors, probe)
    max_diff = float(np.abs(ref - fused).max())
    if max_diff > 1e-4:
        raise RuntimeError(
            f"Self-check failed: fused vs reference max diff {max_diff}"
        )

    layer_desc = []
    for L in config["config"]["layers"]:
        c = L["config"]
        desc = L["class_name"]
        if "filters" in c:
            desc += f"({c['filters']}, k={c['kernel_size']})"
        if "units" in c:
            desc += f"({c['units']})"
        layer_desc.append(desc)

    engine_config = {
        "class_name": class_name,
        "input_shape": [512, 2],
        "num_classes": 16,
        "labels": labels,
        "labels_source": (
            "provided" if args.labels_json else "neutral (unlabeled model)"
        ),
        "normalization": "unit_rms",
        "batch_norm_epsilon": 1e-3,
        "conv_padding": "valid",
        "pooling": "max, stride 2",
        "activation_head": "softmax",
        "keras_version": metadata.get("keras_version"),
        "date_saved": metadata.get("date_saved"),
        "architecture": layer_desc,
        "self_check_max_diff": max_diff,
        "note": (
            "Converted from a Keras pickle without unpickling; the "
            "training dataset and label map were not embedded in the "
            "file. Treat outputs as uncalibrated until validated."
        ),
    }

    tensors["config_json"] = np.frombuffer(
        json.dumps(engine_config, indent=2).encode("utf-8"), dtype=np.uint8
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **tensors)
    config_json_path = config_json_arg if config_json_arg is not None else None

    config_path = config_json_path if config_json_path is not None else output_path.with_suffix(".config.json")
    config_path.write_text(
        json.dumps(engine_config, indent=2), encoding="utf-8"
    )

    print(f"Converted {pickle_path} -> {output_path}")
    print(f"  layers: {' -> '.join(layer_desc)}")
    print(f"  labels: {labels_source(engine_config)}")
    print(f"  self-check max |diff|: {max_diff:.2e}")

    return 0


def labels_source(cfg: dict) -> str:
    return cfg["labels_source"]


def convert(pickle_path: Path, output_path: Path, labels_json: Path | None = None) -> None:
    """One-shot conversion (no unpickling): pickle -> npz + config sidecar.

    Public entry point for the spectra CLI; avoids re-parsing argv.
    """
    config_json_path: Path | None = None
    keras_zip, class_name = extract_embedded_keras_zip(pickle_path)
    config, metadata, datasets = load_model_definition(keras_zip)
    tensors = extract_weights(config, datasets)

    labels = [f"class_{i:02d}" for i in range(16)]
    if labels_json is not None:
        labels = [str(x) for x in json.loads(labels_json.read_bytes())]
        if len(labels) != 16:
            raise ValueError(
                f"Label list has {len(labels)} entries; model expects 16."
            )

    # Self-check: fused and textbook implementations must agree.
    rng = np.random.default_rng(7)
    probe = rng.standard_normal((4, 512, 2)).astype(np.float32)
    ref = _reference_forward(tensors, probe, eps=BN_EPSILON)
    fused = _fused_forward(tensors, probe)
    max_diff = float(np.abs(ref - fused).max())
    if max_diff > 1e-4:
        raise RuntimeError(
            f"Self-check failed: fused vs reference max diff {max_diff}"
        )

    layer_desc = []
    for L in config["config"]["layers"]:
        c = L["config"]
        desc = L["class_name"]
        if "filters" in c:
            desc += f"({c['filters']}, k={c['kernel_size']})"
        if "units" in c:
            desc += f"({c['units']})"
        layer_desc.append(desc)

    engine_config = {
        "class_name": class_name,
        "input_shape": [512, 2],
        "num_classes": 16,
        "labels": labels,
        "labels_source": (
            "provided" if labels_json is not None else "neutral (unlabeled model)"
        ),
        "normalization": "unit_rms",
        "batch_norm_epsilon": 1e-3,
        "conv_padding": "valid",
        "pooling": "max, stride 2",
        "activation_head": "softmax",
        "keras_version": metadata.get("keras_version"),
        "date_saved": metadata.get("date_saved"),
        "architecture": layer_desc,
        "self_check_max_diff": max_diff,
        "note": (
            "Converted from a Keras pickle without unpickling; the "
            "training dataset and label map were not embedded in the "
            "file. Treat outputs as uncalibrated until validated."
        ),
    }

    tensors["config_json"] = np.frombuffer(
        json.dumps(engine_config, indent=2).encode("utf-8"), dtype=np.uint8
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **tensors)
    config_json_path = config_json_arg if config_json_arg is not None else None

    config_path = config_json_path if config_json_path is not None else output_path.with_suffix(".config.json")
    config_path.write_text(
        json.dumps(engine_config, indent=2), encoding="utf-8"
    )

    print(f"Converted {pickle_path} -> {output_path}")
    print(f"  layers: {' -> '.join(layer_desc)}")
    print(f"  labels: {labels_source(engine_config)}")
    print(f"  self-check max |diff|: {max_diff:.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
