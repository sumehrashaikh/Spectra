"""Tests for the optional external real-world validation feature.

These tests build a tiny *synthetic stand-in* HDF5 file with the same
layout as the downloaded dataset, so they never depend on the ~833 MB
real files (which are intentionally not committed) and never write into
the repository.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from prototype.external_validation import hdf5_dataset as loader
from prototype.external_validation.evaluate import (
    evaluate_dataset,
    render_summary,
)

CLASSES = ["BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM"]
CHANNELS = {0: "clean", 1: "multipath"}
SNRS = [20, 22]
FRAME_LENGTH = 1024
PER_CELL = 2


@pytest.fixture()
def fake_dataset(tmp_path):
    """A miniature dataset with the real layout (7 mods x 2 chan x 2 SNR)."""

    modulation_ids, channel_ids, snr_values = [], [], []
    for mod_id in range(len(CLASSES)):
        for channel_id in CHANNELS:
            for snr in SNRS:
                for _ in range(PER_CELL):
                    modulation_ids.append(mod_id)
                    channel_ids.append(channel_id)
                    snr_values.append(snr)

    total = len(modulation_ids)
    rng = np.random.default_rng(0)
    # Deterministic, non-degenerate frames with distinct per-class spectra.
    t = np.arange(FRAME_LENGTH) / 2000.0
    x = np.empty((total, FRAME_LENGTH, 2), dtype=np.float16)
    for row, mod_id in enumerate(modulation_ids):
        tone = 50.0 * (mod_id + 1)
        i = 0.3 * np.cos(2 * np.pi * tone * t) + 0.02 * rng.standard_normal(FRAME_LENGTH)
        q = 0.3 * np.sin(2 * np.pi * tone * t) + 0.02 * rng.standard_normal(FRAME_LENGTH)
        x[row, :, 0] = i
        x[row, :, 1] = q

    subset_path = tmp_path / "subset_test.h5"
    with h5py.File(subset_path, "w") as handle:
        handle.create_dataset("X", data=x, chunks=(32, FRAME_LENGTH, 2))
        handle.create_dataset("y_mod", data=np.asarray(modulation_ids, dtype=np.int16))
        handle.create_dataset("y_chan", data=np.asarray(channel_ids, dtype=np.int8))
        handle.create_dataset("y_snr", data=np.asarray(snr_values, dtype=np.int16))
        handle.attrs["frame_len"] = FRAME_LENGTH
        handle.attrs["mod2id_json"] = json.dumps(
            {name: idx for idx, name in enumerate(CLASSES)}
        )

    (tmp_path / "dataset.config.json").write_text(json.dumps({
        "dataset_id": "test-standin",
        "files": {"test": "subset_test.h5"},
        "frame_length": FRAME_LENGTH,
        "sample_rate_hz": 2.0e6,
        "channels": {str(k): v for k, v in CHANNELS.items()},
        "classes": {str(idx): name for idx, name in enumerate(CLASSES)},
    }), encoding="utf-8")

    return tmp_path, total


def test_inspect_reports_shape_labels_and_balance(fake_dataset):
    dataset_dir, total = fake_dataset
    report = loader.load_dataset(dataset_dir).inspect(["test"])

    assert report["available"] is True
    test = report["subsets"]["test"]
    assert test["X_shape"] == [total, FRAME_LENGTH, 2]
    assert test["X_dtype"] == "float16"
    assert test["num_frames"] == total
    assert sorted(test["modulation_labels"]) == sorted(CLASSES)
    assert set(test["channel_labels"]) == {"clean", "multipath"}
    assert set(test["snr_values_db"]) == {20, 22}
    # 7 classes x 2 channels x 2 SNR x 2 frames, perfectly balanced.
    assert test["class_balance_ratio"] == 1.0
    assert all(count == PER_CELL * 2 * 2 for count in
               test["modulation_labels"].values())


def test_sampling_is_stratified_and_deterministic(fake_dataset):
    dataset_dir, _ = fake_dataset
    dataset = loader.load_dataset(dataset_dir)

    frames = dataset.sample_frames(subsets=["test"], per_cell=1, seed=3)
    assert len(frames) == len(CLASSES) * len(CHANNELS) * len(SNRS)

    cells = {(f.modulation, f.channel, f.snr_db) for f in frames}
    assert len(cells) == len(frames)
    assert {f.modulation for f in frames} == set(CLASSES)

    assert frames[0].samples.shape == (FRAME_LENGTH,)
    assert np.iscomplexobj(frames[0].samples)

    again = dataset.sample_frames(subsets=["test"], per_cell=1, seed=3)
    assert [f.index for f in again] == [f.index for f in frames]


def test_evaluate_reports_breakdowns_and_unsupported_classes(fake_dataset):
    dataset_dir, total = fake_dataset
    payload = evaluate_dataset(
        dataset_dir=dataset_dir, subsets=["test"], per_cell=1, seed=5,
        ml_enabled=False,
    )

    assert payload["available"] is True
    assert payload["sample_plan"]["frames_evaluated"] == (
        len(CLASSES) * len(CHANNELS) * len(SNRS)
    )

    results = payload["results"]
    assert set(results["by_modulation"]) == set(CLASSES)
    assert set(results["by_channel"]) == {"clean", "multipath"}
    assert set(results["by_snr"]) == {"20 dB", "22 dB"}

    # Only BPSK/QPSK live in SPECTRA's deterministic vocabulary.
    assert payload["unsupported_dataset_classes"] == [
        "GMSK", "NBFM", "OFDM", "QAM", "WBFM",
    ]
    overall = results["overall"]
    assert overall["dataset_class_supported"] == len(SNRS) * len(CHANNELS) * 2
    assert overall["unsupported_dataset_class"] == (
        overall["frames"] - overall["dataset_class_supported"]
    )
    # Unsupported classes are never count-eligible for accuracy.
    assert overall["dsp_evaluable"] == overall["dataset_class_supported"]

    # No field presents the dataset paper's number as a SPECTRA result.
    assert "accuracy" not in payload["dataset"]
    assert "84.3" not in json.dumps(payload["results"])

    limits = " ".join(payload["limits"]).lower()
    assert "not a spectra accuracy claim" in limits
    assert "no low-snr claim" in limits
    assert any("2 MSps" in item["dataset"] for item in payload["domain_mismatch"])


def test_evaluate_with_ml_assist_keeps_dsp_and_ml_separate(fake_dataset):
    dataset_dir, _ = fake_dataset
    payload = evaluate_dataset(
        dataset_dir=dataset_dir, subsets=["test"], per_cell=1, seed=5,
        ml_enabled=True,
    )
    overall = payload["results"]["overall"]
    assert overall["ml_enabled"] == overall["frames"]
    # ML metrics are their own fields; DSP accuracy is untouched by them.
    assert "dsp_accuracy" in overall and "ml_accuracy" in overall
    assert payload["ml_artifact"]["available"] in (True, False)


def test_render_summary_is_human_readable(fake_dataset):
    dataset_dir, _ = fake_dataset
    payload = evaluate_dataset(
        dataset_dir=dataset_dir, subsets=["test"], per_cell=1, seed=5,
        ml_enabled=False,
    )
    lines = render_summary(payload)
    text = "\n".join(lines)
    assert "sampled frames:" in text
    assert "unsupported dataset classes:" in text


def test_unavailable_when_h5py_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "h5py_available", lambda: False)
    dataset = loader.load_dataset(tmp_path)
    assert dataset.available is False
    assert "h5py" in (dataset.unavailable_reason or "")


def test_missing_dataset_dir_is_reported_not_crashed(tmp_path):
    payload = evaluate_dataset(dataset_dir=tmp_path / "does-not-exist")
    assert payload["available"] is False
    assert "h5py" in payload["reason"] or "subset_" in payload["reason"]


def test_cli_commands_are_optional_and_registered(fake_dataset):
    from prototype.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["dataset-inspect", "--dataset-dir", str(fake_dataset[0])]
    )
    assert args.func is not None
    assert args.func(args) == 0

    args = parser.parse_args([
        "dataset-eval", "--dataset-dir", str(fake_dataset[0]),
        "--per-cell", "1", "--no-ml", "--table",
    ])
    assert args.ml is False
    assert args.func(args) == 0


def test_cli_dataset_inspect_exits_nonzero_when_absent(tmp_path, capsys):
    from prototype.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["dataset-inspect", "--dataset-dir", str(tmp_path / "nope")]
    )
    assert args.func(args) == 2
    assert "available" in capsys.readouterr().out
