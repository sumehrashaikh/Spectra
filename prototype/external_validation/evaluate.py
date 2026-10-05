"""Evaluate SPECTRA's existing chain on the external real-world dataset.

For each sampled frame the utility runs, in order, the project's **own**
stages — nothing here re-implements or replaces them:

1. ``core.preprocessor.preprocess_signal`` (existing preprocessing)
2. ``classification.classifier.classify_signal`` (deterministic DSP path)
3. ``ml.cnn.predict_modulation`` (the optional ML assist, when enabled)

Results are reported separately for DSP and ML, with agreement,
disagreement, and *unsupported* classes called out rather than scored as
errors. Breakdowns are produced by modulation, clean/multipath channel and
labelled SNR.

Boundaries this module deliberately respects:

* It never feeds the dataset into training and never touches artifacts.
* It never claims the dataset's published baseline accuracy as SPECTRA's.
* It never claims low-SNR performance: the dataset is 20-30 dB only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from prototype.external_validation.hdf5_dataset import (
    RealWorldIqDataset,
    SampledFrame,
    load_dataset,
)

# Labels SPECTRA's deterministic classifier can emit are decided by
# ``ml.fusion``; this module only consumes them.
DEFAULT_SUBSETS: tuple[str, ...] = ("test",)


def _canonical(label: Any) -> str:
    from prototype.ml.fusion import canonical_modulation

    return canonical_modulation(label)


def _comparable(label: Any) -> bool:
    from prototype.ml.fusion import is_comparable

    return is_comparable(label)


def _new_bucket() -> dict[str, Any]:
    return {
        "frames": 0,
        "dataset_class_supported": 0,
        "unsupported_dataset_class": 0,
        "dsp_evaluable": 0,
        "dsp_correct": 0,
        "dsp_unknown": 0,
        "dsp_error": 0,
        "ml_enabled": 0,
        "ml_available": 0,
        "ml_comparable": 0,
        "ml_correct": 0,
        "mentions": Counter(),
        "agreement_denominator": 0,
        "agreement": 0,
        "disagreement": 0,
        "presented_as": Counter(),
        "measured_snr_sum": 0.0,
        "measured_snr_n": 0,
        "dsp_labels": Counter(),
        "ml_labels": Counter(),
    }


def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "frames": bucket["frames"],
        "dataset_class_supported": bucket["dataset_class_supported"],
        "unsupported_dataset_class": bucket["unsupported_dataset_class"],
        "dsp_evaluable": bucket["dsp_evaluable"],
        "dsp_correct": bucket["dsp_correct"],
        "dsp_accuracy": rate(bucket["dsp_correct"], bucket["dsp_evaluable"]),
        "dsp_unknown": bucket["dsp_unknown"],
        "dsp_error": bucket["dsp_error"],
        "dsp_labels": dict(bucket["dsp_labels"].most_common()),
        "ml_enabled": bucket["ml_enabled"],
        "ml_available": bucket["ml_available"],
        "ml_comparable": bucket["ml_comparable"],
        "ml_correct": bucket["ml_correct"],
        "ml_accuracy": rate(bucket["ml_correct"], bucket["ml_comparable"]),
        "ml_labels": dict(bucket["ml_labels"].most_common()),
        "presented_as": dict(bucket["presented_as"]),
        "agreement_denominator": bucket["agreement_denominator"],
        "agreement": bucket["agreement"],
        "disagreement": bucket["disagreement"],
        "agreement_rate": rate(
            bucket["agreement"], bucket["agreement_denominator"]
        ),
        "mean_dsp_measured_snr_db": (
            round(bucket["measured_snr_sum"] / bucket["measured_snr_n"], 2)
            if bucket["measured_snr_n"] else None
        ),
    }


def _classify_frame(
    frame: SampledFrame,
    sample_rate: float,
    ml_enabled: bool,
) -> dict[str, Any]:
    """Run the existing preprocessing + DSP + ML stages on one frame."""

    from prototype.core.preprocessor import preprocess_signal
    from prototype.core.signal import Signal
    from prototype.classification.classifier import classify_signal

    record: dict[str, Any] = {
        "subset": frame.subset,
        "index": frame.index,
        "dataset_modulation": frame.modulation,
        "channel": frame.channel,
        "snr_db": float(frame.snr_db),
        "dsp": None,
        "ml": None,
        "error": None,
        "measured_snr_db": None,
    }

    try:
        signal = Signal(samples=frame.samples.astype(np.complex128),
                        sample_rate=float(sample_rate))
        processed = preprocess_signal(signal)
        record["measured_snr_db"] = processed.metadata.get("snr_db")
    except Exception as exc:  # a frame must never abort the run
        record["error"] = f"preprocessing: {type(exc).__name__}: {exc}"
        return record

    try:
        if ml_enabled:
            from prototype.ml.cnn import predict_modulation

            prediction = predict_modulation(processed.samples)
            if prediction is None:
                record["ml"] = {"available": False}
            else:
                record["ml"] = {
                    "available": True,
                    "label": prediction.get("predicted_class"),
                    "canonical": _canonical(prediction.get("predicted_class")),
                    "comparable": bool(
                        _comparable(prediction.get("predicted_class"))
                    ),
                    "confidence": prediction.get("confidence"),
                    "frame_agreement": prediction.get("frame_agreement"),
                    "presented_as": prediction.get("presented_as"),
                    "trained": prediction.get("trained"),
                    "validated": prediction.get("validated"),
                    "validation_accuracy": prediction.get("validation_accuracy"),
                    "artifact": prediction.get("artifact"),
                    "num_frames": prediction.get("num_frames"),
                }
    except Exception as exc:
        record["ml"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        classification = classify_signal(
            processed, use_constellation=False, synchronized=False
        )
        record["dsp"] = {
            "label": classification.modulation,
            "canonical": _canonical(classification.modulation),
            "comparable": bool(_comparable(classification.modulation)),
            "method": classification.method,
            "confidence": classification.confidence,
        }
    except Exception as exc:
        record["dsp"] = {
            "label": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return record


def evaluate_dataset(
    dataset_dir: str | Path | None = None,
    subsets: tuple[str, ...] | list[str] = DEFAULT_SUBSETS,
    per_cell: int = 3,
    seed: int = 7,
    ml_enabled: bool = True,
    max_frames: int | None = None,
    dump_frames: bool = False,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run the existing SPECTRA chain over a sampled external subset.

    ``max_workers > 1`` parallelises the per-frame DSP/ML calls across
    processes.  It is off by default (deterministic, no fork surprises on
    all platforms).
    """

    dataset = load_dataset(dataset_dir)
    if not dataset.available:
        return {
            "available": False,
            "reason": dataset.unavailable_reason,
            "hint": "Install h5py and place the subset_*.h5 files, or pass "
                    "--dataset-dir. See dataset/README.md.",
        }

    frames = dataset.sample_frames(
        subsets=tuple(subsets), per_cell=per_cell, seed=seed,
        max_frames=max_frames,
    )

    records = _run_frames(frames, dataset.sample_rate_hz, ml_enabled,
                          max_workers)

    overall = _new_bucket()
    groups: dict[str, dict[str, dict[str, Any]]] = {
        "by_modulation": defaultdict(_new_bucket),
        "by_channel": defaultdict(_new_bucket),
        "by_modulation_channel": defaultdict(_new_bucket),
        "by_snr": defaultdict(_new_bucket),
        "by_modulation_snr": defaultdict(_new_bucket),
    }
    confusion: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        _accumulate(overall, record)
        _accumulate(groups["by_modulation"][record["dataset_modulation"]], record)
        _accumulate(groups["by_channel"][record["channel"]], record)
        _accumulate(
            groups["by_modulation_channel"][
                f"{record['dataset_modulation']}|{record['channel']}"
            ],
            record,
        )
        _accumulate(groups["by_snr"][f"{record['snr_db']:g} dB"], record)
        _accumulate(
            groups["by_modulation_snr"][
                f"{record['dataset_modulation']}|{record['snr_db']:g} dB"
            ],
            record,
        )
        dsp_label = (record.get("dsp") or {}).get("label") or "ERROR/UNKNOWN"
        confusion[record["dataset_modulation"]][dsp_label] += 1

    ml_artifact = _ml_artifact_info(records)

    payload: dict[str, Any] = {
        "available": True,
        "dataset": {
            "dataset_dir": str(dataset.dataset_dir),
            "subsets": list(subsets),
            "sample_rate_hz": dataset.sample_rate_hz,
            "frame_length": dataset.frame_length,
            "class_names": list(dataset.class_names),
            "channel_names": list(dataset.channel_names),
        },
        "method": {
            "stages": [
                "core.preprocessor.preprocess_signal",
                "classification.classifier.classify_signal"
                " (use_constellation=False, synchronized=False)",
                "ml.cnn.predict_modulation (optional)"
                if ml_enabled else "ml.cnn.predict_modulation (disabled)",
            ],
            "dsp_only": not ml_enabled,
            "notes": [
                "The deterministic DSP classifier is the SPECTRA classifier; "
                "the ML CNN is reported separately as an assist and is never "
                "used to override it.",
                "dataset_class_supported is decided by ml.fusion.is_comparable: "
                "classes outside the DSP vocabulary are reported as "
                "UNSUPPORTED, not as errors.",
            ],
        },
        "sample_plan": {
            "subset_selection": list(subsets),
            "stratified_by": ["modulation", "channel", "snr"],
            "per_cell": per_cell,
            "seed": seed,
            "frames_evaluated": len(records),
        },
        "results": {
            "overall": _finalize(overall),
            "by_modulation": _finalize_groups(groups["by_modulation"]),
            "by_channel": _finalize_groups(groups["by_channel"]),
            "by_modulation_channel": _finalize_groups(
                groups["by_modulation_channel"]
            ),
            "by_snr": _finalize_groups(groups["by_snr"]),
            "by_modulation_snr": _finalize_groups(
                groups["by_modulation_snr"]
            ),
            "confusion_dataset_to_dsp": {
                label: dict(counter.most_common())
                for label, counter in sorted(confusion.items())
            },
        },
        "ml_artifact": ml_artifact,
        "unsupported_dataset_classes": sorted({
            record["dataset_modulation"] for record in records
            if not _comparable(record["dataset_modulation"])
        }),
        "domain_mismatch": _domain_mismatch(dataset),
        "limits": _limits(),
    }

    if dump_frames:
        payload["frames"] = records
    return payload


def _run_frames(
    frames: list[SampledFrame],
    sample_rate: float,
    ml_enabled: bool,
    max_workers: int | None,
) -> list[dict[str, Any]]:
    if max_workers and max_workers > 1 and len(frames) > 1:
        from concurrent.futures import ProcessPoolExecutor

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(
                    _classify_frame, frames,
                    [sample_rate] * len(frames), [ml_enabled] * len(frames),
                ))
        except Exception:
            # Fall back to the serial path rather than failing the run.
            pass
    return [_classify_frame(frame, sample_rate, ml_enabled) for frame in frames]


def _accumulate(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["frames"] += 1

    dataset_supported = bool(_comparable(record["dataset_modulation"]))
    if dataset_supported:
        bucket["dataset_class_supported"] += 1
    else:
        bucket["unsupported_dataset_class"] += 1

    measured = record.get("measured_snr_db")
    if isinstance(measured, (int, float)):
        bucket["measured_snr_sum"] += float(measured)
        bucket["measured_snr_n"] += 1

    dsp = record.get("dsp") or {}
    dsp_label = dsp.get("label")
    if dsp_label is None:
        bucket["dsp_error"] += 1
    else:
        bucket["dsp_labels"][dsp_label] += 1
        if _canonical(dsp_label) == "Unknown":
            bucket["dsp_unknown"] += 1
        if dataset_supported:
            bucket["dsp_evaluable"] += 1
            if _canonical(dsp_label) == _canonical(record["dataset_modulation"]):
                bucket["dsp_correct"] += 1

    if record.get("ml") is None and not dsp:
        return

    ml = record.get("ml") or {}
    if record.get("ml") is None:
        # ML assist disabled for this run: nothing to compare.
        return

    bucket["ml_enabled"] += 1
    if not ml.get("available"):
        return
    bucket["ml_available"] += 1
    ml_label = ml.get("label")
    bucket["ml_labels"][str(ml_label)] += 1
    presented = ml.get("presented_as")
    if presented:
        bucket["presented_as"][str(presented)] += 1

    ml_comparable = bool(ml.get("comparable"))
    if ml_comparable:
        bucket["ml_comparable"] += 1
        if dataset_supported and _canonical(ml_label) == _canonical(
            record["dataset_modulation"]
        ):
            bucket["ml_correct"] += 1

    # Agreement is only meaningful when both sides produced a comparable
    # label for this frame.
    if ml_comparable and dsp_label is not None and _comparable(dsp_label):
        bucket["agreement_denominator"] += 1
        if _canonical(ml_label) == _canonical(dsp_label):
            bucket["agreement"] += 1
        else:
            bucket["disagreement"] += 1


def _finalize_groups(
    groups: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        key: _finalize(bucket)
        for key, bucket in sorted(groups.items())
    }


def _ml_artifact_info(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        ml = record.get("ml") or {}
        if ml.get("available"):
            return {
                "available": True,
                "artifact": ml.get("artifact"),
                "trained": ml.get("trained"),
                "validated": ml.get("validated"),
                "validation_accuracy": ml.get("validation_accuracy"),
                "num_frames_per_capture": ml.get("num_frames"),
                "reporting": (
                    "The ML columns are the CNN's own scores for this "
                    "external data. They are an assist, not a SPECTRA "
                    "accuracy claim."
                ),
            }
    return {"available": False}


def _domain_mismatch(dataset: RealWorldIqDataset) -> list[dict[str, str]]:
    """Known, verifiable mismatches between this data and SPECTRA's chain."""

    return [
        {
            "aspect": "sample_rate",
            "dataset": f"{dataset.sample_rate_hz:g} Hz (2 MSps SDR capture)",
            "spectra": (
                "8 kHz baseband in the synthetic training data "
                "(ml/dataset.py SAMPLE_RATE) and the CLI self-check"
            ),
            "effect": (
                "The DSP classifier's decision thresholds are absolute "
                "frequencies (e.g. the 80 Hz BFSK boundary), so they are "
                "out of scale for a 2 MSps capture; symbol-rate and spectral "
                "features do not transfer directly."
            ),
        },
        {
            "aspect": "frame_length",
            "dataset": f"{dataset.frame_length} samples per frame",
            "spectra": (
                "extract_frames(max_frames=8) at the artifact-declared "
                "frame length (512 for the shipped model, 1024 for an ML v3 "
                "artifact); ml/cnn.py FRAME_LENGTH = 512 is only the module "
                "default for callers that do not name a length"
            ),
            "effect": (
                "One dataset frame yields two 512-sample ML frames; DSP "
                "detection over a single 1024-sample frame is not the "
                "intended pipeline input (a full capture, not one frame)."
            ),
        },
        {
            "aspect": "class_vocabulary",
            "dataset": "7 coarse classes: " + ", ".join(dataset.class_names),
            "spectra": (
                "DSP vocabulary: BPSK, QPSK, 8-PSK, 16-QAM, BFSK, OOK, "
                "Unknown; ML vocabulary: 16 synthetic classes"
            ),
            "effect": (
                "Only BPSK/QPSK are directly comparable; QAM is not split, "
                "and OFDM/NBFM/WBFM/GMSK have no SPECTRA counterpart, so "
                "most classes are reported unsupported."
            ),
        },
        {
            "aspect": "snr_range",
            "dataset": "20-30 dB only",
            "spectra": "synthetic training/channel simulation 0-18 dB",
            "effect": (
                "This dataset cannot evidence low-SNR behaviour; results "
                "describe a high-SNR regime the synthetic chain was not "
                "tuned for."
            ),
        },
        {
            "aspect": "power_normalization",
            "dataset": "power-normalized frames (unit power)",
            "spectra": "core.preprocessor normalizes per capture (unit RMS)",
            "effect": (
                "Broadly compatible, but the preprocessor's per-frame "
                "normalization and DC removal are applied on top of the "
                "dataset's own scaling."
            ),
        },
    ]


def _limits() -> list[str]:
    return [
        "EXTERNAL DATA, NOT A SPECTRA ACCURACY CLAIM. The dataset is a "
        "third-party real-world capture set; numbers here describe how the "
        "existing chain behaves on it, not a validated product metric.",
        "The dataset paper reports a 84.3% baseline test accuracy for the "
        "AUTHORS' TensorFlow model. That number is theirs and is never "
        "reproduced as a SPECTRA result.",
        "NO low-SNR claim: the dataset spans 20-30 dB SNR only.",
        "Only a sampled, stratified subset is evaluated; counts are not "
        "population estimates for the full 560k-frame subset.",
        "The ML CNN is presented separately and is not validated for this "
        "domain; its scores are evidence, not a classifier verdict.",
    ]


def render_summary(payload: dict[str, Any]) -> list[str]:
    """Compact human-readable lines for the CLI ``--table`` option."""

    if not payload.get("available"):
        return [f"dataset unavailable: {payload.get('reason')}"]

    lines: list[str] = []
    plan = payload["sample_plan"]
    lines.append(
        f"sampled frames: {plan['frames_evaluated']} "
        f"(per cell={plan['per_cell']}, seed={plan['seed']}, "
        f"subsets={','.join(plan['subset_selection'])})"
    )
    overall = payload["results"]["overall"]
    lines.append(
        f"overall: dsp_acc={_fmt(overall['dsp_accuracy'])} "
        f"({overall['dsp_correct']}/{overall['dsp_evaluable']})  "
        f"ml_acc={_fmt(overall['ml_accuracy'])} "
        f"({overall['ml_correct']}/{overall['ml_comparable']})  "
        f"agreement={_fmt(overall['agreement_rate'])}  "
        f"unsupported={overall['unsupported_dataset_class']}/"
        f"{overall['frames']}"
    )

    lines.append("")
    lines.append(f"{'modulation':<12}{'n':>5}{'dsp_acc':>10}{'ml_acc':>10}"
                 f"{'agree':>10}")
    for name, group in payload["results"]["by_modulation"].items():
        lines.append(
            f"{name:<12}{group['frames']:>5}"
            f"{_fmt(group['dsp_accuracy']):>10}{_fmt(group['ml_accuracy']):>10}"
            f"{_fmt(group['agreement_rate']):>10}"
        )

    lines.append("")
    lines.append(f"{'channel':<12}{'n':>5}{'dsp_acc':>10}{'ml_acc':>10}")
    for name, group in payload["results"]["by_channel"].items():
        lines.append(
            f"{name:<12}{group['frames']:>5}"
            f"{_fmt(group['dsp_accuracy']):>10}{_fmt(group['ml_accuracy']):>10}"
        )

    lines.append("")
    lines.append(f"{'snr':<12}{'n':>5}{'dsp_acc':>10}{'ml_acc':>10}")
    for name, group in payload["results"]["by_snr"].items():
        lines.append(
            f"{name:<12}{group['frames']:>5}"
            f"{_fmt(group['dsp_accuracy']):>10}{_fmt(group['ml_accuracy']):>10}"
        )

    lines.append("")
    lines.append(
        "unsupported dataset classes: "
        + (", ".join(payload["unsupported_dataset_classes"]) or "none")
    )
    return lines


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
