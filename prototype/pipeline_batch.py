"""
Multi-signal (wideband) batch analysis.

``analyze_all_candidates`` runs the per-candidate pipeline over every
detected signal in a capture and returns a ranked set of results plus
a frequency-timeline summary suitable for spectrum-monitoring views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prototype.core.config import AnalysisConfig, processing_mode_config
from prototype.core.signal import Signal
from prototype.pipeline import AnalysisResult, analyze_samples


@dataclass
class BatchSummary:
    """Summary of one candidate inside a batch run (cheap, no full DSP)."""

    index: int
    center_frequency_hz: float
    bandwidth_hz: float
    peak_frequency_hz: float
    peak_power_db: float | None
    confidence: float


@dataclass
class BatchResult:
    """Results of analyzing multiple candidates from one capture."""

    input_info: dict[str, Any] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    results: list[AnalysisResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary: full detail for analyzed candidates."""
        return {
            "input": dict(self.input_info),
            "detections": [dict(d) for d in self.detections],
            "analyzed_candidates": [
                {
                    "candidate_index": index,
                    "selected_candidate": r.selected_candidate,
                    "classification": r.classification,
                    "symbol_rate": r.symbol_rate,
                    "synchronization": r.synchronization,
                    "demodulation": _strip_arrays(r.demodulation),
                    "ber": r.ber,
                    "warnings": r.warnings,
                }
                for index, r in self.results
            ],
            "warnings": list(self.warnings),
        }


def _strip_arrays(value: Any) -> Any:
    """Remove non-serializable entries (constellation arrays) recursively."""
    if isinstance(value, dict):
        return {
            k: _strip_arrays(v)
            for k, v in value.items()
            if not isinstance(v, (np.ndarray, complex))
        }
    return value


def analyze_all_candidates(
    samples: np.ndarray,
    sample_rate: float,
    mode: str = "balanced",
    max_candidates: int | None = None,
    reference_bits: np.ndarray | None = None,
    reference_offset_symbols: int = 0,
    config: AnalysisConfig | None = None,
) -> BatchResult:
    """
    Detect every candidate in the capture and analyze each one.

    Parameters
    ----------
    samples, sample_rate :
        Wideband complex capture.
    mode :
        Processing preset used for every candidate.
    max_candidates :
        Limit the number of candidates analyzed (strongest first).
    reference_bits :
        Optional transmitted-bit reference. Applied only to the
        candidate whose index matches ``reference_offset_symbols``
        context — in practice BER is attempted for every candidate and
        ambiguity searches keep wrong-candidate BER near 0.5, which is
        reported honestly rather than hidden.
    """
    from prototype.detection.detector import detect_candidates

    if config is None:
        config = processing_mode_config(mode)

    signal = Signal(samples=np.asarray(samples), sample_rate=sample_rate)
    candidates = detect_candidates(
        signal,
        threshold_db=config.detection.threshold_db,
        min_bandwidth_hz=config.detection.min_bandwidth_hz,
        smoothing_window_bins=config.detection.smoothing_window_bins,
        prominence_db=config.detection.prominence_db,
        min_peak_distance_bins=config.detection.min_peak_distance_bins,
        merge_gap_bins=config.detection.merge_gap_bins,
    )

    batch = BatchResult(
        input_info={
            "sample_rate": float(sample_rate),
            "num_samples": int(np.asarray(samples).size),
            "mode": mode,
        }
    )
    batch.detections = [c.summary() for c in candidates]

    if not candidates:
        batch.warnings.append("No candidates detected in the capture.")
        return batch

    selected = candidates if max_candidates is None else candidates[:max_candidates]

    for index, _candidate in enumerate(selected):
        try:
            result = analyze_samples(
                samples,
                sample_rate,
                config=config,
                reference_bits=reference_bits,
                input_info={"candidate_index": index},
            )
        except Exception as exc:
            batch.warnings.append(f"Candidate {index}: analysis failed: {exc}")
            continue
        batch.results.append((index, result))

    return batch


def candidate_timeline(batch: BatchResult, sample_rate: float) -> list[dict[str, Any]]:
    """
    Build a frequency-ordered candidate timeline from a batch result.

    Returns one row per candidate: frequency bounds, classification,
    confidence, and rank — the data behind a spectrum-monitoring list.
    """
    rows: list[dict[str, Any]] = []
    classification_by_index = {
        index: (r.classification or {}) for index, r in batch.results
    }

    for index, detection in enumerate(batch.detections):
        classification = classification_by_index.get(index, {})
        rows.append(
            {
                "candidate": index,
                "center_frequency_hz": float(detection["center_frequency"]),
                "bandwidth_hz": float(detection["bandwidth"]),
                "peak_power_db": detection.get("peak_power_db"),
                "confidence": detection.get("confidence"),
                "modulation": classification.get("modulation"),
                "classification_confidence": classification.get("confidence"),
            }
        )

    rows.sort(key=lambda row: row["center_frequency_hz"])
    return rows
