"""
End-to-end signal analysis pipeline.

``analyze_capture`` / ``analyze_samples`` run the full V2 chain:

    load -> preprocess -> detect -> select candidate -> isolate (DDC)
      -> coarse classify -> symbol-rate estimate -> synchronize
      -> fine classify -> demodulate -> BER (when a reference exists)

Every stage is timed and recorded in the provenance manifest; a stage
failure degrades gracefully into a warning instead of aborting the
run. Results are structured dataclasses serializable to JSON.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prototype.core.config import (
    AnalysisConfig,
    IsolationConfig,
    processing_mode_config,
    config_to_dict,
    FECMode,
)
from prototype.core.exceptions import PipelineError
from prototype.core.provenance import new_provenance, provenance_to_dict
from prototype.core.signal import Signal
from prototype.fec.identification_interleaving import (
    AUTO_DETECTED,
    identify_interleaving,
)
from prototype.fec.interleaving import deinterleave_bits

logger = logging.getLogger("spectra.pipeline")


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass
class AnalysisResult:
    """Structured result of one end-to-end analysis run."""

    input_info: dict[str, Any] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    isolation: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    symbol_rate: dict[str, Any] | None = None
    synchronization: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    demodulation: dict[str, Any] | None = None
    ber: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    ml: dict[str, Any] | None = None
    fusion: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary (numbers rounded for readability)."""

        def _clean(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_clean(v) for v in value]
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
            if isinstance(value, np.ndarray):
                return _clean(value.tolist())
            if isinstance(value, complex):
                return {"real": value.real, "imag": value.imag}
            if is_dataclass(value):
                return _clean(asdict(value))
            return value

        return {
            "input": _clean(self.input_info),
            "detections": _clean(self.detections),
            "selected_candidate": _clean(self.selected_candidate),
            "isolation": _clean(self.isolation),
            "parameters": _clean(self.parameters),
            "symbol_rate": _clean(self.symbol_rate),
            "synchronization": _clean(self.synchronization),
            "classification": _clean(self.classification),
            "demodulation": _clean(self.demodulation),
            "ber": _clean(self.ber),
            "protocol": _clean(self.protocol),
            "ml": _clean(self.ml),
            "fusion": _clean(self.fusion),
            "warnings": list(self.warnings),
            "provenance": _clean(self.provenance),
        }


# ============================================================
# HELPERS
# ============================================================


def _modulation_order(modulation: str) -> int | None:
    """M-th power order used for carrier recovery per modulation."""
    if modulation == "BPSK":
        return 2
    if modulation in ("QPSK", "16-QAM"):
        return 4
    if modulation == "8-PSK":
        return 8
    if modulation == "OOK":
        return None  # amplitude modulation: no M-th power carrier recovery
    return None


def _timing_only_sync(isolated: Signal, symbol_rate: float) -> Signal:
    """Timing-only synchronization for envelope modulations (OOK)."""
    from prototype.core.synchronizer import synchronize_signal

    return synchronize_signal(isolated, symbol_rate=symbol_rate).signal


def _summary_symbols(samples: np.ndarray, max_points: int = 4) -> dict[str, float]:
    """Tiny constellation summary for reports (keeps JSON small)."""
    if samples.size == 0:
        return {}
    magnitudes = np.abs(samples)
    return {
        "symbol_rms": float(np.sqrt(np.mean(magnitudes**2))),
        "symbol_peak": float(np.max(magnitudes)),
    }


def _classify_and_sync(
    isolated: Signal,
    config: AnalysisConfig,
    provenance,
    warnings: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, Signal | None]:
    """
    Coarse classification -> symbol-rate estimate -> synchronization ->
    fine classification. Returns (classification, symbol_rate, sync,
    synchronized_signal).
    """
    from prototype.classification.classifier import classify_signal
    from prototype.parameters.symbol_rate import (
        estimate_symbol_rate,
        estimate_symbol_rate_fsk,
    )
    from prototype.core.synchronization import synchronize_signal as full_sync

    # ---- coarse classification on the isolated baseband waveform ----
    coarse = classify_signal(isolated, use_constellation=False, synchronized=False)

    modulation = coarse.modulation
    order = _modulation_order(modulation)

    # ---- symbol-rate estimation ----
    if modulation == "BFSK":
        rate_estimate = estimate_symbol_rate_fsk(
            isolated.samples,
            isolated.sample_rate,
            min_symbol_rate=config.symbol_rate.min_symbol_rate,
            max_symbol_rate=config.symbol_rate.max_symbol_rate
            or isolated.sample_rate / 4.0,
        )
    else:
        rate_estimate = estimate_symbol_rate(
            isolated.samples,
            isolated.sample_rate,
            min_symbol_rate=config.symbol_rate.min_symbol_rate,
            max_symbol_rate=config.symbol_rate.max_symbol_rate
            or isolated.sample_rate / 4.0,
        )

    symbol_rate_summary = {
        "symbol_rate_hz": float(rate_estimate.symbol_rate),
        "samples_per_symbol": float(rate_estimate.samples_per_symbol),
        "confidence": float(rate_estimate.confidence),
        "method": rate_estimate.method,
    }

    if rate_estimate.symbol_rate <= 0:
        warnings.append("Symbol-rate estimation failed; synchronization skipped.")
        return (
            {
                "modulation": coarse.modulation,
                "confidence": coarse.confidence,
                "method": coarse.method,
                "stage": "coarse",
            },
            symbol_rate_summary,
            None,
            None,
        )

    # ---- synchronization ----
    if modulation == "BFSK":
        # FSK is carrier-offset robust: timing-only alignment happens
        # inside the BFSK demodulator. Skip carrier sync here.
        return (
            {
                "modulation": coarse.modulation,
                "confidence": coarse.confidence,
                "method": coarse.method,
                "stage": "coarse",
            },
            symbol_rate_summary,
            None,
            None,
        )

    if order is None and modulation != "OOK":
        order = 4
        warnings.append(
            f"Unknown modulation '{modulation}'; assuming 4th-power carrier recovery."
        )

    # Square-QAM signals use a dedicated synchronizer: the generic
    # M-th-power phase estimate is data-dependent on QAM (arbitrary
    # mis-rotation) and the magnitude-variation timing metric has no
    # peak on amplitude-modulated carriers (arbitrary origin).
    if modulation == "16-QAM":
        from prototype.core.synchronization import synchronize_qam_signal

        sync_result = synchronize_qam_signal(
            isolated,
            symbol_rate=rate_estimate.symbol_rate,
        )
    # OOK uses timing-only sync (no carrier phase recovery);
    # every other modulation uses the carrier+timing chain.
    elif order is not None:
        sync_result = full_sync(
            isolated,
            symbol_rate=rate_estimate.symbol_rate,
            modulation_order=order,
        )
    else:
        sync_result = _timing_only_sync(isolated, rate_estimate.symbol_rate)

    synchronized_signal = sync_result.signal

    synchronization_summary = {
        "symbol_rate_hz": float(sync_result.symbol_rate),
        "samples_per_symbol": float(sync_result.samples_per_symbol),
        "timing_offset": int(sync_result.timing_offset),
        "timing_confidence": float(sync_result.timing_confidence),
        "frequency_offset_hz": float(
            synchronized_signal.metadata.get("frequency_offset", 0.0)
        ),
        "frequency_confidence": float(
            synchronized_signal.metadata.get("frequency_confidence", 0.0)
        ),
        "phase_offset_rad": float(
            synchronized_signal.metadata.get("phase_offset", 0.0)
        ),
        "phase_confidence": float(
            synchronized_signal.metadata.get("phase_confidence", 0.0)
        ),
    }

    # ---- fine classification on synchronized symbols ----
    fine = classify_signal(
        synchronized_signal, use_constellation=True, synchronized=True
    )

    if fine.modulation == "Unknown" and coarse.modulation != "Unknown":
        fine = coarse  # fall back to coarse evidence rather than forcing Unknown
        warnings.append(
            "Synchronized-symbol classification returned Unknown; "
            "using coarse waveform result."
        )

    classification_summary = {
        "modulation": fine.modulation,
        "confidence": float(fine.confidence),
        "method": fine.method,
        "stage": "fine",
        "coarse_result": {
            "modulation": coarse.modulation,
            "confidence": float(coarse.confidence),
        },
    }

    return (
        classification_summary,
        symbol_rate_summary,
        synchronization_summary,
        synchronized_signal,
    )


def _demodulate(
    modulation: str,
    isolated: Signal,
    synchronized_signal: Signal | None,
    symbol_rate: dict[str, Any] | None,
    config: AnalysisConfig,
    provenance,
    warnings: list[str],
    capture_symbol_samples: bool = False,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    """Demodulate according to classification. Returns (summary, bits)."""
    from prototype.demodulation.demodulator import demodulate_signal

    if modulation in (None, "Unknown"):
        return None, None

    if modulation == "BFSK":
        if symbol_rate is None or symbol_rate.get("samples_per_symbol", 0) <= 0:
            warnings.append("BFSK demodulation skipped: no symbol rate available.")
            return None, None

        # Estimate the two tone offsets relative to the isolation center.
        from prototype.dsp.spectral import welch_psd

        freqs, psd = welch_psd(
            isolated.samples, isolated.sample_rate, nperseg=min(2048, isolated.num_samples)
        )
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(psd, prominence=float(np.max(psd)) * 0.01, distance=4)
        if peaks.size < 2:
            warnings.append("BFSK demodulation skipped: two tones not found.")
            return None, None
        strongest = peaks[np.argsort(psd[peaks])[::-1][:2]]
        f0, f1 = sorted(float(freqs[i]) for i in strongest)

        sps_int = max(2, int(round(symbol_rate["samples_per_symbol"])))
        result = demodulate_signal(
            isolated,
            modulation="BFSK",
            samples_per_symbol=float(sps_int),
            freq_0=f0,
            freq_1=f1,
        )
        # Note: demodulate_bfsk slices at integer sps boundaries; with a
        # fractional sps estimate, symbol-boundary drift accumulates over
        # long captures. Report this limitation honestly in the summary.
        sps_actual = float(symbol_rate["samples_per_symbol"])
        if abs(sps_actual - sps_int) > 0.02 * sps_int:
            warnings.append(
                f"BFSK samples-per-symbol {sps_actual:.2f} rounded to "
                f"{sps_int} for demodulation; boundary drift possible "
                "on long captures."
            )
        summary = {
            "modulation": "BFSK",
            "num_symbols": int(result.num_symbols),
            "num_bits": int(result.num_bits),
            "decision_margin": float(result.decision_margin),
            "tone_0_hz": float(f0),
            "tone_1_hz": float(f1),
        }
        return summary, np.asarray(result.bits)

    if modulation in ("8-PSK", "OOK", "ASK"):
        # These demodulators run their own symbol extraction on the
        # isolated waveform (they need sps, not pre-extracted symbols).
        if symbol_rate is None or symbol_rate.get("samples_per_symbol", 0) <= 0:
            warnings.append(
                f"{modulation} demodulation skipped: no symbol rate available."
            )
            return None, None
        sps = float(symbol_rate["samples_per_symbol"])
        result = demodulate_signal(
            isolated,
            modulation=modulation,
            samples_per_symbol=sps,
        )
        summary = {
            "modulation": modulation,
            "num_symbols": int(result.num_symbols),
            "num_bits": int(result.num_bits),
            "decision_margin": float(result.decision_margin),
        }
        return summary, np.asarray(result.bits)

    if synchronized_signal is None:
        warnings.append(
            f"{modulation} demodulation skipped: synchronization failed."
        )
        return None, None

    # 16-QAM demodulation hard-decides on pre-extracted symbol
    # samples. The synchronized signal is already at ~1 sample per
    # symbol, so extraction must not decimate again — the raw sps
    # estimate (e.g. 80) here decimated the stream a second time and
    # yielded ~n_symbols/80 decisions.
    demod_kwargs: dict[str, Any] = {"synchronized": True}

    if modulation == "16-QAM":
        demod_kwargs["samples_per_symbol"] = 1.0

    result = demodulate_signal(
        synchronized_signal,
        modulation=modulation,
        **demod_kwargs,
    )

    constellation_summary: dict[str, Any] = _summary_symbols(
        synchronized_signal.samples
    )

    if capture_symbol_samples:

        symbols = np.asarray(synchronized_signal.samples)

        if symbols.size > 4096:

            symbols = symbols[:: int(np.ceil(symbols.size / 4096))]

        constellation_summary["symbols"] = [
            [float(s.real), float(s.imag)] for s in symbols
        ]

    summary = {
        "modulation": modulation,
        "num_symbols": int(result.num_symbols),
        "num_bits": int(result.num_bits),
        "decision_margin": float(result.decision_margin),
        "constellation": constellation_summary,
    }
    return summary, np.asarray(result.bits)


def _resolve_codeword_alignment(
    modulation: str | None,
    bits: np.ndarray,
    reference_bits: np.ndarray,
    synchronized_signal: Signal | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the receiver's blind ambiguity against a transmitted reference.

    Mirrors the ambiguity model used by the BER stage (polarity / QPSK
    rotation / 16-QAM rotation x symbol-origin) but *returns the bitstream*,
    so the deinterleaver and FEC decoder can run on an aligned input.

    A candidate is only accepted when it is a strict improvement over the
    raw demodulated stream against the reference (fewer bit errors and a
    BER below 0.25); otherwise the raw bits are returned with
    ``aligned=False``.  That guard keeps a reference that does not describe
    the demodulated domain (e.g. a payload reference against an FEC-coded
    stream) from silently mangling the stream: every candidate scores ~0.5
    there, so nothing is accepted.
    """
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    reference = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)

    def _errors(candidate: np.ndarray) -> tuple[int, int]:
        n = min(candidate.size, reference.size)
        return int(np.count_nonzero(candidate[:n] != reference[:n])), n

    raw_errors, raw_compared = _errors(arr)
    info: dict[str, Any] = {
        "aligned": False,
        "modulation": modulation,
        "raw_bit_errors": raw_errors,
        "raw_compared_bits": raw_compared,
        "candidates_tried": 0,
    }

    if raw_compared == 0 or synchronized_signal is None:
        info["reason"] = "no synchronized symbols or empty reference"
        return arr, info

    best_bits = arr
    best_errors = raw_errors
    best_label = "raw demodulated bits"
    best_ber = raw_errors / raw_compared

    def _consider(candidate: np.ndarray, label: str) -> None:
        nonlocal best_bits, best_errors, best_label, best_ber
        info["candidates_tried"] = int(info["candidates_tried"]) + 1
        errors, compared = _errors(candidate)
        if compared == 0:
            return
        candidate_ber = errors / compared
        if errors < best_errors and candidate_ber < best_ber:
            best_bits = candidate
            best_errors = errors
            best_label = label
            best_ber = candidate_ber

    symbols = np.asarray(synchronized_signal.samples)

    if modulation in ("QPSK", "16-QAM"):
        # Square PSK/QAM mappings leave a 90-degree phase fold and an
        # integer-symbol origin ambiguity (pulse-shaping group delay), so
        # both are searched in whole symbols.  This reuses the demodulator
        # helpers the BER stage uses, so the resolved stream is exactly the
        # one BER reports on.
        if modulation == "QPSK":
            from prototype.modulation.demodulator import qpsk_decision

            decision = qpsk_decision
            symbols_for_search = symbols
            bits_per_symbol = 2
        else:
            from prototype.modulation.demodulator import (
                normalize_qam16_symbols,
                qam16_decision,
            )

            decision = qam16_decision
            symbols_for_search = normalize_qam16_symbols(symbols)
            bits_per_symbol = 4

        max_offset = min(16, max(0, symbols_for_search.size - 1))

        for k in range(4):
            rotated = symbols_for_search * np.exp(1j * k * np.pi / 2.0)
            rot_bits, _, _ = decision(rotated)
            rot_bits = np.asarray(rot_bits, dtype=np.uint8)
            if rot_bits.size == 0:
                continue
            for inverted in (0, 1):
                candidate = rot_bits ^ 1 if inverted else rot_bits
                for sym_off in range(-max_offset, max_offset + 1):
                    shift = bits_per_symbol * sym_off
                    shifted = candidate[shift:] if shift >= 0 else candidate
                    if shifted.size == 0:
                        continue
                    label = (
                        f"{modulation} rotation {k * 90} deg, "
                        f"origin {sym_off:+d} symbols"
                        f"{' + inversion' if inverted else ''}"
                    )
                    _consider(shifted, label)

    else:
        # Polarity is the only blind ambiguity for the remaining mappings.
        _consider(arr ^ 1, "polarity inversion")

    accepted = best_errors < raw_errors and best_ber < 0.25

    info.update(
        {
            "aligned": bool(accepted),
            "label": best_label,
            "aligned_bit_errors": int(best_errors),
            "aligned_compared_bits": int(raw_compared),
            "aligned_ber": float(best_ber),
            "reference_domain": "transmitted bits as demodulated (pre-deinterleave)",
        }
    )
    if not accepted:
        info["reason"] = (
            "no candidate improved on the raw demodulated stream "
            f"(raw bit errors {raw_errors}/{raw_compared})"
        )
        return arr, info

    return best_bits, info


def _evaluate_ber(
    modulation: str,
    bits: np.ndarray | None,
    reference_bits: np.ndarray | None,
    synchronized_signal: Signal | None,
    provenance,
    warnings: list[str],
) -> dict[str, Any] | None:
    """
    BER against a reference, handling blind ambiguity honestly:
    BPSK polarity and QPSK 90-degree rotation are searched and the
    best case reported with the applied correction noted.
    """
    from prototype.modulation.demodulator import calculate_ber, qpsk_decision

    if bits is None or reference_bits is None:
        return None

    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    reference_bits = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)

    if modulation == "BPSK":
        report = calculate_ber(bits, reference_bits)
        best = min(report["direct_ber"], report["inverted_ber"])
        return {
            "modulation": "BPSK",
            "compared_bits": int(report["compared_bits"]),
            "bit_errors": int(
                report["direct_errors"]
                if report["direct_ber"] <= report["inverted_ber"]
                else report["inverted_errors"]
            ),
            "ber": float(best),
            "ambiguity_resolution": "polarity search (0/180 deg)",
        }

    if modulation == "QPSK" and synchronized_signal is not None:
        symbols = synchronized_signal.samples
        best_ber = 1.0
        best_rotation = 0.0
        best_errors = None
        for k in range(4):
            rotated = symbols * np.exp(1j * k * np.pi / 2.0)
            rot_bits, _, _ = qpsk_decision(rotated)
            report = calculate_ber(rot_bits, reference_bits)
            candidate_ber = min(report["direct_ber"], report["inverted_ber"])
            if candidate_ber < best_ber:
                best_ber = candidate_ber
                best_rotation = float(k * 90.0)
                best_errors = (
                    report["direct_errors"]
                    if report["direct_ber"] <= report["inverted_ber"]
                    else report["inverted_errors"]
                )
        compared = min(len(bits), len(reference_bits))
        return {
            "modulation": "QPSK",
            "compared_bits": int(compared),
            "bit_errors": int(best_errors) if best_errors is not None else None,
            "ber": float(best_ber),
            "ambiguity_resolution": (
                f"rotation search (applied {best_rotation:.0f} deg)"
            ),
        }

    if modulation == "16-QAM" and synchronized_signal is not None:
        # Square 16-QAM lattices are invariant under 90-degree rotations
        # as a point set, so blind carrier recovery leaves an
        # unobservable phase fold — but each fold maps the Gray labels
        # to different bits (unlike the geometry, which cannot
        # distinguish them). The blind receiver also cannot know which
        # symbol is "first" (frame origin), so with a reference stream
        # available, search folds x symbol offsets and report the best.
        from prototype.modulation.demodulator import (
            normalize_qam16_symbols,
            qam16_decision,
        )

        symbols = normalize_qam16_symbols(synchronized_signal.samples)

        best_ber = 1.0
        best_rotation = 0.0
        best_offset = 0
        best_errors = None

        # Symbol-aligned origin search. The recovered symbol stream's
        # origin is ambiguous by an integer number of SYMBOLS (pulse-
        # shaping group delay alone offsets it by span_symbols/2, e.g.
        # 8 symbols for an 8-span RRC), so the search must step in
        # whole symbols (4 bits each), in both directions, over at
        # least the filter half-span. A bit-granular search over a
        # few bits cannot reach the true alignment.
        max_offset = min(16, symbols.size - 1)

        for k in range(4):

            rotated = symbols * np.exp(1j * k * np.pi / 2.0)
            rot_bits, _, _ = qam16_decision(rotated)

            for sym_off in range(-max_offset, max_offset + 1):

                if sym_off >= 0:
                    shifted = rot_bits[4 * sym_off :]
                    reference = reference_bits
                else:
                    shifted = rot_bits
                    reference = reference_bits[4 * (-sym_off) :]

                report = calculate_ber(shifted, reference)

                candidate_ber = min(
                    report["direct_ber"], report["inverted_ber"]
                )

                if candidate_ber < best_ber:
                    best_ber = candidate_ber
                    best_rotation = float(k * 90.0)
                    best_offset = sym_off
                    best_errors = (
                        report["direct_errors"]
                        if report["direct_ber"] <= report["inverted_ber"]
                        else report["inverted_errors"]
                    )

        compared = min(len(bits), len(reference_bits))

        return {
            "modulation": "16-QAM",
            "compared_bits": int(compared),
            "bit_errors": int(best_errors) if best_errors is not None else None,
            "ber": float(best_ber),
            "ambiguity_resolution": (
                f"rotation x origin search "
                f"(applied {best_rotation:.0f} deg, {best_offset:+d} symbols)"
            ),
        }

    # Remaining modulations: direct comparison (RMS-normalized constellation).
    report = calculate_ber(bits, reference_bits)
    return {
        "modulation": modulation,
        "compared_bits": int(report["compared_bits"]),
        "bit_errors": int(report["direct_errors"]),
        "ber": float(report["direct_ber"]),
        "ambiguity_resolution": "none (unambiguous mapping)",
    }


# ============================================================
# MAIN ENTRY POINTS
# ============================================================


def analyze_samples(
    samples: np.ndarray,
    sample_rate: float,
    config: AnalysisConfig | None = None,
    mode: str = "balanced",
    reference_bits: np.ndarray | None = None,
    input_info: dict[str, Any] | None = None,
    capture_symbol_samples: bool = False,
) -> AnalysisResult:
    """
    Run the full analysis chain on in-memory complex samples.

    Parameters
    ----------
    samples :
        Complex baseband/IF samples.
    sample_rate :
        Sample rate in Hz.
    config :
        Full configuration; when omitted, built from ``mode``.
    mode :
        "quick" | "balanced" | "deep" | "realtime" (used when config is None).
    reference_bits :
        Optional transmitted bit reference for BER measurement.
    capture_symbol_samples :
        When True, attach a downsampled copy of the synchronized symbol
        stream to ``result.demodulation["constellation"]["symbols"]``
        (max 4096 points, JSON-safe [real, imag] pairs). Used by the GUI
        to render the *recovered* constellation, which is the actual
        analysis deliverable; off by default so bulk/batch runs stay
        small.
    """
    if config is None:
        config = processing_mode_config(mode)

    provenance = new_provenance(configuration=config_to_dict(config))
    result = AnalysisResult(
        input_info=dict(input_info or {}),
        warnings=[],
    )
    result.input_info.setdefault("sample_rate", float(sample_rate))
    result.input_info.setdefault("num_samples", int(np.asarray(samples).size))

    try:
        signal = Signal(samples=np.asarray(samples), sample_rate=sample_rate)
    except ValueError as exc:
        provenance.finish()
        result.provenance = provenance_to_dict(provenance)
        raise PipelineError(f"Input validation failed: {exc}") from exc

    # Keep a raw copy; preprocessing produces a derived signal.
    raw_signal = signal.copy()

    # ---- preprocessing --------------------------------------------
    with provenance.record_step("preprocessing") as step:
        from prototype.core.preprocessor import preprocess_signal

        processed = preprocess_signal(signal)
        step["dc_magnitude"] = processed.metadata.get("dc_magnitude", 0.0)
        step["snr_db"] = processed.metadata.get("snr_db")

    # ---- detection --------------------------------------------------
    candidates = []
    with provenance.record_step("detection") as step:
        from prototype.detection.detector import detect_candidates

        candidates = detect_candidates(
            raw_signal,
            threshold_db=config.detection.threshold_db,
            min_bandwidth_hz=config.detection.min_bandwidth_hz,
            smoothing_window_bins=config.detection.smoothing_window_bins,
            prominence_db=config.detection.prominence_db,
            min_peak_distance_bins=config.detection.min_peak_distance_bins,
            merge_gap_bins=config.detection.merge_gap_bins,
        )
        result.detections = [c.summary() for c in candidates]
        step["candidates"] = len(candidates)

    if not candidates:
        result.warnings.append(
            "No signal candidates detected; pipeline ends after detection."
        )
        provenance.finish()
        result.provenance = provenance_to_dict(provenance)
        return result

    # ---- candidate selection -----------------------------------------
    index = min(max(0, config.candidate_index), len(candidates) - 1)
    selected = candidates[index]
    result.selected_candidate = selected.summary()

    # ---- isolation / DDC ----------------------------------------------
    with provenance.record_step("isolation") as step:
        from prototype.core.isolator import isolate_signal

        isolation_result = isolate_signal(
            raw_signal,
            center_frequency=selected.center_frequency,
            bandwidth=selected.bandwidth,
            filter_margin=config.isolation.filter_margin,
            filter_order=config.isolation.filter_order,
        )
        isolated = isolation_result.signal
        result.isolation = isolation_result.summary()
        step["cutoff_hz"] = isolation_result.high_cutoff

    # ---- parameter extraction -------------------------------------------
    with provenance.record_step("parameter_extraction") as step:
        try:
            from prototype.parameters.extractor import (
                extract_parameters,
                parameters_to_dict,
            )

            params = extract_parameters(isolated)
            result.parameters = parameters_to_dict(params)
            step["snr_db"] = params.snr_db
        except Exception as exc:
            logger.exception("parameter extraction failed")
            result.warnings.append(f"Parameter extraction failed: {exc}")
            result.parameters = None

    # ---- classification + symbol rate + synchronization -----------------
    with provenance.record_step("classification_sync") as step:
        try:
            (
                classification_summary,
                symbol_rate_summary,
                synchronization_summary,
                synchronized_signal,
            ) = _classify_and_sync(isolated, config, provenance, result.warnings)
            result.classification = classification_summary
            result.symbol_rate = symbol_rate_summary
            result.synchronization = synchronization_summary
            step["modulation"] = (
                classification_summary or {}
            ).get("modulation", "Unknown")
        except Exception as exc:  # graceful degradation
            logger.exception("classification/sync stage failed")
            result.warnings.append(f"Classification/synchronization failed: {exc}")
            classification_summary = None
            synchronized_signal = None
            symbol_rate_summary = None

    modulation = (result.classification or {}).get("modulation")

    # ---- demodulation ------------------------------------------------------
    with provenance.record_step("demodulation") as step:
        try:
            demod_summary, bits = _demodulate(
                modulation,
                isolated,
                synchronized_signal,
                symbol_rate_summary,
                config,
                provenance,
                result.warnings,
                capture_symbol_samples=capture_symbol_samples,
            )
            result.demodulation = demod_summary
            if demod_summary:
                step["num_bits"] = demod_summary.get("num_bits", 0)
        except Exception as exc:
            logger.exception("demodulation stage failed")
            result.warnings.append(f"Demodulation failed: {exc}")
            bits = None

    # Preserve the received (post-demodulation, pre-deinterleaving)
    # bitstream: the result must always carry the original demodulated
    # bits regardless of what the interleaving/FEC stages do next.
    if bits is not None:
        result.demodulation = result.demodulation or {}
        result.demodulation["received_bits"] = bits

    # ---- codeword alignment (transmitted-bit reference available) --------
    # Blind carrier/timing recovery leaves an unobservable phase fold and
    # a symbol-origin offset (pulse-shaping group delay).  When the caller
    # supplies the transmitted bitstream as a reference (the same reference
    # BER uses) the receiver can resolve that ambiguity *before* the
    # deinterleaver/decoder run, which is what makes the coded chain
    # decodable at all.  Without a reference the raw demodulated stream is
    # passed on unchanged and this is recorded honestly.
    if bits is not None and reference_bits is not None:
        with provenance.record_step("codeword_alignment") as step:
            try:
                aligned_bits, alignment_info = _resolve_codeword_alignment(
                    modulation, bits, reference_bits, synchronized_signal
                )
                result.demodulation = result.demodulation or {}
                result.demodulation["codeword_alignment"] = alignment_info
                step["aligned"] = bool(alignment_info.get("aligned"))
                step["label"] = alignment_info.get("label")
                if alignment_info.get("aligned"):
                    bits = aligned_bits
                    result.demodulation["aligned_bits"] = bits
            except Exception as exc:  # never fatal: fall back to raw bits
                logger.exception("codeword alignment failed")
                result.demodulation = result.demodulation or {}
                result.demodulation["codeword_alignment"] = {
                    "aligned": False,
                    "reason": f"alignment failed: {exc}",
                }

    # ---- block-interleaving identification / deinterleaving --------------
    # Demodulated ``bits`` are the received bitstream after the
    # row-column block interleaver (if the transmitter used one).
    #
    # AUTO: run the structural identifier. Only a genuine, depth-
    # discriminative AUTO_DETECTED result may cause deinterleaving;
    # without structural evidence or on insufficient evidence the bits
    # stay unchanged and no depth is forced.
    # MANUAL: apply the configured interleave_depth authoritatively;
    # no identification is run.
    # NONE: bits stay untouched.
    with provenance.record_step("interleaving_identification") as step:
        try:
            fec_cfg = config.fec

            if fec_cfg.interleaving_mode == FECMode.AUTO:
                interleave_result = identify_interleaving(
                    bits, min_depth=2, max_depth=16
                )
                result.warnings.extend(interleave_result.warnings)

                if (
                    interleave_result.status == AUTO_DETECTED
                    and interleave_result.best_depth is not None
                ):
                    # Genuine structural evidence: deinterleave on a copy
                    # so the received ``bits`` are never mutated.
                    result.demodulation["interleaving_result"] = interleave_result.to_dict()
                    deinterleaved = deinterleave_bits(
                        bits,
                        depth=int(interleave_result.best_depth),
                        original_size=int(bits.size),
                    )
                    bits = deinterleaved
                    result.demodulation["deinterleaved_bits"] = bits
                    step["status"] = AUTO_DETECTED
                    step["best_depth"] = int(interleave_result.best_depth)
                    step["confidence"] = float(interleave_result.confidence)
                else:
                    # No structural evidence: bits unchanged; no depth forced.
                    # Record the identifier result so callers can inspect
                    # whether AUTO ran and what it found.
                    step["status"] = interleave_result.status
                    step["best_depth"] = interleave_result.best_depth
                    result.demodulation["interleaving_result"] = interleave_result.to_dict()
            elif fec_cfg.interleaving_mode == FECMode.MANUAL:
                # MANUAL is authoritative: apply the configured family and
                # depth directly (block / convolutional / diagonal /
                # pseudo-random), with no identification run.
                family = fec_cfg.interleave_family
                padded = 0
                # Interleaver geometry is defined by the TRANSMITTED frame
                # size, so a capture whose tail was lost cannot be
                # deinterleaved on its own (the permutation changes with
                # the length).  When the transmitted reference is available
                # the receiver zero-fills the missing tail back to the
                # transmitted length - the standard partial-frame handling -
                # and reports how many bits were filled; the affected
                # codeword is then reported as uncorrectable instead of
                # being silently mis-decoded.
                if (
                    reference_bits is not None
                    and np.asarray(reference_bits).size > bits.size
                ):
                    padded = int(np.asarray(reference_bits).size - bits.size)
                    bits = np.concatenate(
                        [bits, np.zeros(padded, dtype=np.uint8)]
                    )
                    result.demodulation["zero_filled_bits"] = padded
                applicable, reason = fec_cfg.manual_deinterleave_plan(int(bits.size))
                result.demodulation["interleaving_result"] = {
                    "status": "MANUALLY_CONFIGURED",
                    "best_depth": fec_cfg.interleave_depth,
                    "family": family,
                    "confidence": 1.0,
                    "candidates": [],
                    "applicable": applicable,
                    "evidence": {
                        "mode": "manual_configured_family_apply",
                        "family": family,
                        "depth": fec_cfg.interleave_depth,
                        "applicable": applicable,
                        "reason": reason,
                        "zero_filled_bits": padded,
                    },
                }
                if applicable:
                    if family == "block":
                        # Keep the block path on the pipeline's own
                        # deinterleave_bits reference for behaviour parity.
                        bits = deinterleave_bits(
                            bits,
                            depth=int(fec_cfg.interleave_depth),
                            original_size=int(bits.size),
                        )
                    else:
                        bits = fec_cfg.deinterleave_bits(bits)
                    result.demodulation["deinterleaved_bits"] = bits
                else:
                    # Inapplicable plan: keep the received bits untouched and
                    # surface the reason instead of silently passing through.
                    result.warnings.append(
                        f"Manual {family} deinterleaving not applied: {reason}"
                    )
                step["status"] = "MANUALLY_CONFIGURED"
                step["best_depth"] = fec_cfg.interleave_depth
                step["family"] = family
            elif fec_cfg.interleaving_mode == FECMode.NONE:
                # NONE skips interleaving entirely; bits stay as received.
                result.demodulation["interleaving_result"] = {
                    "status": "NONE",
                    "best_depth": None,
                    "confidence": 0.0,
                    "candidates": [],
                    "evidence": {"mode": "none"},
                }
                step["status"] = "NONE"
            else:
                # AUTO with structurally weak / unresolved evidence: no
                # deinterleaving, no forced depth. ``bits`` remains the
                # received demodulated bitstream for BER and the FEC path.
                result.demodulation["interleaving_result"] = {
                    "status": getattr(interleave_result, "status", "UNRESOLVED"),
                    "best_depth": getattr(interleave_result, "best_depth", None),
                    "confidence": float(getattr(interleave_result, "confidence", 0.0)),
                    "candidates": list(getattr(interleave_result, "candidates", [])),
                    "evidence": dict(getattr(interleave_result, "evidence", {})),
                }
                step["status"] = getattr(interleave_result, "status", "UNRESOLVED")
        except Exception as exc:
            result.warnings.append(f"Interleaving identification failed: {exc}")

    # ---- BER ----------------------------------------------------------------
    if bits is not None and reference_bits is not None:
        with provenance.record_step("ber") as step:
            result.ber = _evaluate_ber(
                modulation,
                bits,
                reference_bits,
                synchronized_signal,
                provenance,
                result.warnings,
            )
            if result.ber:
                step["ber"] = result.ber["ber"]

    # ---- automatic FEC identification (AUTO mode only) ---------------------
    # The identifier is a *candidate evaluator*: it runs the existing
    # decoders to score hypotheses and only reports a scheme when the
    # evidence clears its internal confidence rule.  UNKNOWN/UNRESOLVED
    # is a valid, honest outcome and never a guess.
    if bits is not None and config.fec.identification_runs():
        with provenance.record_step("fec_identification") as step:
            try:
                from prototype.fec.identification import identify_fec

                identification = identify_fec(bits, reference_bits)

                result.demodulation = result.demodulation or {}
                result.demodulation[
                    "fec_identification"
                ] = identification.to_dict()

                step["status"] = identification.status
                step["best_scheme"] = identification.best_scheme
                step["confidence"] = float(identification.confidence)

                if (
                    identification.status == "AUTO_DETECTED"
                    and identification.best_scheme
                ):
                    # Decode a copy of the bits; the received stream (and
                    # the deinterleaved stream) are never mutated.
                    from prototype.fec import decode_bits

                    # ``trim_partial_codeword`` drops the truncated tail a
                    # real capture always has and reports how much: decoding
                    # whole codewords only, never inventing the missing bits.
                    decoded, fec_result = decode_bits(
                        np.asarray(bits).copy(),
                        identification.best_scheme,
                        trim_partial_codeword=True,
                    )
                    result.demodulation["fec"] = {
                        "scheme": fec_result.scheme,
                        "corrected_errors": fec_result.corrected_errors,
                        "uncorrectable_blocks": (
                            fec_result.uncorrectable_blocks
                        ),
                        # The recovered bitstream itself (never the count).
                        "decoded_bits": decoded,
                        "decoded_bit_count": int(fec_result.output_bits),
                        "trimmed_tail_bits": int(
                            (fec_result.extra or {}).get("trimmed_tail_bits", 0)
                        ),
                        "source": "auto_identified",
                    }
                    step["decoded"] = True
                else:
                    step["decoded"] = False
            except Exception as exc:
                result.warnings.append(f"FEC identification failed: {exc}")

    # ---- explicit FEC decode (MANUAL mode) ---------------------------------
    if config.fec.scheme and bits is not None:
        with provenance.record_step("fec_decode") as step:
            try:
                from prototype.fec import decode_bits

                decoded, fec_result = decode_bits(
                    bits,
                    config.fec.scheme,
                    trim_partial_codeword=True,
                )
                result.demodulation = result.demodulation or {}
                result.demodulation["fec"] = {
                    "scheme": fec_result.scheme,
                    "corrected_errors": fec_result.corrected_errors,
                    "uncorrectable_blocks": fec_result.uncorrectable_blocks,
                    # The recovered bitstream itself (never the count).
                    "decoded_bits": decoded,
                    "decoded_bit_count": int(fec_result.output_bits),
                    "trimmed_tail_bits": int(
                        (fec_result.extra or {}).get("trimmed_tail_bits", 0)
                    ),
                    "source": "explicit_config",
                }
                step["scheme"] = config.fec.scheme
            except Exception as exc:
                result.warnings.append(f"FEC decode failed: {exc}")

    # ---- ML classification + fusion (optional second opinion) ----------------
    # A CNN scores the isolated candidate's raw IQ as supplementary evidence;
    # it never overrides or gates the deterministic DSP classification.
    if config.ml.enabled:
        with provenance.record_step("ml_classification") as step:
            try:
                from prototype.ml.cnn import predict_modulation
                from prototype.ml.fusion import fuse_classification

                ml_result = predict_modulation(
                    isolated.samples, labels_json=config.ml.labels_json
                )
                if ml_result is not None:
                    result.ml = ml_result
                    step["predicted_class"] = ml_result["predicted_class"]
                    step["confidence"] = ml_result["confidence"]
                else:
                    result.warnings.append(
                        "ML stage skipped: capture too short for one "
                        "512-sample frame or artifact unavailable."
                    )
                    ml_result = None

                # Deterministic DSP classification is already stored on the
                # result; record it here as well for the fusion record.
                dsp_modulation = (
                    result.classification or {}
                ).get("modulation", "Unknown")
                fusion = fuse_classification(
                    dsp_modulation=dsp_modulation,
                    ml_prediction=ml_result,
                    ml_active=config.ml.enabled,
                    fusion=config.ml.fusion,
                )
                result.fusion = fusion
                step["fusion_method"] = fusion.get("method", "side_by_side")
                step["dsp_modulation"] = dsp_modulation
                step["ml_modulation"] = fusion.get("ml_modulation")
                step["final_modulation"] = fusion.get("final_modulation")
                step["disagreement"] = fusion.get("disagreement")
            except Exception as exc:
                result.warnings.append(f"ML classification/fusion failed: {exc}")

    # ---- protocol / frame analysis (explicit configuration only) -----
    # Frames are decoded from the demodulated (optionally FEC-received)
    # bit stream.  This stage is never inferred: it runs only when a
    # FrameConfig is attached to AnalysisConfig.protocol.  Unknown is a
    # first-class result when no sync word is found.
    #
    # Runs after the FEC pass so a receiver that de-codes arrives at
    # this stage as plain bits, ready to match against a sync word.
    if config.protocol is not None:
        with provenance.record_step("protocol") as step:
            try:
                from prototype.protocol.semantics import decode_frame_semantics

                # Demodulated bits (post-FEC, post-mask).
                stream_bits = bits
                if stream_bits is None:
                    result.protocol = None
                    result.warnings.append(
                        "Protocol stage skipped: no bits available."
                    )
                else:
                    result.protocol = decode_frame_semantics(
                        bits=stream_bits,
                        sync_word=config.protocol.sync_word,
                        sync_word_capacity=config.protocol.sync_word_capacity,
                        data_bytes=config.protocol.data_bytes,
                        crc_poly=config.protocol.crc[0] if config.protocol.crc else None,
                        crc_final_xor=config.protocol.crc[1] if config.protocol.crc else None,
                        crc_orientation=config.protocol.crc_orientation,
                        header_len_bytes=config.protocol.header_len_bytes,
                    )
                    step["protocol"] = result.protocol.protocol
                    step["sync_found"] = bool(result.protocol.sync_found)
            except Exception as exc:
                result.warnings.append(f"Protocol stage failed: {exc}")
                result.protocol = None

    provenance.finish()
    result.provenance = provenance_to_dict(provenance)
    return result


def analyze_capture(
    path: str | Path,
    mode: str = "balanced",
    candidate_index: int = 0,
    reference_bits_path: str | Path | None = None,
    config: AnalysisConfig | None = None,
    protocol: Any | None = None,
    interleaving_mode: str | None = None,
    interleave_depth: int | None = None,
    interleave_family: str | None = None,
    fec_mode: str | None = None,
    fec_scheme: str | None = None,
    ml_enabled: bool | None = None,
    ml_fusion: str | None = None,
    labels_path: str | Path | None = None,
    **load_overrides,
) -> AnalysisResult:
    """
    Load a capture file and run the full analysis chain.

    ``load_overrides`` are forwarded to :func:`io.loaders.load_signal`
    (e.g. ``sample_rate``, ``dtype``, ``endianness`` for raw IQ files).

    ``protocol`` / ``interleaving_mode`` / ``interleave_depth`` /
    ``fec_mode`` / ``fec_scheme`` are explicit configuration overrides
    mapped onto ``AnalysisConfig``; they are never inferred from data.

    A BER reference is loaded automatically from a companion
    ``<file>.reference.npz`` (array key ``bits``) when present, or from
    an explicit ``reference_bits_path``.
    """
    from prototype.io.loaders import load_signal

    file_path = Path(path)
    if not file_path.is_file():
        raise PipelineError(f"Capture file not found: {file_path}")

    signal = load_signal(file_path, **load_overrides)

    reference_bits = None
    if reference_bits_path is not None:
        with np.load(reference_bits_path, allow_pickle=False) as data:
            reference_bits = np.asarray(data["bits"], dtype=np.uint8)
    else:
        companion = file_path.with_name(f"{file_path.stem}.reference.npz")
        if companion.is_file():
            with np.load(companion, allow_pickle=False) as data:
                reference_bits = np.asarray(data["bits"], dtype=np.uint8)

    from dataclasses import replace

    if config is None:
        config = processing_mode_config(mode)

    config = replace(config, candidate_index=candidate_index)

    # Explicit FEC / interleaving / protocol overrides.  These map onto
    # the existing AnalysisConfig fields and are validated by them
    # (an invalid mode raises rather than being silently ignored).
    fec_overrides: dict[str, Any] = {}
    if fec_mode is not None:
        fec_overrides["mode"] = str(fec_mode).strip().lower()
    if fec_scheme is not None:
        fec_overrides["scheme"] = fec_scheme
    if interleaving_mode is not None:
        fec_overrides["interleaving_mode"] = str(interleaving_mode).strip().lower()
    if interleave_depth is not None:
        fec_overrides["interleave_depth"] = int(interleave_depth)
    if interleave_family is not None:
        fec_overrides["interleave_family"] = str(interleave_family).strip().lower()

    if fec_overrides:
        config = replace(config, fec=replace(config.fec, **fec_overrides))

    if protocol is not None:
        config = replace(config, protocol=protocol)

    # Optional ML stage overrides (CLI `--ml/--ml-fusion/--labels`).  The
    # CNN never overrides the DSP classification; it only adds evidence.
    ml_overrides: dict[str, Any] = {}
    if ml_enabled:
        ml_overrides["enabled"] = True
    if ml_fusion is not None:
        ml_overrides["fusion"] = replace(
            config.ml.fusion, method=str(ml_fusion).strip().lower()
        )
    if labels_path is not None:
        ml_overrides["labels_json"] = str(labels_path)
    if ml_overrides:
        config = replace(config, ml=replace(config.ml, **ml_overrides))

    capture_info = signal.metadata.get("capture", {})
    return analyze_samples(
        samples=signal.samples,
        sample_rate=signal.sample_rate,
        config=config,
        reference_bits=reference_bits,
        input_info={
            "file": str(file_path),
            "format": capture_info.get("file_format", "memory"),
            "center_frequency_hz": capture_info.get("center_frequency_hz"),
        },
    )
