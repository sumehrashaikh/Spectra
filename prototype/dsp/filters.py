"""Filter design and application: FIR (Kaiser), IIR (SOS), polyphase resampling."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

try:
    from scipy.signal import (
        butter,
        cheby1,
        cheby2,
        ellip,
        firwin,
        resample_poly,
        sosfilt,
        sosfiltfilt,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for the dsp.filters module.") from exc


def _validate_rate(sample_rate: float) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return float(sample_rate)


def _normalized(cutoff_hz: float, sample_rate: float) -> float:
    nyquist = sample_rate / 2.0
    wn = float(cutoff_hz) / nyquist
    if not 0 < wn < 1:
        raise ValueError(
            f"Cutoff {cutoff_hz} Hz must be between 0 and Nyquist ({nyquist} Hz)."
        )
    return wn


def fir_lowpass(
    numtaps: int,
    cutoff_hz: float,
    sample_rate: float,
    transition_hz: float | None = None,
) -> np.ndarray:
    """Design a linear-phase FIR low-pass using a Kaiser window.

    If ``transition_hz`` is given, the Kaiser beta is derived from the
    desired transition width assuming ~80 dB stop-band attenuation cap.
    """
    _validate_rate(sample_rate)
    if numtaps < 4:
        raise ValueError("numtaps must be >= 4.")
    if transition_hz is not None and transition_hz > 0:
        nyquist = sample_rate / 2.0
        width = transition_hz / nyquist
        beta = min(8.7, 0.1102 * (80.0 - 8.7))  # ~80 dB design
        return firwin(numtaps, _normalized(cutoff_hz, sample_rate),
                      window=("kaiser", beta))
    return firwin(numtaps, _normalized(cutoff_hz, sample_rate))


def fir_highpass(numtaps: int, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Design a linear-phase FIR high-pass (Kaiser window, odd numtaps)."""
    _validate_rate(sample_rate)
    if numtaps % 2 == 0:
        numtaps += 1
    return firwin(numtaps, _normalized(cutoff_hz, sample_rate), pass_zero=False)


def fir_bandpass(
    numtaps: int,
    low_hz: float,
    high_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """Design a linear-phase FIR band-pass (Kaiser window, odd numtaps)."""
    _validate_rate(sample_rate)
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")
    if numtaps % 2 == 0:
        numtaps += 1
    nyquist = sample_rate / 2.0
    return firwin(numtaps, [low_hz / nyquist, high_hz / nyquist], pass_zero=False)


def fir_bandstop(
    numtaps: int,
    low_hz: float,
    high_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """Design a linear-phase FIR band-stop (Kaiser window, odd numtaps)."""
    _validate_rate(sample_rate)
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")
    if numtaps % 2 == 0:
        numtaps += 1
    nyquist = sample_rate / 2.0
    return firwin(numtaps, [low_hz / nyquist, high_hz / nyquist])


def iir_sos(
    order: int,
    cutoff_hz: float | tuple[float, float],
    sample_rate: float,
    btype: str = "lowpass",
    ftype: str = "butter",
    rp: float = 1.0,
    rs: float = 40.0,
) -> np.ndarray:
    """Design an IIR filter as second-order sections (numerically stable)."""
    _validate_rate(sample_rate)
    if order < 1:
        raise ValueError("order must be >= 1.")
    nyquist = sample_rate / 2.0

    if isinstance(cutoff_hz, (tuple, list)):
        wn = [c / nyquist for c in cutoff_hz]
        if not all(0 < w < 1 for w in wn):
            raise ValueError("All cutoffs must lie between 0 and Nyquist.")
    else:
        wn = _normalized(cutoff_hz, sample_rate)

    if btype not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(f"Unknown btype '{btype}'.")

    designers = {
        "butter": lambda: butter(order, wn, btype=btype, output="sos"),
        "cheby1": lambda: cheby1(order, rp, wn, btype=btype, output="sos"),
        "cheby2": lambda: cheby2(order, rs, wn, btype=btype, output="sos"),
        "ellip": lambda: ellip(order, rp, rs, wn, btype=btype, output="sos"),
    }
    if ftype not in designers:
        raise ValueError(f"Unknown ftype '{ftype}'.")
    return designers[ftype]()


def apply_fir(samples: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply an FIR filter (same length output, zero-delay compensation
    is NOT applied; use for causal filtering)."""
    samples = np.asarray(samples)
    taps = np.asarray(taps)
    if taps.size == 0:
        raise ValueError("taps cannot be empty.")
    pad = taps.size - 1
    padded = np.concatenate([samples, np.zeros(pad, dtype=samples.dtype)])
    filtered = np.convolve(padded, taps, mode="valid")
    return filtered


def apply_sos(
    samples: np.ndarray,
    sos: np.ndarray,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply second-order sections; complex input is supported directly."""
    samples = np.asarray(samples)
    sos = np.asarray(sos)
    if sos.size == 0:
        raise ValueError("sos cannot be empty.")
    if zero_phase:
        return sosfiltfilt(sos, samples)
    return sosfilt(sos, samples)


def resample(samples: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase rational resampling by ``up/down``."""
    up, down = int(up), int(down)
    if up <= 0 or down <= 0:
        raise ValueError("up and down must be positive.")
    frac = Fraction(up, down)
    return resample_poly(np.asarray(samples), frac.numerator, frac.denominator)


def resample_to_rate(
    samples: np.ndarray,
    source_rate: float,
    target_rate: float,
) -> tuple[np.ndarray, float]:
    """Resample to a target rate; returns (samples, achieved_rate)."""
    _validate_rate(source_rate)
    _validate_rate(target_rate)
    frac = Fraction(target_rate, source_rate).limit_denominator(10_000)
    out = resample(np.asarray(samples), frac.numerator, frac.denominator)
    achieved = source_rate * frac.numerator / frac.denominator
    return out, achieved


def rrc_taps(
    samples_per_symbol: int,
    rolloff: float = 0.35,
    span_symbols: int = 8,
) -> np.ndarray:
    """Root-raised-cosine taps (delegates to the validated V1/V2 kernel)."""
    from ..parameters.symbol_rate import rrc_filter

    return rrc_filter(samples_per_symbol, rolloff=rolloff, span_symbols=span_symbols)
