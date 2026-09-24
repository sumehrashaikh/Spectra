"""Baseband operations: analytic signal, envelope, instantaneous frequency,
mixing, DC estimation, blind IQ imbalance correction, clipping detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import hilbert
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for the dsp.baseband module.") from exc


def analytic_signal(real_samples: np.ndarray) -> np.ndarray:
    """Convert a real waveform to its analytic (complex) signal."""
    real = np.asarray(real_samples, dtype=np.float64)
    if real.ndim != 1:
        raise ValueError("analytic_signal expects a 1-D real array.")
    return hilbert(real).astype(np.complex128)


def envelope(samples: np.ndarray) -> np.ndarray:
    """Signal magnitude envelope |x[n]|."""
    return np.abs(np.asarray(samples))


def instantaneous_phase(samples: np.ndarray) -> np.ndarray:
    """Unwrapped instantaneous phase in radians."""
    samples = np.asarray(samples)
    if samples.size < 2:
        raise ValueError("At least 2 samples are required.")
    return np.unwrap(np.angle(samples))


def instantaneous_frequency(
    samples: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Instantaneous frequency in Hz (length N-1)."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    phase = instantaneous_phase(samples)
    return np.diff(phase) * sample_rate / (2.0 * np.pi)


def mix(
    samples: np.ndarray,
    frequency_hz: float,
    sample_rate: float,
    downconvert: bool = True,
) -> np.ndarray:
    """
    Mix samples with a complex exponential at ``frequency_hz``.

    ``downconvert=True`` multiplies by exp(-j*2*pi*f*t) (shifts the
    spectrum down by ``frequency_hz``); False shifts up.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    samples = np.asarray(samples)
    n = np.arange(samples.size, dtype=np.float64)
    exponent = -1.0 if downconvert else 1.0
    oscillator = np.exp(exponent * 1j * 2.0 * np.pi * frequency_hz * n / sample_rate)
    return samples * oscillator


def estimate_dc(samples: np.ndarray) -> complex:
    """Estimate the DC offset (complex mean) of a signal."""
    return complex(np.mean(np.asarray(samples)))


@dataclass(frozen=True)
class IQImbalance:
    """Estimated IQ imbalance parameters."""

    gain_imbalance_db: float
    phase_imbalance_deg: float
    dc_offset: complex


def correct_iq_imbalance(samples: np.ndarray) -> tuple[np.ndarray, IQImbalance]:
    """
    Blind IQ gain/phase imbalance correction via covariance whitening.

    Estimates the 2x2 covariance of (I, Q), removes the correlation
    (phase imbalance) and equalizes the variances (gain imbalance).
    Returns the corrected samples and the measured imbalance.
    """
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.size < 4:
        raise ValueError("At least 4 samples are required for IQ correction.")

    i = np.real(samples)
    q = np.imag(samples)
    dc = complex(np.mean(i), np.mean(q))

    i_zero = i - np.mean(i)
    q_zero = q - np.mean(q)

    cov = np.cov(np.vstack([i_zero, q_zero]))
    if not np.all(np.isfinite(cov)):
        raise ValueError("IQ covariance is degenerate; cannot correct.")

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if evals[-1] <= 1e-30:
        raise ValueError("IQ covariance is singular; cannot correct.")

    whitening = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    whitening = whitening / np.sqrt(np.linalg.det(whitening))

    corrected = np.empty(samples.size, dtype=np.complex128)
    stacked = whitening @ np.vstack([i_zero, q_zero])
    corrected.real = stacked[0]
    corrected.imag = stacked[1]

    gain_db = 10.0 * np.log10(evals[0] / max(evals[1], 1e-30))
    angle = np.degrees(
        np.arcsin(np.clip(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), -1, 1))
    )

    imbalance = IQImbalance(
        gain_imbalance_db=float(gain_db),
        phase_imbalance_deg=float(angle),
        dc_offset=dc,
    )
    return corrected, imbalance


def detect_clipping(
    samples: np.ndarray,
    level: float = 1.0,
) -> dict[str, float | int]:
    """
    Detect amplitude clipping/saturation.

    Returns a report with the count and fraction of samples at or
    beyond ``level`` and the peak amplitude.
    """
    magnitude = np.abs(np.asarray(samples))
    clipped = int(np.count_nonzero(magnitude >= level))
    return {
        "clipped_samples": clipped,
        "clipped_fraction": clipped / magnitude.size if magnitude.size else 0.0,
        "peak_amplitude": float(np.max(magnitude)) if magnitude.size else 0.0,
        "level": float(level),
    }
