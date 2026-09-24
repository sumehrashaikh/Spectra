"""Spectral estimators: Welch PSD, spectrogram/STFT with inverse, windows."""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import istft, stft, welch
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for the dsp.spectral module.") from exc


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= n (minimum 1)."""
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def get_window(window: str | tuple, n: int) -> np.ndarray:
    """Return a window of ``n`` points (scipy.get_window passthrough)."""
    from scipy.signal.windows import get_window as _get_window

    return _get_window(window, n)


def welch_psd(
    samples: np.ndarray,
    sample_rate: float,
    nperseg: int = 4096,
    noverlap: int | None = None,
    window: str | tuple = "hann",
    detrend: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the power spectral density.

    For complex input a two-sided, DC-centered (fftshifted) PSD is
    returned; for real input the standard one-sided PSD.

    Returns
    -------
    frequencies : ndarray
        Frequency axis in Hz (centered for complex input).
    psd : ndarray
        Power spectral density in units of power per Hz.
    """
    samples = np.asarray(samples)
    if samples.size < 2:
        raise ValueError("At least 2 samples are required for a PSD.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if nperseg > samples.size:
        nperseg = next_power_of_two(samples.size)
    if noverlap is None:
        noverlap = nperseg // 2

    is_complex = np.iscomplexobj(samples)
    freqs, psd = welch(
        samples,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=min(noverlap, nperseg - 1),
        window=window,
        detrend=detrend,
        return_onesided=not is_complex,
        scaling="density",
    )

    if is_complex:
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)

    return freqs, psd


def psd_db(psd: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Convert a PSD to dB (10*log10), clamped at ``floor``."""
    return 10.0 * np.log10(np.maximum(np.asarray(psd, dtype=float), floor))


def spectrogram_stft(
    samples: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str | tuple = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a spectrogram (magnitude-squared STFT).

    Complex input produces a two-sided, DC-centered spectrogram.

    Returns
    -------
    frequencies : ndarray, shape (F,)
    times : ndarray, shape (T,)
    sxx : ndarray, shape (F, T), power per bin
    """
    samples = np.asarray(samples)
    if samples.size < 2:
        raise ValueError("At least 2 samples are required for a spectrogram.")
    if nperseg > samples.size:
        nperseg = max(2, samples.size // 2)
    if noverlap is None:
        noverlap = nperseg // 2

    is_complex = np.iscomplexobj(samples)
    freqs, times, sxx = stft(
        samples,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=min(noverlap, nperseg - 1),
        window=window,
        return_onesided=not is_complex,
    )

    if is_complex:
        freqs = np.fft.fftshift(freqs)
        sxx = np.fft.fftshift(sxx, axes=0)

    return freqs, times, np.abs(sxx) ** 2


def inverse_stft(
    stft_matrix: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str | tuple = "hann",
    input_two_sided: bool = True,
) -> np.ndarray:
    """
    Reconstruct time-domain samples from a complex STFT produced by
    :func:`spectrogram_stft`-style analysis (pass the complex STFT,
    not |STFT|^2). ``input_two_sided`` selects two-sided (complex)
    input handling; scipy's istft performs the actual overlap-add.
    """
    matrix = np.asarray(stft_matrix)
    _, reconstructed = istft(
        matrix,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=(nperseg // 2 if noverlap is None else noverlap),
        window=window,
        input_onesided=not input_two_sided,
    )
    return reconstructed
