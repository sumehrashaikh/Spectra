from __future__ import annotations

import numpy as np

from .signal import Signal


def validate_samples(samples: np.ndarray) -> np.ndarray:
    """
    Validate and convert signal samples to a complex NumPy array.
    """
    samples = np.asarray(samples, dtype=np.complex128)

    if samples.size == 0:
        raise ValueError("Signal contains no samples.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Signal contains NaN or infinite values.")

    return samples


def remove_dc(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Remove the DC component from a signal.

    Returns:
        cleaned_samples
        dc_magnitude
    """
    samples = validate_samples(samples)

    dc = np.mean(samples)
    cleaned = samples - dc

    return cleaned, float(np.abs(dc))


def normalize_amplitude(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Normalize signal amplitude so that RMS becomes approximately 1.

    Returns:
        normalized_samples
        original_rms
    """
    samples = validate_samples(samples)

    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))

    if rms <= 0:
        return samples.copy(), 0.0

    normalized = samples / rms

    return normalized, rms


def estimate_noise_floor(samples: np.ndarray) -> float:
    """
    Estimate total noise power using a robust spectral median.

    The FFT bin powers are calculated using Parseval-compatible
    normalization. The median is used because strong signal bins
    should not dominate the estimate.

    For white Gaussian noise, the median of the exponential
    periodogram distribution is approximately ln(2) times its mean,
    so a correction factor is applied.

    Returns:
        Estimated total noise power.
    """
    samples = validate_samples(samples)

    n = len(samples)

    if n < 4:
        raise ValueError("At least 4 samples are required for noise estimation.")

    spectrum = np.fft.fft(samples)

    # Parseval-compatible bin power.
    bin_power = (np.abs(spectrum) ** 2) / (n ** 2)

    median_bin_power = float(np.median(bin_power))

    # Correction for the median of exponential-distributed
    # periodogram power.
    noise_power = (median_bin_power * n) / np.log(2)

    # Prevent numerical issues with completely clean signals.
    noise_power = max(noise_power, np.finfo(float).eps)

    return float(noise_power)


def estimate_snr(
    samples: np.ndarray,
    noise_power: float | None = None,
) -> float:
    """
    Estimate signal-to-noise ratio in dB.

    SNR = signal power / noise power

    Signal power is estimated as:

        total power - noise power
    """
    samples = validate_samples(samples)

    total_power = float(np.mean(np.abs(samples) ** 2))

    if noise_power is None:
        noise_power = estimate_noise_floor(samples)

    noise_power = float(noise_power)

    if noise_power <= 0:
        return float("inf")

    signal_power = max(total_power - noise_power, np.finfo(float).eps)

    snr_db = 10.0 * np.log10(signal_power / noise_power)

    return float(snr_db)


def preprocess_signal(
    signal: Signal,
    remove_dc_offset: bool = True,
    normalize: bool = True,
    estimate_noise: bool = True,
) -> Signal:
    """
    Run the complete preprocessing pipeline on a Signal.

    Steps:
        1. Validate samples
        2. Measure input statistics
        3. Remove DC offset
        4. Estimate noise power
        5. Estimate SNR
        6. Normalize amplitude
        7. Store processing information in metadata

    Returns:
        A new processed Signal object.
    """
    if not isinstance(signal, Signal):
        raise TypeError("preprocess_signal() requires a Signal object.")

    samples = validate_samples(signal.samples)

    input_samples = len(samples)
    input_rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))

    metadata = signal.metadata.copy()

    metadata["input_samples"] = input_samples
    metadata["input_rms"] = input_rms

    # ---------------------------------------------------------
    # Step 1: Remove DC
    # ---------------------------------------------------------
    dc_removed = False
    dc_magnitude = 0.0

    if remove_dc_offset:
        samples, dc_magnitude = remove_dc(samples)
        dc_removed = True

    metadata["dc_removed"] = dc_removed
    metadata["dc_magnitude"] = dc_magnitude

    # ---------------------------------------------------------
    # Step 2: Estimate noise and SNR
    # ---------------------------------------------------------
    noise_power = None
    snr_db = None

    if estimate_noise:
        noise_power = estimate_noise_floor(samples)
        snr_db = estimate_snr(samples, noise_power)

        metadata["noise_power"] = noise_power
        metadata["snr_db"] = snr_db

    # ---------------------------------------------------------
    # Step 3: Normalize amplitude
    # ---------------------------------------------------------
    normalized = False
    original_rms = float(
        np.sqrt(np.mean(np.abs(samples) ** 2))
    )

    if normalize:
        samples, original_rms = normalize_amplitude(samples)
        normalized = True

    metadata["normalized"] = normalized
    metadata["original_rms"] = original_rms

    # ---------------------------------------------------------
    # Step 4: Final statistics
    # ---------------------------------------------------------
    output_rms = float(
        np.sqrt(np.mean(np.abs(samples) ** 2))
    )

    output_peak = float(
        np.max(np.abs(samples))
    )

    metadata["output_rms"] = output_rms
    metadata["output_peak"] = output_peak

    return Signal(
        samples=samples,
        sample_rate=signal.sample_rate,
        metadata=metadata,
    )