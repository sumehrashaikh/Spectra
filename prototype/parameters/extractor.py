from dataclasses import dataclass, asdict

import numpy as np

from prototype.core.signal import Signal


@dataclass
class SignalParameters:
    """Extracted parameters describing a detected signal."""

    center_frequency: float
    peak_frequency: float
    bandwidth: float

    rms: float
    peak: float
    power: float

    noise_power: float
    snr_db: float

    num_samples: int
    sample_rate: float
    duration: float


def compute_spectrum(
    samples: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a two-sided FFT spectrum.

    Returns:
        frequencies
        power_spectrum
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot compute spectrum of empty samples."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be positive."
        )

    spectrum = np.fft.fftshift(
        np.fft.fft(samples)
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(samples),
            d=1.0 / sample_rate,
        )
    )

    # Normalize FFT power by number of samples.
    power = (
        np.abs(spectrum) ** 2
        / len(samples) ** 2
    )

    return frequencies, power


def estimate_peak_frequency(
    samples: np.ndarray,
    sample_rate: float,
) -> float:
    """Estimate the strongest frequency component."""

    frequencies, power = compute_spectrum(
        samples,
        sample_rate,
    )

    index = int(
        np.argmax(power)
    )

    return float(
        frequencies[index]
    )


def estimate_center_frequency(
    samples: np.ndarray,
    sample_rate: float,
) -> float:
    """
    Estimate spectral center frequency using
    power-weighted frequency centroid.
    """

    frequencies, power = compute_spectrum(
        samples,
        sample_rate,
    )

    total_power = float(
        np.sum(power)
    )

    if total_power <= 0:
        return 0.0

    center = np.sum(
        frequencies * power
    ) / total_power

    return float(center)


def estimate_bandwidth(
    samples: np.ndarray,
    sample_rate: float,
    occupied_fraction: float = 0.99,
) -> float:
    """
    Estimate occupied bandwidth.

    The default uses the frequency interval containing
    99% of the total spectral power.
    """

    if not 0.0 < occupied_fraction < 1.0:
        raise ValueError(
            "occupied_fraction must be between 0 and 1."
        )

    frequencies, power = compute_spectrum(
        samples,
        sample_rate,
    )

    total_power = float(
        np.sum(power)
    )

    if total_power <= 0:
        return 0.0

    cumulative = np.cumsum(power)

    lower_power = (
        total_power
        * (1.0 - occupied_fraction)
        / 2.0
    )

    upper_power = (
        total_power
        * (
            1.0
            - (
                1.0
                - occupied_fraction
            )
            / 2.0
        )
    )

    lower_index = int(
        np.searchsorted(
            cumulative,
            lower_power,
        )
    )

    upper_index = int(
        np.searchsorted(
            cumulative,
            upper_power,
        )
    )

    lower_index = np.clip(
        lower_index,
        0,
        len(frequencies) - 1,
    )

    upper_index = np.clip(
        upper_index,
        0,
        len(frequencies) - 1,
    )

    bandwidth = (
        frequencies[upper_index]
        - frequencies[lower_index]
    )

    return float(
        max(0.0, bandwidth)
    )


def estimate_noise_power(
    samples: np.ndarray,
) -> float:
    """
    Estimate noise power from the spectral floor.

    The strongest spectral components are excluded so that
    a clean tone or communication carrier is not mistaken
    for noise.

    This provides a robust baseline estimate for parameter
    extraction. More advanced noise estimation can be added
    later for complex RF environments.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot estimate noise from empty samples."
        )

    spectrum = np.fft.fft(
        samples
    )

    power = (
        np.abs(spectrum) ** 2
        / samples.size ** 2
    )

    if power.size < 8:
        return float(
            max(
                np.mean(power),
                1e-12,
            )
        )

    # Sort spectral bins by power.
    sorted_power = np.sort(power)

    # Use the lower 50% of spectral bins as a robust
    # estimate of the spectral floor.
    floor_bins = sorted_power[
        : max(
            1,
            len(sorted_power) // 2,
        )
    ]

    noise_power = float(
        np.median(floor_bins)
        * len(power)
    )

    # Prevent numerical zero.
    return float(
        max(
            noise_power,
            1e-12,
        )
    )


def estimate_snr(
    samples: np.ndarray,
    noise_power: float | None = None,
) -> float:
    """Estimate signal-to-noise ratio in dB."""

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot estimate SNR from empty samples."
        )

    total_power = float(
        np.mean(
            np.abs(samples) ** 2
        )
    )

    if noise_power is None:
        noise_power = estimate_noise_power(
            samples
        )

    noise_power = max(
        float(noise_power),
        1e-12,
    )

    signal_power = max(
        total_power - noise_power,
        1e-12,
    )

    return float(
        10.0
        * np.log10(
            signal_power
            / noise_power
        )
    )


def extract_parameters(
    signal: Signal,
    noise_power: float | None = None,
) -> SignalParameters:
    """
    Extract measurable parameters from a Signal.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "extract_parameters expects a Signal object."
        )

    peak_frequency = estimate_peak_frequency(
        signal.samples,
        signal.sample_rate,
    )

    center_frequency = estimate_center_frequency(
        signal.samples,
        signal.sample_rate,
    )

    bandwidth = estimate_bandwidth(
        signal.samples,
        signal.sample_rate,
    )

    rms = signal.rms
    peak = signal.peak

    power = float(
        np.mean(
            np.abs(signal.samples) ** 2
        )
    )

    if noise_power is None:
        noise_power = estimate_noise_power(
            signal.samples
        )

    snr_db = estimate_snr(
        signal.samples,
        noise_power=noise_power,
    )

    return SignalParameters(
        center_frequency=center_frequency,
        peak_frequency=peak_frequency,
        bandwidth=bandwidth,
        rms=rms,
        peak=peak,
        power=power,
        noise_power=float(noise_power),
        snr_db=snr_db,
        num_samples=signal.num_samples,
        sample_rate=signal.sample_rate,
        duration=signal.duration,
    )


def parameters_to_dict(
    parameters: SignalParameters,
) -> dict:
    """Convert extracted parameters to a dictionary."""

    return asdict(parameters)