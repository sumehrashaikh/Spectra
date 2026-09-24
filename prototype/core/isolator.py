from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal import Signal


@dataclass
class IsolationResult:
    """
    Result of signal isolation and digital down conversion.
    """

    signal: Signal
    center_frequency: float
    bandwidth: float
    low_cutoff: float
    high_cutoff: float

    def summary(self) -> dict:
        return {
            "center_frequency": self.center_frequency,
            "bandwidth": self.bandwidth,
            "low_cutoff": self.low_cutoff,
            "high_cutoff": self.high_cutoff,
            "num_samples": self.signal.num_samples,
            "sample_rate": self.signal.sample_rate,
            "duration": self.signal.duration,
        }


def frequency_shift(
    signal: Signal,
    frequency_offset: float,
) -> Signal:
    """
    Shift a signal in frequency.

    Positive frequency_offset shifts the signal downward
    by that amount.

    Example:
        500 Hz signal + frequency_offset=500 Hz
        -> approximately 0 Hz.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "frequency_shift() requires a Signal object."
        )

    samples = signal.samples
    sample_rate = signal.sample_rate

    time = np.arange(
        signal.num_samples,
        dtype=np.float64,
    ) / sample_rate

    oscillator = np.exp(
        -1j * 2.0 * np.pi * frequency_offset * time
    )

    shifted_samples = samples * oscillator

    metadata = signal.metadata.copy()
    metadata["frequency_shift_hz"] = float(
        frequency_offset
    )

    return Signal(
        samples=shifted_samples,
        sample_rate=sample_rate,
        metadata=metadata,
    )


def lowpass_filter(
    signal: Signal,
    cutoff_hz: float,
    filter_order: int = 6,
) -> Signal:
    """
    Apply a Butterworth low-pass filter.

    The filter is applied using zero-phase filtering
    when SciPy is available.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "lowpass_filter() requires a Signal object."
        )

    if cutoff_hz <= 0:
        raise ValueError(
            "Cutoff frequency must be positive."
        )

    nyquist = signal.sample_rate / 2.0

    if cutoff_hz >= nyquist:
        raise ValueError(
            "Cutoff frequency must be below the Nyquist frequency."
        )

    if filter_order < 1:
        raise ValueError(
            "Filter order must be at least 1."
        )

    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for low-pass filtering."
        ) from exc

    normalized_cutoff = cutoff_hz / nyquist

    sos = butter(
        filter_order,
        normalized_cutoff,
        btype="lowpass",
        output="sos",
    )

    # Filter real and imaginary components separately.
    filtered_real = sosfiltfilt(
        sos,
        np.real(signal.samples),
    )

    filtered_imag = sosfiltfilt(
        sos,
        np.imag(signal.samples),
    )

    filtered_samples = (
        filtered_real
        + 1j * filtered_imag
    )

    metadata = signal.metadata.copy()
    metadata["lowpass_cutoff_hz"] = float(
        cutoff_hz
    )
    metadata["filter_order"] = int(
        filter_order
    )

    return Signal(
        samples=filtered_samples,
        sample_rate=signal.sample_rate,
        metadata=metadata,
    )


def isolate_signal(
    signal: Signal,
    center_frequency: float,
    bandwidth: float,
    filter_margin: float = 1.25,
    filter_order: int = 6,
) -> IsolationResult:
    """
    Isolate a detected signal and move it to baseband.

    Steps:

        Wideband signal
             ↓
        Frequency shift
             ↓
        Baseband signal
             ↓
        Low-pass filter
             ↓
        Isolated signal
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "isolate_signal() requires a Signal object."
        )

    if bandwidth < 0:
        raise ValueError(
            "Bandwidth cannot be negative."
        )

    if filter_margin <= 0:
        raise ValueError(
            "Filter margin must be positive."
        )

    nyquist = signal.sample_rate / 2.0

    # A zero bandwidth can occur with a single-bin tone.
    # Give the filter a small practical bandwidth in that case.
    effective_bandwidth = max(
        bandwidth,
        signal.sample_rate / signal.num_samples,
    )

    cutoff_hz = (
        effective_bandwidth
        / 2.0
        * filter_margin
    )

    # Keep the cutoff safely below Nyquist.
    cutoff_hz = min(
        cutoff_hz,
        nyquist * 0.95,
    )

    # 1. Move the detected carrier to DC.
    baseband = frequency_shift(
        signal,
        center_frequency,
    )

    # 2. Remove frequencies outside the isolated signal.
    isolated = lowpass_filter(
        baseband,
        cutoff_hz,
        filter_order=filter_order,
    )

    metadata = isolated.metadata.copy()

    metadata["isolated"] = True
    metadata["original_center_frequency_hz"] = float(
        center_frequency
    )
    metadata["original_bandwidth_hz"] = float(
        bandwidth
    )
    metadata["isolation_cutoff_hz"] = float(
        cutoff_hz
    )

    isolated = Signal(
        samples=isolated.samples,
        sample_rate=isolated.sample_rate,
        metadata=metadata,
    )

    return IsolationResult(
        signal=isolated,
        center_frequency=float(center_frequency),
        bandwidth=float(bandwidth),
        low_cutoff=-cutoff_hz,
        high_cutoff=cutoff_hz,
    )
