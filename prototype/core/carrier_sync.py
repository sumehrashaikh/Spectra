from dataclasses import dataclass

import numpy as np

from prototype.core.signal import Signal


@dataclass
class CarrierRecoveryResult:
    """Result produced by carrier/frequency recovery."""

    signal: Signal
    frequency_offset: float
    phase_offset: float
    frequency_confidence: float
    phase_confidence: float


def _normalize_phase(
    phase: float,
    period: float,
) -> float:
    """Normalize phase to [-period/2, period/2)."""

    return float(
        (
            phase
            + period / 2.0
        )
        % period
        - period / 2.0
    )


def estimate_frequency_offset(
    samples: np.ndarray,
    sample_rate: float,
    modulation_order: int = 4,
) -> tuple[float, float]:
    """
    Estimate carrier frequency offset using the M-th power
    spectral method.

    For QPSK, modulation_order = 4.

    Raising QPSK samples to the fourth power removes the
    four-fold modulation pattern and leaves a tone at
    four times the carrier frequency offset.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot estimate frequency offset from empty samples."
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be positive."
        )

    if modulation_order < 2:
        raise ValueError(
            "modulation_order must be at least 2."
        )

    # Remove DC before frequency estimation.
    samples = samples - np.mean(samples)

    powered = samples ** modulation_order

    n = powered.size

    window = np.hanning(n)

    spectrum = np.fft.fftshift(
        np.fft.fft(
            powered * window
        )
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1.0 / sample_rate,
        )
    )

    power = np.abs(spectrum) ** 2

    peak_power = float(
        np.max(power)
    )

    if peak_power <= 1e-15:
        return 0.0, 0.0

    peak_index = int(
        np.argmax(power)
    )

    measured_frequency = float(
        frequencies[peak_index]
    )

    # M-th power multiplies the frequency offset by M.
    frequency_offset = (
        measured_frequency
        / modulation_order
    )

    noise_floor = float(
        np.median(power)
    )

    if noise_floor <= 1e-15:
        confidence = 100.0
    else:
        ratio = (
            peak_power
            / noise_floor
        )

        confidence = min(
            100.0,
            max(
                0.0,
                100.0
                * ratio
                / (1.0 + ratio),
            ),
        )

    return (
        float(frequency_offset),
        float(confidence),
    )


def correct_frequency_offset(
    samples: np.ndarray,
    sample_rate: float,
    frequency_offset: float,
) -> np.ndarray:
    """
    Remove a known frequency offset from complex samples.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot correct empty samples."
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be positive."
        )

    time = (
        np.arange(samples.size)
        / sample_rate
    )

    correction = np.exp(
        -1j
        * 2.0
        * np.pi
        * frequency_offset
        * time
    )

    return samples * correction


def estimate_phase_offset(
    samples: np.ndarray,
    modulation_order: int = 4,
) -> tuple[float, float]:
    """
    Estimate residual carrier phase using the M-th power
    method.

    For the QPSK constellation used by Spectra:

        45°, 135°, 225°, 315°

    raising the constellation to the fourth power produces
    a phase of 180°.

    Therefore the intrinsic constellation phase must be
    removed before dividing by four.

    Example:

        transmitted phase offset = +30°

        4 * 30° + 180°
        = 300°
        = -60°

        (-60° - 180°) / 4
        = -60° modulo 90°
        = +30° equivalent

    The returned result is normalized to the QPSK ambiguity
    interval of ±45°.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot estimate phase from empty samples."
        )

    if modulation_order < 2:
        raise ValueError(
            "modulation_order must be at least 2."
        )

    powered = samples ** modulation_order

    mean_value = np.mean(powered)

    if abs(mean_value) <= 1e-12:
        return 0.0, 0.0

    measured_phase = float(
        np.angle(mean_value)
    )

    # The reference QPSK constellation used by the
    # Spectra test is centered at 45°.
    #
    # 4 × 45° = 180°.
    #
    # This is the intrinsic phase contributed by the
    # ideal QPSK constellation after fourth-power
    # transformation.
    reference_symbol_phase = np.pi / 4.0

    intrinsic_phase = (
        modulation_order
        * reference_symbol_phase
    )

    phase_difference = (
        measured_phase
        - intrinsic_phase
    )

    ambiguity_period = (
        2.0
        * np.pi
        / modulation_order
    )

    phase_offset = _normalize_phase(
        phase_difference
        / modulation_order,
        ambiguity_period,
    )

    # Confidence is based on how strongly the M-th power
    # samples cluster around their mean.
    magnitude = float(
        abs(mean_value)
    )

    spread = float(
        np.mean(
            np.abs(
                powered
                - mean_value
            ) ** 2
        )
    )

    if spread <= 1e-15:
        confidence = 100.0
    else:
        ratio = (
            magnitude ** 2
            / spread
        )

        confidence = min(
            100.0,
            max(
                0.0,
                100.0
                * ratio
                / (1.0 + ratio),
            ),
        )

    return (
        float(phase_offset),
        float(confidence),
    )


def correct_phase_offset(
    samples: np.ndarray,
    phase_offset: float,
) -> np.ndarray:
    """
    Remove a known phase offset.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    return samples * np.exp(
        -1j * phase_offset
    )


def recover_carrier(
    signal: Signal,
    modulation_order: int = 4,
) -> CarrierRecoveryResult:
    """
    Perform frequency and phase recovery.

    Current implementation is a QPSK/PSK baseline using
    the M-th power method.

    Pipeline:

        Input signal
             ↓
        Frequency-offset estimation
             ↓
        Frequency correction
             ↓
        Phase estimation
             ↓
        Phase correction
             ↓
        Recovered carrier
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "recover_carrier expects a Signal object."
        )

    if modulation_order < 2:
        raise ValueError(
            "modulation_order must be at least 2."
        )

    # ---------------------------------------------------------
    # Frequency recovery
    # ---------------------------------------------------------

    frequency_offset, frequency_confidence = (
        estimate_frequency_offset(
            signal.samples,
            signal.sample_rate,
            modulation_order,
        )
    )

    frequency_corrected = (
        correct_frequency_offset(
            signal.samples,
            signal.sample_rate,
            frequency_offset,
        )
    )

    # ---------------------------------------------------------
    # Phase recovery
    # ---------------------------------------------------------

    phase_offset, phase_confidence = (
        estimate_phase_offset(
            frequency_corrected,
            modulation_order,
        )
    )

    recovered = correct_phase_offset(
        frequency_corrected,
        phase_offset,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    synchronized_signal = Signal(
        samples=recovered,
        sample_rate=signal.sample_rate,
        metadata=signal.metadata.copy(),
    )

    synchronized_signal.add_metadata(
        synchronization="carrier",
        modulation_order=modulation_order,
        frequency_offset=float(
            frequency_offset
        ),
        phase_offset=float(
            phase_offset
        ),
        frequency_confidence=float(
            frequency_confidence
        ),
        phase_confidence=float(
            phase_confidence
        ),
    )

    return CarrierRecoveryResult(
        signal=synchronized_signal,
        frequency_offset=float(
            frequency_offset
        ),
        phase_offset=float(
            phase_offset
        ),
        frequency_confidence=float(
            frequency_confidence
        ),
        phase_confidence=float(
            phase_confidence
        ),
    )