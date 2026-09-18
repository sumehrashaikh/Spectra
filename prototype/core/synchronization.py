from dataclasses import dataclass

from prototype.core.signal import Signal
from prototype.core.synchronizer import (
    SynchronizationResult as TimingSynchronizationResult,
    synchronize_signal as synchronize_timing,
)
from prototype.core.carrier_sync import (
    CarrierRecoveryResult,
    recover_carrier,
)


@dataclass
class FullSynchronizationResult:
    """Complete timing + carrier synchronization result."""

    signal: Signal

    symbol_rate: float
    samples_per_symbol: float

    timing_offset: int
    timing_confidence: float

    frequency_offset: float
    frequency_confidence: float

    phase_offset: float
    phase_confidence: float


def synchronize_signal(
    signal: Signal,
    symbol_rate: float,
    modulation_order: int = 4,
) -> FullSynchronizationResult:
    """
    Perform the complete Spectra synchronization pipeline.

    Carrier recovery is performed BEFORE timing recovery.

    This is important because timing recovery reduces the
    sampling rate to the symbol rate. Performing carrier
    recovery after that reduction can cause large frequency
    offsets to alias.

    Pipeline:

        Input Signal
             ↓
        Carrier/Frequency Recovery
             ↓
        Phase Recovery
             ↓
        Timing Recovery
             ↓
        One sample / symbol
             ↓
        Synchronized Signal
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "synchronize_signal expects a Signal object."
        )

    if symbol_rate <= 0:
        raise ValueError(
            "symbol_rate must be positive."
        )

    if modulation_order < 2:
        raise ValueError(
            "modulation_order must be at least 2."
        )

    # ---------------------------------------------------------
    # Stage 1: Carrier and phase synchronization
    # ---------------------------------------------------------

    carrier_result: CarrierRecoveryResult = (
        recover_carrier(
            signal,
            modulation_order=modulation_order,
        )
    )

    carrier_signal = carrier_result.signal

    # ---------------------------------------------------------
    # Stage 2: Timing synchronization
    # ---------------------------------------------------------

    timing_result: TimingSynchronizationResult = (
        synchronize_timing(
            carrier_signal,
            symbol_rate=symbol_rate,
        )
    )

    synchronized_signal = timing_result.signal

    # ---------------------------------------------------------
    # Preserve complete synchronization metadata
    # ---------------------------------------------------------

    synchronized_signal.add_metadata(
        synchronization="carrier_and_timing",

        symbol_rate=float(
            symbol_rate
        ),

        samples_per_symbol=float(
            timing_result.samples_per_symbol
        ),

        timing_offset=int(
            timing_result.timing_offset
        ),

        timing_confidence=float(
            timing_result.timing_confidence
        ),

        frequency_offset=float(
            carrier_result.frequency_offset
        ),

        frequency_confidence=float(
            carrier_result.frequency_confidence
        ),

        phase_offset=float(
            carrier_result.phase_offset
        ),

        phase_confidence=float(
            carrier_result.phase_confidence
        ),
    )

    return FullSynchronizationResult(
        signal=synchronized_signal,

        symbol_rate=float(
            symbol_rate
        ),

        samples_per_symbol=float(
            timing_result.samples_per_symbol
        ),

        timing_offset=int(
            timing_result.timing_offset
        ),

        timing_confidence=float(
            timing_result.timing_confidence
        ),

        frequency_offset=float(
            carrier_result.frequency_offset
        ),

        frequency_confidence=float(
            carrier_result.frequency_confidence
        ),

        phase_offset=float(
            carrier_result.phase_offset
        ),

        phase_confidence=float(
            carrier_result.phase_confidence
        ),
    )