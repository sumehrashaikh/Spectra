from dataclasses import dataclass

import numpy as np

from .signal import Signal


@dataclass
class SynchronizationResult:
    """Result produced by timing synchronization."""

    signal: Signal
    symbol_rate: float
    samples_per_symbol: float
    timing_offset: int
    timing_confidence: float
    estimated_symbol_rate: float | None = None


def estimate_timing_offset(
    samples: np.ndarray,
    samples_per_symbol: float,
) -> tuple[int, float]:
    """
    Estimate the best integer sampling phase.

    The method evaluates each possible sampling phase and
    measures how strongly the samples cluster around stable
    symbol values.

    This function assumes that the symbol rate has already
    been estimated by the parameter-extraction stage.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot estimate timing from empty samples."
        )

    if samples_per_symbol <= 0:
        raise ValueError(
            "samples_per_symbol must be positive."
        )

    sps = max(
        2,
        int(round(samples_per_symbol)),
    )

    scores = np.zeros(sps)

    for offset in range(sps):

        phase_samples = samples[
            offset::sps
        ]

        if phase_samples.size < 4:
            continue

        # A good sampling point should produce stable
        # symbol amplitudes. Use the inverse of normalized
        # within-symbol variation as the timing metric.
        magnitudes = np.abs(
            phase_samples
        )

        mean_magnitude = float(
            np.mean(magnitudes)
        )

        if mean_magnitude <= 1e-12:
            continue

        variation = float(
            np.std(magnitudes)
            / mean_magnitude
        )

        scores[offset] = (
            1.0
            / (1.0 + variation)
        )

    best_offset = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[best_offset]
    )

    if best_score <= 0:
        confidence = 0.0
    else:
        sorted_scores = np.sort(
            scores
        )

        second_best = float(
            sorted_scores[-2]
        ) if len(sorted_scores) > 1 else 0.0

        if second_best <= 0:
            confidence = 100.0
        else:
            confidence = min(
                100.0,
                max(
                    0.0,
                    100.0
                    * (
                        best_score
                        - second_best
                    )
                    / best_score,
                ),
            )

    return (
        best_offset,
        confidence,
    )


def recover_symbols(
    samples: np.ndarray,
    samples_per_symbol: float,
    timing_offset: int,
) -> np.ndarray:
    """
    Recover one sample per symbol using the estimated
    timing phase.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot recover symbols from empty samples."
        )

    sps = max(
        2,
        int(round(samples_per_symbol)),
    )

    if not 0 <= timing_offset < sps:
        raise ValueError(
            "Timing offset must be within one symbol period."
        )

    symbols = samples[
        timing_offset::sps
    ]

    if symbols.size == 0:
        raise RuntimeError(
            "No symbols were recovered."
        )

    return symbols


def synchronize_signal(
    signal: Signal,
    symbol_rate: float | None = None,
    min_symbol_rate: float = 20.0,
    max_symbol_rate: float | None = None,
) -> SynchronizationResult:
    """
    Perform timing synchronization using a known/estimated
    symbol rate.

    A known/estimated ``symbol_rate`` (normally produced by
    the parameter-extraction stage) may be supplied directly.
    If it is omitted, the symbol rate is estimated from the
    cyclostationary structure of the signal.

    Pipeline:

        Signal
          ↓
        Known symbol rate (or cyclostationary estimate)
          ↓
        Samples-per-symbol
          ↓
        Timing-offset estimation
          ↓
        Symbol recovery
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "synchronize_signal expects a Signal object."
        )

    estimated_symbol_rate: float | None = None

    if symbol_rate is None:

        from ..parameters.symbol_rate import (
            estimate_symbol_rate,
        )

        if max_symbol_rate is None:
            max_symbol_rate = signal.sample_rate / 4.0

        rate_estimate = estimate_symbol_rate(
            signal.samples,
            signal.sample_rate,
            min_symbol_rate=min_symbol_rate,
            max_symbol_rate=max_symbol_rate,
        )

        estimated_symbol_rate = float(
            rate_estimate.symbol_rate
        )

        if estimated_symbol_rate <= 0:
            raise ValueError(
                "Could not estimate a symbol rate from the "
                "signal. Supply symbol_rate explicitly."
            )

        symbol_rate = estimated_symbol_rate

    if symbol_rate <= 0:
        raise ValueError(
            "symbol_rate must be positive."
        )

    samples_per_symbol = (
        signal.sample_rate
        / symbol_rate
    )

    (
        timing_offset,
        timing_confidence,
    ) = estimate_timing_offset(
        signal.samples,
        samples_per_symbol,
    )

    recovered_symbols = recover_symbols(
        signal.samples,
        samples_per_symbol,
        timing_offset,
    )

    synchronized_signal = Signal(
        samples=recovered_symbols,
        sample_rate=symbol_rate,
        metadata=signal.metadata.copy(),
    )

    synchronized_signal.add_metadata(
        synchronization="timing",
        symbol_rate=float(symbol_rate),
        samples_per_symbol=float(
            samples_per_symbol
        ),
        timing_offset=timing_offset,
        timing_confidence=timing_confidence,
        input_samples=signal.num_samples,
        output_symbols=synchronized_signal.num_samples,
    )

    return SynchronizationResult(
        signal=synchronized_signal,
        symbol_rate=float(symbol_rate),
        samples_per_symbol=float(
            samples_per_symbol
        ),
        timing_offset=timing_offset,
        timing_confidence=float(
            timing_confidence
        ),
        estimated_symbol_rate=estimated_symbol_rate,
    )