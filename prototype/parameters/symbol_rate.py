from dataclasses import dataclass

import numpy as np


@dataclass
class SymbolRateEstimate:
    """Result of symbol-rate estimation."""

    symbol_rate: float
    samples_per_symbol: float
    confidence: float
    method: str


def rrc_filter(
    samples_per_symbol: int,
    rolloff: float = 0.35,
    span_symbols: int = 8,
) -> np.ndarray:
    """
    Create a Root Raised Cosine (RRC) filter.

    Args:
        samples_per_symbol: Samples used to represent each symbol.
        rolloff: RRC rolloff factor, 0 < rolloff <= 1.
        span_symbols: Filter length in symbols.

    Returns:
        Normalized RRC filter coefficients.
    """

    if samples_per_symbol <= 0:
        raise ValueError(
            "samples_per_symbol must be positive."
        )

    if not 0 < rolloff <= 1:
        raise ValueError(
            "rolloff must be between 0 and 1."
        )

    if span_symbols <= 0:
        raise ValueError(
            "span_symbols must be positive."
        )

    num_taps = (
        span_symbols
        * samples_per_symbol
        * 2
        + 1
    )

    time = np.arange(
        -span_symbols * samples_per_symbol,
        span_symbols * samples_per_symbol + 1,
        dtype=float,
    ) / samples_per_symbol

    taps = np.zeros_like(time)

    for i, t in enumerate(time):

        if abs(t) < 1e-12:
            taps[i] = (
                1
                + rolloff
                * (
                    4 / np.pi - 1
                )
            )

        elif abs(
            abs(t) - 1 / (4 * rolloff)
        ) < 1e-12:

            taps[i] = (
                rolloff
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(
                        np.pi
                        / (4 * rolloff)
                    )
                    + (
                        1 - 2 / np.pi
                    )
                    * np.cos(
                        np.pi
                        / (4 * rolloff)
                    )
                )
            )

        else:
            numerator = (
                np.sin(
                    np.pi
                    * t
                    * (1 - rolloff)
                )
                + 4
                * rolloff
                * t
                * np.cos(
                    np.pi
                    * t
                    * (1 + rolloff)
                )
            )

            denominator = (
                np.pi
                * t
                * (
                    1
                    - (
                        4
                        * rolloff
                        * t
                    ) ** 2
                )
            )

            taps[i] = (
                numerator
                / denominator
            )

    energy = np.sqrt(
        np.sum(taps ** 2)
    )

    if energy <= 0:
        raise RuntimeError(
            "Failed to create RRC filter."
        )

    taps /= energy

    return taps


def generate_rrc_bpsk(
    sample_rate: float,
    symbol_rate: float,
    duration: float,
    seed: int = 42,
    rolloff: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pulse-shaped BPSK.

    Returns:
        samples
        transmitted_bits
    """

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be positive."
        )

    if symbol_rate <= 0:
        raise ValueError(
            "symbol_rate must be positive."
        )

    if duration <= 0:
        raise ValueError(
            "duration must be positive."
        )

    samples_per_symbol_float = (
        sample_rate / symbol_rate
    )

    samples_per_symbol = int(
        round(samples_per_symbol_float)
    )

    if abs(
        samples_per_symbol
        - samples_per_symbol_float
    ) > 1e-6:

        raise ValueError(
            "This test generator requires "
            "an integer samples-per-symbol value."
        )

    num_symbols = int(
        duration * symbol_rate
    )

    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        num_symbols,
    )

    symbols = (
        2 * bits - 1
    ).astype(float)

    upsampled = np.zeros(
        num_symbols
        * samples_per_symbol
    )

    upsampled[
        ::samples_per_symbol
    ] = symbols

    taps = rrc_filter(
        samples_per_symbol,
        rolloff=rolloff,
    )

    shaped = np.convolve(
        upsampled,
        taps,
        mode="same",
    )

    return (
        shaped.astype(np.complex128),
        bits,
    )


def estimate_symbol_rate_from_cyclostationarity(
    samples: np.ndarray,
    sample_rate: float,
    min_symbol_rate: float,
    max_symbol_rate: float,
) -> SymbolRateEstimate:
    """
    Estimate symbol rate using second-order cyclostationarity.

    The squared magnitude of a pulse-shaped communication
    signal contains periodic information related to the
    symbol clock.

    Candidate rates are evaluated directly rather than
    selecting an arbitrary FFT harmonic.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size < 128:
        raise ValueError(
            "Not enough samples for symbol-rate estimation."
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be positive."
        )

    if min_symbol_rate <= 0:
        raise ValueError(
            "min_symbol_rate must be positive."
        )

    if max_symbol_rate <= min_symbol_rate:
        raise ValueError(
            "max_symbol_rate must be greater than "
            "min_symbol_rate."
        )

    # Remove DC.
    x = (
        samples
        - np.mean(samples)
    )

    # Squared magnitude removes carrier phase and
    # emphasizes symbol-rate periodicity.
    timing_signal = (
        np.abs(x) ** 2
    )

    timing_signal -= np.mean(
        timing_signal
    )

    # Use FFT to find candidate cyclic frequencies.
    n = len(timing_signal)

    spectrum = np.fft.rfft(
        timing_signal
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate,
    )

    power = (
        np.abs(spectrum) ** 2
    )

    valid = (
        (frequencies >= min_symbol_rate)
        & (
            frequencies
            <= max_symbol_rate
        )
    )

    if not np.any(valid):
        raise RuntimeError(
            "No valid symbol-rate candidates found."
        )

    valid_indices = np.flatnonzero(
        valid
    )

    # Find local maxima.
    local_maxima = []

    for index in valid_indices:

        if index == 0:
            continue

        if index >= len(power) - 1:
            continue

        if (
            power[index]
            >= power[index - 1]
            and power[index]
            >= power[index + 1]
        ):
            local_maxima.append(
                index
            )

    if not local_maxima:
        local_maxima = [
            valid_indices[
                np.argmax(
                    power[
                        valid_indices
                    ]
                )
            ]
        ]

    # Sort by spectral strength.
    local_maxima.sort(
        key=lambda index: power[index],
        reverse=True,
    )

    strongest_index = local_maxima[0]

    strongest_frequency = float(
        frequencies[strongest_index]
    )

    strongest_power = float(
        power[strongest_index]
    )

    if strongest_power <= 0:
        return SymbolRateEstimate(
            symbol_rate=0.0,
            samples_per_symbol=0.0,
            confidence=0.0,
            method="cyclostationarity",
        )

    # ---------------------------------------------------------
    # Harmonic-aware selection
    # ---------------------------------------------------------
    #
    # Communication signals can produce harmonics of the
    # symbol rate. We therefore inspect subharmonics of the
    # strongest candidate.
    # ---------------------------------------------------------

    selected_frequency = (
        strongest_frequency
    )

    selected_power = (
        strongest_power
    )

    for harmonic in range(2, 9):

        candidate_frequency = (
            strongest_frequency
            / harmonic
        )

        if (
            candidate_frequency
            < min_symbol_rate
        ):
            continue

        if (
            candidate_frequency
            > max_symbol_rate
        ):
            continue

        index = int(
            np.argmin(
                np.abs(
                    frequencies
                    - candidate_frequency
                )
            )
        )

        candidate_power = float(
            power[index]
        )

        # If the subharmonic retains significant energy,
        # prefer it as the fundamental symbol rate.
        if (
            candidate_power
            >= selected_power * 0.10
        ):
            selected_frequency = (
                float(frequencies[index])
            )

            selected_power = (
                candidate_power
            )

    # Confidence is based on the selected peak relative
    # to the median spectral floor.
    floor = float(
        np.median(
            power[
                valid_indices
            ]
        )
    )

    if floor <= 0:
        confidence = 100.0
    else:
        ratio = (
            selected_power
            / floor
        )

        confidence = min(
            100.0,
            max(
                0.0,
                100.0
                * (
                    1
                    - 1 / max(
                        ratio,
                        1.0,
                    )
                ),
            ),
        )

    samples_per_symbol = (
        sample_rate
        / selected_frequency
    )

    return SymbolRateEstimate(
        symbol_rate=float(
            selected_frequency
        ),
        samples_per_symbol=float(
            samples_per_symbol
        ),
        confidence=float(
            confidence
        ),
        method="cyclostationarity",
    )


def estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    min_symbol_rate: float = 20.0,
    max_symbol_rate: float | None = None,
) -> SymbolRateEstimate:
    """
    Public symbol-rate estimation API.
    """

    if max_symbol_rate is None:
        max_symbol_rate = sample_rate / 4.0

    return (
        estimate_symbol_rate_from_cyclostationarity(
            samples,
            sample_rate,
            min_symbol_rate,
            max_symbol_rate,
        )
    )