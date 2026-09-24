import numpy as np

from prototype.core.signal import Signal
from prototype.core.synchronization import (
    synchronize_signal,
)
from prototype.parameters.symbol_rate import rrc_filter


def generate_qpsk_signal(
    sample_rate=8000,
    symbol_rate=100,
    num_symbols=1000,
    timing_offset=13,
    frequency_offset=37.0,
    phase_offset=np.deg2rad(30.0),
    seed=42,
):
    """
    Generate RRC-shaped QPSK with simultaneous:

        - timing offset
        - frequency offset
        - phase offset

    The transmitted symbols are returned for validation.
    """

    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        size=(num_symbols, 2),
    )

    constellation = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    indices = (
        bits[:, 0] * 2
        + bits[:, 1]
    )

    mapping = np.array(
        [
            0,
            1,
            3,
            2,
        ]
    )

    symbols = constellation[
        mapping[indices]
    ]

    samples_per_symbol = int(
        round(
            sample_rate
            / symbol_rate
        )
    )

    # ---------------------------------------------------------
    # Upsample
    # ---------------------------------------------------------

    upsampled = np.zeros(
        num_symbols
        * samples_per_symbol,
        dtype=np.complex128,
    )

    upsampled[
        ::samples_per_symbol
    ] = symbols

    # ---------------------------------------------------------
    # RRC pulse shaping
    # ---------------------------------------------------------

    taps = rrc_filter(
        samples_per_symbol,
        rolloff=0.35,
        span_symbols=8,
    )

    samples = np.convolve(
        upsampled,
        taps,
        mode="same",
    )

    # ---------------------------------------------------------
    # Timing offset
    # ---------------------------------------------------------

    samples = np.concatenate(
        [
            np.zeros(timing_offset),
            samples,
        ]
    )

    # ---------------------------------------------------------
    # Frequency offset
    # ---------------------------------------------------------

    time = (
        np.arange(samples.size)
        / sample_rate
    )

    samples = samples * np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_offset
        * time
    )

    # ---------------------------------------------------------
    # Phase offset
    # ---------------------------------------------------------

    samples = samples * np.exp(
        1j * phase_offset
    )

    return samples, symbols


def qpsk_decision(
    samples: np.ndarray,
) -> np.ndarray:
    """Map received samples to nearest QPSK point."""

    constellation = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    decisions = []

    for sample in samples:

        distances = np.abs(
            sample
            - constellation
        )

        decisions.append(
            np.argmin(distances)
        )

    return np.asarray(
        decisions
    )


def test_test_synchronization():

    print("=" * 70)
    print("V2 FULL SYNCHRONIZATION INTEGRATION TEST")
    print("=" * 70)

    sample_rate = 8000
    symbol_rate = 100

    known_timing_offset = 13
    known_frequency_offset = 37.0
    known_phase_offset = np.deg2rad(
        30.0
    )

    # ---------------------------------------------------------
    # Generate impaired signal
    # ---------------------------------------------------------

    samples, transmitted_symbols = (
        generate_qpsk_signal(
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            num_symbols=1000,
            timing_offset=known_timing_offset,
            frequency_offset=known_frequency_offset,
            phase_offset=known_phase_offset,
            seed=42,
        )
    )

    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": "full synchronization",

            "known_timing_offset":
                known_timing_offset,

            "known_frequency_offset":
                known_frequency_offset,

            "known_phase_offset":
                known_phase_offset,
        },
    )

    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------

    print("\n[1] Input")

    print(
        f"Sample rate       : "
        f"{signal.sample_rate:.2f} Hz"
    )

    print(
        f"Samples           : "
        f"{signal.num_samples}"
    )

    print(
        f"Symbols           : "
        f"{len(transmitted_symbols)}"
    )

    print(
        f"Symbol rate       : "
        f"{symbol_rate:.2f} baud"
    )

    print(
        f"Timing offset     : "
        f"{known_timing_offset} samples"
    )

    print(
        f"Frequency offset  : "
        f"{known_frequency_offset:.2f} Hz"
    )

    print(
        f"Phase offset      : "
        f"{np.rad2deg(known_phase_offset):.2f}°"
    )

    # ---------------------------------------------------------
    # Full synchronization
    # ---------------------------------------------------------

    print("\n[2] Full synchronization")

    result = synchronize_signal(
        signal,
        symbol_rate=symbol_rate,
        modulation_order=4,
    )

    print("\nCarrier recovery")

    print(
        f"  Frequency offset : "
        f"{result.frequency_offset:.4f} Hz"
    )

    print(
        f"  Frequency error  : "
        f"{abs(result.frequency_offset - known_frequency_offset):.4f} Hz"
    )

    print(
        f"  Frequency conf.  : "
        f"{result.frequency_confidence:.2f}%"
    )

    print(
        f"  Phase offset     : "
        f"{np.rad2deg(result.phase_offset):.4f}°"
    )

    print(
        f"  Phase confidence : "
        f"{result.phase_confidence:.2f}%"
    )

    print("\nTiming recovery")

    print(
        f"  Estimated offset : "
        f"{result.timing_offset}"
    )

    print(
        f"  Confidence       : "
        f"{result.timing_confidence:.2f}%"
    )

    print(
        f"  SPS              : "
        f"{result.samples_per_symbol:.2f}"
    )

    print(
        f"\n  Output symbols   : "
        f"{result.signal.num_samples}"
    )

    # ---------------------------------------------------------
    # Symbol validation
    # ---------------------------------------------------------

    print("\n[3] Symbol validation")

    recovered_symbols = (
        result.signal.samples
    )

    comparison_length = min(
        len(recovered_symbols),
        len(transmitted_symbols),
    )

    recovered_symbols = (
        recovered_symbols[
            :comparison_length
        ]
    )

    expected_symbols = (
        transmitted_symbols[
            :comparison_length
        ]
    )

    decisions = qpsk_decision(
        recovered_symbols
    )

    constellation = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    expected_indices = []

    for symbol in expected_symbols:

        distances = np.abs(
            symbol
            - constellation
        )

        expected_indices.append(
            np.argmin(distances)
        )

    expected_indices = np.asarray(
        expected_indices
    )

    symbol_errors = int(
        np.sum(
            decisions
            != expected_indices
        )
    )

    ser = (
        symbol_errors
        / comparison_length
    )

    print(
        f"Compared symbols : "
        f"{comparison_length}"
    )

    print(
        f"Symbol errors    : "
        f"{symbol_errors}"
    )

    print(
        f"SER              : "
        f"{ser:.6f}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n[4] Validation")

    frequency_error = abs(
        result.frequency_offset
        - known_frequency_offset
    )

    if frequency_error > 1.0:
        raise AssertionError(
            "Frequency offset estimation error "
            "is greater than 1 Hz."
        )

    if comparison_length < 900:
        raise AssertionError(
            "Too few symbols recovered."
        )

    if ser > 0.01:
        raise AssertionError(
            "Synchronization SER is too high."
        )

    print(
        "✓ Carrier synchronization passed."
    )

    print(
        "✓ Frequency correction passed."
    )

    print(
        "✓ Phase synchronization passed."
    )

    print(
        "✓ Timing synchronization executed."
    )

    print(
        "✓ End-to-end QPSK synchronization passed."
    )

    print(
        "✓ SER validation passed."
    )

    print("\n" + "=" * 70)
    print(
        "✓ FULL SYNCHRONIZATION TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
