import numpy as np

from prototype.core.signal import Signal
from prototype.core.synchronizer import synchronize_signal
from prototype.parameters.symbol_rate import rrc_filter


def generate_test_bpsk(
    sample_rate=8000,
    symbol_rate=100,
    num_symbols=200,
    timing_offset=13,
    seed=42,
):
    """
    Generate pulse-shaped BPSK with an intentional
    sample timing offset.
    """

    rng = np.random.default_rng(seed)

    samples_per_symbol = int(
        round(
            sample_rate
            / symbol_rate
        )
    )

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
        rolloff=0.35,
        span_symbols=8,
    )

    shaped = np.convolve(
        upsampled,
        taps,
        mode="same",
    )

    # Apply a deliberate integer timing shift.
    shifted = np.concatenate(
        [
            np.zeros(timing_offset),
            shaped,
        ]
    )

    return (
        shifted.astype(np.complex128),
        bits,
    )


def test_test_timing_recovery():

    print("=" * 70)
    print("V2 TIMING RECOVERY TEST")
    print("=" * 70)

    sample_rate = 8000
    symbol_rate = 100
    known_offset = 13

    # ---------------------------------------------------------
    # Generate test signal
    # ---------------------------------------------------------

    samples, transmitted_bits = (
        generate_test_bpsk(
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            num_symbols=200,
            timing_offset=known_offset,
            seed=42,
        )
    )

    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": "BPSK timing recovery",
            "known_symbol_rate": symbol_rate,
            "known_timing_offset": known_offset,
        },
    )

    print("\n[1] Input")

    print(
        f"Sample rate      : "
        f"{signal.sample_rate:.2f} Hz"
    )

    print(
        f"Samples          : "
        f"{signal.num_samples}"
    )

    print(
        f"Known symbol rate: "
        f"{symbol_rate:.2f} baud"
    )

    print(
        f"Known SPS        : "
        f"{sample_rate / symbol_rate:.2f}"
    )

    print(
        f"Injected offset  : "
        f"{known_offset} samples"
    )

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------

    print("\n[2] Timing synchronization")

    result = synchronize_signal(
        signal,
        symbol_rate=symbol_rate,
    )

    print(
        f"Symbol rate      : "
        f"{result.symbol_rate:.2f} baud"
    )

    print(
        f"SPS              : "
        f"{result.samples_per_symbol:.2f}"
    )

    print(
        f"Estimated offset : "
        f"{result.timing_offset}"
    )

    print(
        f"Confidence       : "
        f"{result.timing_confidence:.2f}%"
    )

    print(
        f"Recovered symbols: "
        f"{result.signal.num_samples}"
    )

    # ---------------------------------------------------------
    # Symbol validation
    # ---------------------------------------------------------

    print("\n[3] Symbol validation")

    recovered = result.signal.samples

    recovered_bits = (
        np.real(recovered) >= 0
    ).astype(int)

    # Ignore edge symbols affected by RRC filter startup/end.
    comparison_length = min(
        len(recovered_bits),
        len(transmitted_bits),
    )

    recovered_bits = recovered_bits[
        :comparison_length
    ]

    expected_bits = transmitted_bits[
        :comparison_length
    ]

    bit_errors = int(
        np.sum(
            recovered_bits
            != expected_bits
        )
    )

    ber = (
        bit_errors
        / comparison_length
    )

    print(
        f"Compared symbols : "
        f"{comparison_length}"
    )

    print(
        f"Bit errors       : "
        f"{bit_errors}"
    )

    print(
        f"BER              : "
        f"{ber:.6f}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n[4] Validation")

    if abs(
        result.symbol_rate
        - symbol_rate
    ) > 1e-6:
        raise AssertionError(
            "Incorrect symbol rate."
        )

    if not (
        0
        <= result.timing_offset
        < round(
            result.samples_per_symbol
        )
    ):
        raise AssertionError(
            "Invalid timing offset."
        )

    if result.signal.num_samples < 100:
        raise AssertionError(
            "Too few symbols recovered."
        )

    if ber > 0.10:
        raise AssertionError(
            "Timing recovery BER is too high."
        )

    print(
        "✓ Symbol rate correctly supplied."
    )

    print(
        "✓ Valid timing offset recovered."
    )

    print(
        "✓ Symbol recovery passed."
    )

    print(
        "✓ BER validation passed."
    )

    print("\n" + "=" * 70)
    print(
        "✓ TIMING RECOVERY TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
