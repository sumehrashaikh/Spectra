import numpy as np

from prototype.core.signal import Signal
from prototype.core.carrier_sync import recover_carrier


def generate_qpsk(
    sample_rate=8000,
    symbol_rate=100,
    num_symbols=1000,
    frequency_offset=37.0,
    phase_offset=np.deg2rad(30.0),
    seed=42,
):
    """
    Generate QPSK symbols and deliberately apply
    frequency and phase offsets.
    """

    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        size=(num_symbols, 2),
    )

    # Gray-coded QPSK:
    #
    # 00 -> +1 + 1j
    # 01 -> -1 + 1j
    # 11 -> -1 - 1j
    # 10 -> +1 - 1j

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

    # Map the binary indices into a QPSK
    # constellation while keeping a deterministic
    # known transmitted sequence.
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

    samples = np.repeat(
        symbols,
        samples_per_symbol,
    )

    time = (
        np.arange(samples.size)
        / sample_rate
    )

    # Apply frequency offset.
    samples = samples * np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_offset
        * time
    )

    # Apply phase rotation.
    samples = samples * np.exp(
        1j * phase_offset
    )

    return samples, symbols


def qpsk_decision(samples):
    """
    Convert QPSK constellation points into
    nearest ideal constellation points.
    """

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


def test_test_carrier_recovery():

    print("=" * 70)
    print("V2 CARRIER & PHASE RECOVERY TEST")
    print("=" * 70)

    sample_rate = 8000
    symbol_rate = 100

    known_frequency_offset = 37.0
    known_phase_offset = np.deg2rad(
        30.0
    )

    samples, transmitted_symbols = (
        generate_qpsk(
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            num_symbols=1000,
            frequency_offset=known_frequency_offset,
            phase_offset=known_phase_offset,
            seed=42,
        )
    )

    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": "QPSK carrier recovery",
            "known_frequency_offset":
                known_frequency_offset,
            "known_phase_offset":
                known_phase_offset,
        },
    )

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
        f"Frequency offset  : "
        f"{known_frequency_offset:.2f} Hz"
    )

    print(
        f"Phase offset      : "
        f"{np.rad2deg(known_phase_offset):.2f}°"
    )

    print("\n[2] Carrier recovery")

    result = recover_carrier(
        signal,
        modulation_order=4,
    )

    print(
        f"Estimated frequency offset : "
        f"{result.frequency_offset:.4f} Hz"
    )

    print(
        f"Frequency error            : "
        f"{abs(result.frequency_offset - known_frequency_offset):.4f} Hz"
    )

    print(
        f"Frequency confidence       : "
        f"{result.frequency_confidence:.2f}%"
    )

    print(
        f"Estimated phase offset     : "
        f"{np.rad2deg(result.phase_offset):.4f}°"
    )

    print(
        f"Phase confidence           : "
        f"{result.phase_confidence:.2f}%"
    )

    # ---------------------------------------------------------
    # Validate carrier correction.
    # ---------------------------------------------------------

    recovered_symbols = (
        result.signal.samples[
            :: int(
                round(
                    sample_rate
                    / symbol_rate
                )
            )
        ]
    )

    recovered_symbols = (
        recovered_symbols[
            : len(transmitted_symbols)
        ]
    )

    transmitted = (
        transmitted_symbols[
            : len(recovered_symbols)
        ]
    )

    decisions = qpsk_decision(
        recovered_symbols
    )

    ideal_constellation = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    expected_indices = []

    for symbol in transmitted:

        distances = np.abs(
            symbol
            - ideal_constellation
        )

        expected_indices.append(
            np.argmin(distances)
        )

    expected_indices = np.asarray(
        expected_indices
    )

    errors = int(
        np.sum(
            decisions
            != expected_indices
        )
    )

    ber = (
        errors
        / len(expected_indices)
    )

    print("\n[3] Symbol validation")

    print(
        f"Compared symbols : "
        f"{len(expected_indices)}"
    )

    print(
        f"Symbol errors    : "
        f"{errors}"
    )

    print(
        f"SER              : "
        f"{ber:.6f}"
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

    if len(recovered_symbols) < 900:
        raise AssertionError(
            "Too few symbols recovered."
        )

    if ber > 0.01:
        raise AssertionError(
            "Carrier recovery symbol error rate "
            "is too high."
        )

    print(
        "✓ Frequency offset estimation passed."
    )

    print(
        "✓ Frequency correction passed."
    )

    print(
        "✓ Phase recovery executed."
    )

    print(
        "✓ QPSK constellation recovery passed."
    )

    print(
        "✓ Symbol validation passed."
    )

    print("\n" + "=" * 70)
    print(
        "✓ CARRIER & PHASE RECOVERY TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
