import numpy as np

from prototype.core.signal import Signal
from prototype.core.synchronizer import synchronize_signal


def generate_bpsk(
    sample_rate=8000,
    symbol_rate=100,
    duration=2.0,
    frequency=500.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    samples_per_symbol = int(
        sample_rate / symbol_rate
    )

    num_symbols = int(
        duration * symbol_rate
    )

    bits = rng.integers(
        0,
        2,
        num_symbols,
    )

    symbols = 2 * bits - 1

    baseband = np.repeat(
        symbols,
        samples_per_symbol,
    ).astype(np.complex128)

    t = np.arange(
        len(baseband)
    ) / sample_rate

    carrier = np.exp(
        1j * 2 * np.pi * frequency * t
    )

    return baseband * carrier


def main():

    print("=" * 70)
    print("V2 SYNCHRONIZATION TEST")
    print("=" * 70)

    sample_rate = 8000
    symbol_rate = 100

    samples = generate_bpsk(
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        duration=2.0,
        frequency=500.0,
        seed=42,
    )

    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": "BPSK synchronization",
            "known_symbol_rate": symbol_rate,
        },
    )

    print("\n[1] Input signal")

    print(
        f"Sample rate : "
        f"{signal.sample_rate:.2f} Hz"
    )

    print(
        f"Samples     : "
        f"{signal.num_samples}"
    )

    print(
        f"Duration    : "
        f"{signal.duration:.2f} s"
    )

    print(
        f"Known symbol rate : "
        f"{symbol_rate} baud"
    )

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------

    print("\n[2] Running synchronization...")

    result = synchronize_signal(
        signal,
        min_symbol_rate=20,
        max_symbol_rate=500,
    )

    print(
        f"Estimated symbol rate : "
        f"{result.estimated_symbol_rate:.2f} baud"
    )

    print(
        f"Samples per symbol    : "
        f"{result.samples_per_symbol:.2f}"
    )

    print(
        f"Timing offset         : "
        f"{result.timing_offset}"
    )

    print(
        f"Timing confidence     : "
        f"{result.timing_confidence:.2f}%"
    )

    print(
        f"Recovered symbols     : "
        f"{result.signal.num_samples}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n[3] Validation")

    symbol_rate_error = abs(
        result.estimated_symbol_rate
        - symbol_rate
    )

    print(
        f"Symbol-rate error : "
        f"{symbol_rate_error:.2f} baud"
    )

    if symbol_rate_error > 10:
        raise AssertionError(
            "Symbol-rate estimation failed."
        )

    expected_sps = (
        sample_rate / symbol_rate
    )

    sps_error = abs(
        result.samples_per_symbol
        - expected_sps
    )

    print(
        f"SPS error         : "
        f"{sps_error:.2f}"
    )

    if sps_error > 10:
        raise AssertionError(
            "Samples-per-symbol estimation failed."
        )

    if result.signal.num_samples < 100:
        raise AssertionError(
            "Too few synchronized symbols recovered."
        )

    print(
        "✓ Symbol-rate estimation passed."
    )

    print(
        "✓ Samples-per-symbol estimation passed."
    )

    print(
        "✓ Symbol recovery passed."
    )

    print("\n" + "=" * 70)
    print("✓ SYNCHRONIZATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()