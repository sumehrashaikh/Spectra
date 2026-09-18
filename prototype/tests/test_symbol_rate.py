from prototype.parameters.symbol_rate import (
    generate_rrc_bpsk,
    estimate_symbol_rate,
)


def run_test(
    sample_rate,
    symbol_rate,
    duration=5.0,
):
    print("\n" + "-" * 60)

    print(
        f"Testing symbol rate: "
        f"{symbol_rate} baud"
    )

    samples, bits = generate_rrc_bpsk(
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        duration=duration,
        seed=42,
    )

    result = estimate_symbol_rate(
        samples,
        sample_rate,
        min_symbol_rate=50,
        max_symbol_rate=400,
    )

    print(
        f"Estimated rate : "
        f"{result.symbol_rate:.2f} baud"
    )

    print(
        f"Expected rate  : "
        f"{symbol_rate:.2f} baud"
    )

    print(
        f"SPS             : "
        f"{result.samples_per_symbol:.2f}"
    )

    print(
        f"Confidence      : "
        f"{result.confidence:.2f}%"
    )

    print(
        f"Method          : "
        f"{result.method}"
    )

    error = abs(
        result.symbol_rate
        - symbol_rate
    )

    print(
        f"Error           : "
        f"{error:.2f} baud"
    )

    return error


def main():

    print("=" * 70)
    print("V2 SYMBOL-RATE ESTIMATION TEST")
    print("=" * 70)

    sample_rate = 8000

    test_rates = [
        100,
        200,
        250,
    ]

    errors = []

    for rate in test_rates:

        error = run_test(
            sample_rate,
            rate,
        )

        errors.append(
            error
        )

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    for rate, error in zip(
        test_rates,
        errors,
    ):
        print(
            f"{rate:>4} baud → "
            f"error = {error:.2f} baud"
        )

    # Allow FFT-bin resolution and estimation
    # tolerance while still requiring the correct
    # symbol-rate region.
    for rate, error in zip(
        test_rates,
        errors,
    ):
        tolerance = max(
            10.0,
            rate * 0.15,
        )

        if error > tolerance:
            raise AssertionError(
                f"Symbol-rate estimation failed "
                f"for {rate} baud."
            )

    print(
        "\n✓ 100 baud estimation passed."
    )

    print(
        "✓ 200 baud estimation passed."
    )

    print(
        "✓ 250 baud estimation passed."
    )

    print("\n" + "=" * 70)
    print(
        "✓ SYMBOL-RATE ESTIMATION TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()