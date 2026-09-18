import numpy as np

from prototype.core.signal import Signal
from prototype.classification.classifier import (
    classify_signal,
)


def generate_bpsk(
    num_symbols=500,
    samples_per_symbol=8,
    seed=42,
):
    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        num_symbols,
    )

    symbols = (
        2 * bits - 1
    ).astype(float)

    samples = np.repeat(
        symbols,
        samples_per_symbol,
    )

    return (
        samples.astype(np.complex128),
        symbols,
    )


def generate_qpsk(
    num_symbols=500,
    samples_per_symbol=8,
    seed=42,
):
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

    samples = np.repeat(
        symbols,
        samples_per_symbol,
    )

    return (
        samples,
        symbols,
    )


def generate_16qam(
    num_symbols=500,
    samples_per_symbol=8,
    seed=42,
):
    rng = np.random.default_rng(seed)

    levels = np.array(
        [
            -3,
            -1,
            1,
            3,
        ],
        dtype=float,
    )

    i = rng.choice(
        levels,
        num_symbols,
    )

    q = rng.choice(
        levels,
        num_symbols,
    )

    symbols = (
        i + 1j * q
    )

    symbols = (
        symbols
        / np.sqrt(
            np.mean(
                np.abs(symbols) ** 2
            )
        )
    )

    samples = np.repeat(
        symbols,
        samples_per_symbol,
    )

    return (
        samples.astype(np.complex128),
        symbols,
    )


def generate_bfsk(
    num_symbols=500,
    samples_per_symbol=80,
    sample_rate=8000,
    f0=500,
    f1=700,
    seed=42,
):
    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        num_symbols,
    )

    samples = []

    for bit in bits:

        frequency = (
            f1
            if bit
            else f0
        )

        t = (
            np.arange(
                samples_per_symbol
            )
            / sample_rate
        )

        symbol = np.exp(
            1j
            * 2.0
            * np.pi
            * frequency
            * t
        )

        samples.extend(symbol)

    return np.asarray(
        samples,
        dtype=np.complex128,
    )


def run_test(
    name,
    samples,
    sample_rate,
    expected,
    synchronized=False,
):
    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": name,
            "synchronized": synchronized,
        },
    )

    result = classify_signal(
        signal,
        use_constellation=synchronized,
        synchronized=synchronized,
    )

    print(
        f"{name:<10} → "
        f"{result.modulation:<8} "
        f"confidence="
        f"{result.confidence:.2f}% "
        f"method="
        f"{result.method}"
    )

    if result.modulation != expected:

        raise AssertionError(
            f"{name}: expected "
            f"{expected}, got "
            f"{result.modulation}"
        )

    return result


def main():

    print("=" * 70)
    print("V2 CLASSIFICATION PIPELINE TEST")
    print("=" * 70)

    sample_rate = 8000

    # ---------------------------------------------------------
    # BPSK
    # ---------------------------------------------------------

    print("\n[1] BPSK")

    bpsk_samples, _ = (
        generate_bpsk()
    )

    run_test(
        "BPSK",
        bpsk_samples,
        sample_rate,
        "BPSK",
        synchronized=True,
    )

    # ---------------------------------------------------------
    # QPSK
    # ---------------------------------------------------------

    print("\n[2] QPSK")

    qpsk_samples, _ = (
        generate_qpsk()
    )

    run_test(
        "QPSK",
        qpsk_samples,
        sample_rate,
        "QPSK",
        synchronized=True,
    )

    # ---------------------------------------------------------
    # 16-QAM
    # ---------------------------------------------------------

    print("\n[3] 16-QAM")

    qam_samples, _ = (
        generate_16qam()
    )

    run_test(
        "16-QAM",
        qam_samples,
        sample_rate,
        "16-QAM",
        synchronized=True,
    )

    # ---------------------------------------------------------
    # BFSK
    # ---------------------------------------------------------

    print("\n[4] BFSK")

    bfsk_samples = generate_bfsk(
        sample_rate=sample_rate,
    )

    run_test(
        "BFSK",
        bfsk_samples,
        sample_rate,
        "BFSK",
        synchronized=False,
    )

    # ---------------------------------------------------------
    # Unknown / pure tone
    # ---------------------------------------------------------

    print("\n[5] Unknown signal")

    t = (
        np.arange(8000)
        / sample_rate
    )

    tone = np.exp(
        1j
        * 2.0
        * np.pi
        * 1000
        * t
    )

    tone_signal = Signal(
        samples=tone,
        sample_rate=sample_rate,
        metadata={
            "test": "pure tone",
            "synchronized": False,
        },
    )

    tone_result = classify_signal(
        tone_signal,
        use_constellation=False,
        synchronized=False,
    )

    print(
        f"Pure tone → "
        f"{tone_result.modulation} "
        f"method="
        f"{tone_result.method}"
    )

    if tone_result.modulation != "Unknown":
        raise AssertionError(
            "Pure tone: expected Unknown, "
            f"got {tone_result.modulation}"
        )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    print("\n[6] Validation")

    print(
        "✓ BPSK classification passed."
    )

    print(
        "✓ QPSK classification passed."
    )

    print(
        "✓ 16-QAM classification passed."
    )

    print(
        "✓ BFSK classification passed."
    )

    print(
        "✓ Unknown-signal handling passed."
    )

    print(
        "✓ V2 classification wrapper passed."
    )

    print("\n" + "=" * 70)

    print(
        "✓ CLASSIFICATION PIPELINE TEST PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()