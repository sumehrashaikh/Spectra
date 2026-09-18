import numpy as np

from prototype.core.signal import Signal
from prototype.demodulation.demodulator import (
    demodulate_signal,
    calculate_ber,
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
        bits,
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
        bits.reshape(-1),
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

    # Convert each axis back into the
    # same Gray mapping used by V1.

    normalized_levels = (
        levels / np.sqrt(10.0)
    )

    bits_map = {
        -3: (0, 0),
        -1: (0, 1),
        1: (1, 1),
        3: (1, 0),
    }

    # Since the generated symbols are normalized
    # using their actual RMS, determine their
    # original levels using the normalized
    # constellation geometry.

    raw_i = np.round(
        i
    ).astype(int)

    raw_q = np.round(
        q
    ).astype(int)

    reference_bits = []

    for i_level, q_level in zip(
        raw_i,
        raw_q,
    ):
        reference_bits.extend(
            bits_map[int(i_level)]
        )

        reference_bits.extend(
            bits_map[int(q_level)]
        )

    samples = np.repeat(
        symbols,
        samples_per_symbol,
    )

    return (
        samples.astype(np.complex128),
        np.asarray(
            reference_bits,
            dtype=np.uint8,
        ),
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

        samples.extend(
            symbol
        )

    return (
        np.asarray(
            samples,
            dtype=np.complex128,
        ),
        bits,
    )


def validate(
    name,
    samples,
    sample_rate,
    modulation,
    reference_bits,
    samples_per_symbol,
    timing_offset=0,
    freq_0=None,
    freq_1=None,
):
    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": name,
        },
    )

    result = demodulate_signal(
        signal,
        modulation=modulation,
        samples_per_symbol=samples_per_symbol,
        timing_offset=timing_offset,
        freq_0=freq_0,
        freq_1=freq_1,
    )

    ber = calculate_ber(
        result.bits,
        reference_bits,
    )

    print(
        f"{name:<8} → "
        f"{result.modulation:<7} "
        f"bits={result.num_bits:<5} "
        f"BER={ber['direct_ber']:.6f} "
        f"margin={result.decision_margin:.4f}"
    )

    if ber["direct_ber"] != 0.0:
        raise AssertionError(
            f"{name}: BER is not zero."
        )


def main():

    print("=" * 70)
    print("V2 DEMODULATION PIPELINE TEST")
    print("=" * 70)

    sample_rate = 8000

    print("\n[1] BPSK")

    bpsk_samples, bpsk_bits = generate_bpsk()

    validate(
        "BPSK",
        bpsk_samples,
        sample_rate,
        "BPSK",
        bpsk_bits,
        8,
    )

    print("\n[2] QPSK")

    qpsk_samples, qpsk_bits = generate_qpsk()

    validate(
        "QPSK",
        qpsk_samples,
        sample_rate,
        "QPSK",
        qpsk_bits,
        8,
    )

    print("\n[3] 16-QAM")

    qam_samples, qam_bits = generate_16qam()

    validate(
        "16-QAM",
        qam_samples,
        sample_rate,
        "16-QAM",
        qam_bits,
        8,
    )

    print("\n[4] BFSK")

    bfsk_samples, bfsk_bits = generate_bfsk(
        sample_rate=sample_rate,
    )

    validate(
        "BFSK",
        bfsk_samples,
        sample_rate,
        "BFSK",
        bfsk_bits,
        80,
        freq_0=500,
        freq_1=700,
    )

    print("\n" + "=" * 70)
    print("✓ DEMODULATION PIPELINE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()