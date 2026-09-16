import numpy as np

from prototype.modulation.classifier import (
    classify_from_constellation
)


def make_bpsk():

    bits = np.array(
        [0, 1] * 500
    )

    return (
        2 * bits - 1
    ).astype(
        np.complex128
    )


def make_qpsk():

    symbols = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ]
        * 250,
        dtype=np.complex128
    )

    return (
        symbols
        / np.sqrt(2)
    )


def make_16qam():

    levels = [
        -3,
        -1,
        1,
        3
    ]

    symbols = []

    for i in levels:

        for q in levels:

            symbols.append(
                i + 1j * q
            )

    symbols = np.array(
        symbols * 62
        + symbols[:8],
        dtype=np.complex128
    )

    return (
        symbols
        / np.sqrt(10)
    )


def main():

    signals = {
        "BPSK": make_bpsk(),
        "QPSK": make_qpsk(),
        "16-QAM": make_16qam(),
    }

    print()
    print(
        "CONSTELLATION CLASSIFIER TEST"
    )
    print(
        "=" * 50
    )

    for expected, symbols in signals.items():

        detected, features = (
            classify_from_constellation(
                symbols
            )
        )

        print()
        print(
            f"Expected: {expected}"
        )

        print(
            f"Detected: {detected}"
        )

        print(
            f"Axis ratio: "
            f"{features['axis_energy_ratio']:.3f}"
        )

        print(
            f"I 2-level error: "
            f"{features['i_error_2']:.4f}"
        )

        print(
            f"I 4-level error: "
            f"{features['i_error_4']:.4f}"
        )

        print(
            f"Q 2-level error: "
            f"{features['q_error_2']:.4f}"
        )

        print(
            f"Q 4-level error: "
            f"{features['q_error_4']:.4f}"
        )


if __name__ == "__main__":
    main()