import numpy as np

from prototype.core.timing import (
    recover_symbol_timing,
)

from prototype.modulation.classifier import (
    classify_from_constellation,
)


SAMPLE_RATE = 8000
SYMBOL_RATE = 100
SAMPLES_PER_SYMBOL = 80
NUM_SYMBOLS = 1000

SNR_VALUES = [
    20,
    15,
    10,
    5,
    0,
]

RNG = np.random.default_rng(2026)


# ============================================================
# GENERATE QPSK
# ============================================================

def generate_qpsk():

    bits = RNG.integers(
        0,
        2,
        NUM_SYMBOLS * 2,
        dtype=np.uint8
    )

    groups = bits.reshape(
        NUM_SYMBOLS,
        2
    )

    symbols = np.zeros(
        NUM_SYMBOLS,
        dtype=np.complex128
    )

    for index, group in enumerate(groups):

        b0 = int(group[0])
        b1 = int(group[1])

        if (b0, b1) == (0, 0):

            symbols[index] = (
                1 + 1j
            )

        elif (b0, b1) == (0, 1):

            symbols[index] = (
                -1 + 1j
            )

        elif (b0, b1) == (1, 1):

            symbols[index] = (
                -1 - 1j
            )

        else:

            symbols[index] = (
                1 - 1j
            )

    symbols = (
        symbols
        / np.sqrt(2.0)
    )

    signal = np.repeat(
        symbols,
        SAMPLES_PER_SYMBOL
    )

    return signal


# ============================================================
# ADD AWGN
# ============================================================

def add_awgn(
    signal,
    snr_db
):

    signal_power = np.mean(
        np.abs(signal) ** 2
    )

    snr_linear = (
        10 ** (snr_db / 10.0)
    )

    noise_power = (
        signal_power
        / snr_linear
    )

    noise_std = np.sqrt(
        noise_power / 2.0
    )

    noise = (
        RNG.normal(
            0,
            noise_std,
            len(signal)
        )
        +
        1j
        * RNG.normal(
            0,
            noise_std,
            len(signal)
        )
    )

    return (
        signal + noise
    )


# ============================================================
# MAIN
# ============================================================

def main():

    clean_signal = generate_qpsk()

    print()
    print(
        "=" * 65
    )

    print(
        "QPSK CONSTELLATION CLASSIFIER - SNR TEST"
    )

    print(
        "=" * 65
    )

    for snr_db in SNR_VALUES:

        noisy_signal = add_awgn(
            clean_signal,
            snr_db
        )

        # ----------------------------------------------------
        # Timing recovery
        # ----------------------------------------------------

        timing = recover_symbol_timing(
            noisy_signal,
            SAMPLE_RATE
        )

        # ----------------------------------------------------
        # Extract symbol-rate samples
        # ----------------------------------------------------

        position = (
            timing.timing_offset
        )

        symbols = []

        while position < len(
            noisy_signal
        ):

            index = int(
                round(position)
            )

            if index >= len(
                noisy_signal
            ):
                break

            symbols.append(
                noisy_signal[index]
            )

            position += (
                timing.estimated_sps
            )

        symbols = np.asarray(
            symbols,
            dtype=np.complex128
        )

        # ----------------------------------------------------
        # Constellation classification
        # ----------------------------------------------------

        detected, features = (
            classify_from_constellation(
                symbols
            )
        )

        print(
            "I centers:",
            np.round(features["i_centers"], 3),
            "| I occupancy:",
            np.round(features["i_occupancy"], 3)
        )

        print(
            "Q centers:",
            np.round(features["q_centers"], 3),
            "| Q occupancy:",
            np.round(features["q_occupancy"], 3)
        )

        print(
            "I spacing:",
            round(features["i_spacing_ratio"], 3),
            "| Q spacing:",
            round(features["q_spacing_ratio"], 3)
        )

        print()

        print(
            f"SNR={snr_db:>2} dB | "
            f"Detected={detected:<7} | "
            f"SPS={timing.estimated_sps:>6.1f} | "
            f"Confidence="
            f"{timing.confidence * 100:>5.1f}% | "
            f"AxisRatio="
            f"{features['axis_energy_ratio']:.3f}"
        )

        print(
            f"           "
            f"I2={features['i_error_2']:.4f} | "
            f"I4={features['i_error_4']:.4f} | "
            f"Q2={features['q_error_2']:.4f} | "
            f"Q4={features['q_error_4']:.4f}"
        )


if __name__ == "__main__":
    main()