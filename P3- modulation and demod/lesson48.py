import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SIGNAL GENERATORS
# ============================================================

def generate_bpsk(num_symbols, sps):
    bits = np.random.randint(0, 2, num_symbols)
    symbols = 1 - 2 * bits

    return np.repeat(
        symbols,
        sps
    ).astype(complex)


def generate_qpsk(num_symbols, sps):
    bits = np.random.randint(0, 2, 2 * num_symbols)
    pairs = bits.reshape(-1, 2)

    mapping = {
        (0, 0): np.exp(1j * np.pi / 4),
        (0, 1): np.exp(1j * 3 * np.pi / 4),
        (1, 1): np.exp(1j * 5 * np.pi / 4),
        (1, 0): np.exp(1j * 7 * np.pi / 4)
    }

    symbols = np.array([
        mapping[tuple(pair)]
        for pair in pairs
    ])

    return np.repeat(
        symbols,
        sps
    )


def generate_16qam(num_symbols, sps):
    bits = np.random.randint(
        0,
        2,
        4 * num_symbols
    )

    groups = bits.reshape(-1, 4)

    gray_map = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 1): 1,
        (1, 0): 3
    }

    I = np.array([
        gray_map[tuple(group[:2])]
        for group in groups
    ])

    Q = np.array([
        gray_map[tuple(group[2:])]
        for group in groups
    ])

    symbols = I + 1j * Q

    symbols /= np.sqrt(
        np.mean(np.abs(symbols) ** 2)
    )

    return np.repeat(
        symbols,
        sps
    )


def generate_bfsk(num_symbols, sps, fs):
    bits = np.random.randint(
        0,
        2,
        num_symbols
    )

    f0 = 400
    f1 = 800

    n = np.arange(sps)

    output = []

    for bit in bits:

        frequency = f1 if bit == 1 else f0

        symbol = np.exp(
            1j * 2 * np.pi * frequency * n / fs
        )

        output.extend(symbol)

    return np.array(output)


# ============================================================
# AWGN
# ============================================================

def add_awgn(x, snr_db):

    signal_power = np.mean(
        np.abs(x) ** 2
    )

    snr_linear = 10 ** (
        snr_db / 10
    )

    noise_power = (
        signal_power / snr_linear
    )

    noise = np.sqrt(
        noise_power / 2
    ) * (
        np.random.randn(len(x))
        + 1j * np.random.randn(len(x))
    )

    return x + noise


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(x, fs, sps):

    # -----------------------------
    # Amplitude
    # -----------------------------

    amplitude = np.abs(x)

    mean_amplitude = np.mean(
        amplitude
    )

    amplitude_std = np.std(
        amplitude
    )

    amplitude_cv = (
        amplitude_std
        / (mean_amplitude + 1e-12)
    )

    # -----------------------------
    # Symbol-center samples
    # -----------------------------

    centers = x[
        sps // 2::sps
    ]

    phases = np.angle(
        centers
    )

    # -----------------------------
    # Frequency estimate per symbol
    # -----------------------------

    symbol_frequencies = []

    for i in range(
        0,
        len(x) - sps + 1,
        sps
    ):

        segment = x[
            i:i + sps
        ]

        phase = np.unwrap(
            np.angle(segment)
        )

        phase_change = (
            phase[-1] - phase[0]
        )

        frequency = (
            fs
            * phase_change
            / (
                2
                * np.pi
                * (sps - 1)
            )
        )

        symbol_frequencies.append(
            frequency
        )

    return {
        "amplitude_cv": amplitude_cv,
        "phases": phases,
        "symbol_frequencies":
            np.array(symbol_frequencies)
    }


# ============================================================
# PHASE COHERENCE
# ============================================================

def phase_coherence(phases, order):

    return np.abs(
        np.mean(
            np.exp(
                1j * order * phases
            )
        )
    )


# ============================================================
# MODULATION CLASSIFIER
# ============================================================

def classify_modulation(x, fs, sps):

    features = extract_features(
        x,
        fs,
        sps
    )

    amplitude_cv = (
        features["amplitude_cv"]
    )

    phases = features["phases"]

    symbol_frequencies = (
        features["symbol_frequencies"]
    )

    # -----------------------------
    # 1. FSK
    # -----------------------------

    mean_abs_frequency = np.mean(
        np.abs(symbol_frequencies)
    )

    frequency_std = np.std(
        symbol_frequencies
    )

    if (
        mean_abs_frequency > 100
        and frequency_std > 100
    ):

        return "BFSK", features

    # -----------------------------
    # 2. QAM
    # -----------------------------

    if amplitude_cv > 0.20:

        return "16-QAM", features

    # -----------------------------
    # 3. BPSK / QPSK
    # -----------------------------

    R2 = phase_coherence(
        phases,
        2
    )

    R4 = phase_coherence(
        phases,
        4
    )

    if R2 > 0.7:

        return "BPSK", features

    if R4 > 0.7:

        return "QPSK", features

    return "Unknown", features


# ============================================================
# CREATE TEST SIGNALS
# ============================================================

np.random.seed(42)

fs = 4000

symbol_rate = 100

sps = fs // symbol_rate

num_symbols = 500


signals = {

    "BPSK":
        generate_bpsk(
            num_symbols,
            sps
        ),

    "QPSK":
        generate_qpsk(
            num_symbols,
            sps
        ),

    "16-QAM":
        generate_16qam(
            num_symbols,
            sps
        ),

    "BFSK":
        generate_bfsk(
            num_symbols,
            sps,
            fs
        )
}


# ============================================================
# SNR STRESS TEST + CONFUSION MATRIX
# ============================================================

snr_values = [20, 15, 10, 5, 0, -5]

trials_per_modulation = 20

labels = [
    "BPSK",
    "QPSK",
    "16-QAM",
    "BFSK"
]

results = {}


for snr in snr_values:

    # confusion[actual][predicted]
    confusion = {
        actual: {
            predicted: 0
            for predicted in labels
        }
        for actual in labels
    }

    for true_label, x in signals.items():

        for _ in range(
            trials_per_modulation
        ):

            noisy = add_awgn(
                x,
                snr
            )

            predicted, _ = classify_modulation(
                noisy,
                fs,
                sps
            )

            if predicted not in labels:
                continue

            confusion[
                true_label
            ][
                predicted
            ] += 1

    # -----------------------------
    # Overall accuracy
    # -----------------------------

    correct = sum(
        confusion[label][label]
        for label in labels
    )

    total = (
        trials_per_modulation
        * len(labels)
    )

    accuracy = correct / total

    results[snr] = accuracy

    # -----------------------------
    # Print confusion information
    # -----------------------------

    print("\n" + "=" * 60)
    print(f"SNR = {snr} dB")

    print(
        f"Overall accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nConfusion matrix:")
    print(
        "Actual \\ Predicted",
        *labels
    )

    for actual in labels:

        row = [
            confusion[actual][predicted]
            for predicted in labels
        ]

        print(
            actual.ljust(18),
            *row
        )

    # -----------------------------
    # Per-class accuracy
    # -----------------------------

    print("\nPer-class accuracy:")

    for label in labels:

        class_total = sum(
            confusion[label].values()
        )

        class_correct = (
            confusion[label][label]
        )

        if class_total > 0:
            class_accuracy = (
                class_correct
                / class_total
            )
        else:
            class_accuracy = 0

        print(
            f"{label:8s}: "
            f"{class_accuracy * 100:.2f}%"
        )


# ============================================================
# ACCURACY VS SNR
# ============================================================

plt.plot(
    list(results.keys()),
    [
        value * 100
        for value in results.values()
    ],
    "o-"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Classification Accuracy (%)")
plt.title(
    "Modulation Classification Accuracy vs SNR"
)

plt.grid()
plt.show()

# ============================================================
# PLOT ACCURACY
# ============================================================

plt.plot(
    list(results.keys()),
    [
        value * 100
        for value in results.values()
    ],
    "o-"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Classification Accuracy (%)")
plt.title(
    "Modulation Classification Accuracy vs SNR"
)

plt.grid()
plt.show()