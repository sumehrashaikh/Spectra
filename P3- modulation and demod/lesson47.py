import numpy as np


# ============================================================
# SIGNAL GENERATORS
# ============================================================

def generate_bpsk(num_symbols, sps):
    """Generate ideal BPSK complex-baseband samples."""
    bits = np.random.randint(0, 2, num_symbols)

    # Gray/trivial BPSK mapping:
    # 0 -> +1
    # 1 -> -1
    symbols = 1 - 2 * bits

    signal = np.repeat(
        symbols,
        sps
    ).astype(complex)

    return signal


def generate_qpsk(num_symbols, sps):
    """Generate ideal QPSK complex-baseband samples."""
    bits = np.random.randint(0, 2, 2 * num_symbols)

    pairs = bits.reshape(-1, 2)

    # Gray-coded QPSK
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

    signal = np.repeat(
        symbols,
        sps
    )

    return signal


def generate_16qam(num_symbols, sps):
    """Generate normalized 16-QAM complex-baseband samples."""
    bits = np.random.randint(0, 2, 4 * num_symbols)

    groups = bits.reshape(-1, 4)

    # Gray-coded amplitude mapping
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

    # Normalize average symbol power to 1
    symbols /= np.sqrt(
        np.mean(np.abs(symbols) ** 2)
    )

    signal = np.repeat(
        symbols,
        sps
    )

    return signal


def generate_bfsk(num_symbols, sps, fs):
    """Generate ideal coherent BFSK complex-baseband samples."""
    bits = np.random.randint(0, 2, num_symbols)

    f0 = 400
    f1 = 800

    signal = []

    n = np.arange(sps)

    for bit in bits:

        frequency = f1 if bit == 1 else f0

        symbol = np.exp(
            1j * 2 * np.pi * frequency * n / fs
        )

        signal.extend(symbol)

    return np.array(signal)


# ============================================================
# AWGN
# ============================================================

def add_awgn(x, snr_db):
    """Add complex AWGN at the requested SNR."""
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

    # --------------------------------------------------------
    # Amplitude
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Symbol-center samples
    # --------------------------------------------------------

    centers = x[
        sps // 2::sps
    ]

    # --------------------------------------------------------
    # Phase at symbol centers
    # --------------------------------------------------------

    phases = np.angle(centers)

    # --------------------------------------------------------
    # Estimate instantaneous frequency
    # within each symbol
    # --------------------------------------------------------

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

    symbol_frequencies = np.array(
        symbol_frequencies
    )

    return {
        "amplitude_cv": amplitude_cv,
        "phases": phases,
        "symbol_frequencies":
            symbol_frequencies
    }


# ============================================================
# CIRCULAR ANGLE DISTANCE
# ============================================================

def angle_distance(a, b):
    """Shortest angular distance."""
    return np.abs(
        np.angle(
            np.exp(
                1j * (a - b)
            )
        )
    )


# ============================================================
# PSK PHASE COHERENCE
# ============================================================

def phase_coherence(phases, order):
    """
    Measure m-th order phase coherence.

    For BPSK:
        second-order coherence is strong.

    For QPSK:
        fourth-order coherence is strong.
    """

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

    # --------------------------------------------------------
    # 1. Check for FSK
    # --------------------------------------------------------

    mean_abs_frequency = np.mean(
        np.abs(symbol_frequencies)
    )

    frequency_std = np.std(
        symbol_frequencies
    )

    # This threshold is ONLY for our
    # controlled demonstration.
    if (
        mean_abs_frequency > 100
        and frequency_std > 100
    ):

        return "BFSK", features

    # --------------------------------------------------------
    # 2. Check for QAM
    # --------------------------------------------------------

    # Noise creates amplitude variation
    # even in constant-envelope signals.
    #
    # Our demonstration separates QAM
    # from the PSK signals around this region.
    if amplitude_cv > 0.20:

        return "16-QAM", features

    # --------------------------------------------------------
    # 3. Distinguish BPSK vs QPSK
    # --------------------------------------------------------

    R2 = phase_coherence(
        phases,
        2
    )

    R4 = phase_coherence(
        phases,
        4
    )

    # BPSK should have strong
    # second-order phase coherence.
    if R2 > 0.7:

        return "BPSK", features

    # QPSK should have stronger
    # fourth-order coherence.
    if R4 > 0.7:

        return "QPSK", features

    return "Unknown", features


# ============================================================
# MAIN
# ============================================================

np.random.seed(42)

fs = 4000

symbol_rate = 100

sps = fs // symbol_rate

num_symbols = 500

snr_db = 15


# ============================================================
# Generate test signals
# ============================================================

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
# Classification test
# ============================================================

correct = 0
total = len(signals)

for true_label, x in signals.items():

    noisy = add_awgn(
        x,
        snr_db
    )

    predicted, features = (
        classify_modulation(
            noisy,
            fs,
            sps
        )
    )

    phases = features["phases"]

    R2 = phase_coherence(
        phases,
        2
    )

    R4 = phase_coherence(
        phases,
        4
    )

    mean_frequency = np.mean(
        np.abs(
            features["symbol_frequencies"]
        )
    )

    print("\n" + "=" * 45)

    print(
        f"Actual:    {true_label}"
    )

    print(
        f"Predicted: {predicted}"
    )

    print(
        f"Amplitude CV: "
        f"{features['amplitude_cv']:.3f}"
    )

    print(
        f"Mean |symbol frequency|: "
        f"{mean_frequency:.2f} Hz"
    )

    print(
        f"R2 phase coherence: "
        f"{R2:.3f}"
    )

    print(
        f"R4 phase coherence: "
        f"{R4:.3f}"
    )

    if predicted == true_label:
        correct += 1
        print("Result: ✅ CORRECT")
    else:
        print("Result: ❌ WRONG")


# ============================================================
# Accuracy
# ============================================================

accuracy = correct / total

print("\n" + "=" * 45)

print(
    f"Accuracy: "
    f"{correct}/{total} "
    f"= {accuracy * 100:.1f}%"
)