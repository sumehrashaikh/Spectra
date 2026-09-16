import numpy as np


def describe_signal(name, x, fs):
    # -----------------------------
    # Amplitude
    # -----------------------------
    amplitude = np.abs(x)

    amplitude_mean = np.mean(amplitude)
    amplitude_std = np.std(amplitude)

    # Coefficient of variation
    amplitude_cv = amplitude_std / (amplitude_mean + 1e-12)

    # -----------------------------
    # Phase
    # -----------------------------
    phase = np.unwrap(np.angle(x))

    # Phase difference
    phase_diff = np.diff(phase)

    # -----------------------------
    # Instantaneous frequency
    # -----------------------------
    inst_freq = (
        fs / (2 * np.pi)
    ) * phase_diff

    # Ignore extreme values caused by
    # symbol transitions / numerical effects
    valid_freq = inst_freq[
        np.abs(inst_freq) < fs / 2
    ]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Mean amplitude: {amplitude_mean:.4f}")
    print(f"Amplitude std:  {amplitude_std:.4f}")
    print(f"Amplitude CV:   {amplitude_cv:.4f}")
    print(
        f"Instantaneous-frequency std: "
        f"{np.std(valid_freq):.2f} Hz"
    )


# =========================================================
# Common parameters
# =========================================================
np.random.seed(42)

fs = 4000
symbol_rate = 100
sps = fs // symbol_rate

num_symbols = 500


# =========================================================
# BPSK
# =========================================================
bits_bpsk = np.random.randint(0, 2, num_symbols)

bpsk_symbols = 1 - 2 * bits_bpsk

bpsk = np.repeat(
    bpsk_symbols,
    sps
)

bpsk = bpsk.astype(complex)


# =========================================================
# QPSK
# =========================================================
bits_qpsk = np.random.randint(
    0,
    2,
    2 * num_symbols
)

pairs = bits_qpsk.reshape(-1, 2)

qpsk_map = {
    (0, 0): np.exp(1j * np.pi / 4),
    (0, 1): np.exp(1j * 3 * np.pi / 4),
    (1, 1): np.exp(1j * 5 * np.pi / 4),
    (1, 0): np.exp(1j * 7 * np.pi / 4),
}

qpsk_symbols = np.array([
    qpsk_map[tuple(pair)]
    for pair in pairs
])

qpsk = np.repeat(
    qpsk_symbols,
    sps
)


# =========================================================
# 16-QAM
# =========================================================
bits_qam = np.random.randint(
    0,
    2,
    4 * num_symbols
)

groups = bits_qam.reshape(-1, 4)

gray_map = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): 1,
    (1, 0): 3,
}

I = np.array([
    gray_map[tuple(g[:2])]
    for g in groups
])

Q = np.array([
    gray_map[tuple(g[2:])]
    for g in groups
])

qam_symbols = I + 1j * Q

qam_symbols /= np.sqrt(
    np.mean(np.abs(qam_symbols) ** 2)
)

qam = np.repeat(
    qam_symbols,
    sps
)


# =========================================================
# BFSK
# =========================================================
bits_fsk = np.random.randint(
    0,
    2,
    num_symbols
)

f0 = 400
f1 = 800

fsk = []

for bit in bits_fsk:

    frequency = f1 if bit == 1 else f0

    n = np.arange(sps)

    symbol = np.exp(
        1j * 2 * np.pi * frequency * n / fs
    )

    fsk.extend(symbol)

fsk = np.array(fsk)


# =========================================================
# Feature extraction
# =========================================================
describe_signal(
    "BPSK",
    bpsk,
    fs
)

describe_signal(
    "QPSK",
    qpsk,
    fs
)

describe_signal(
    "16-QAM",
    qam,
    fs
)

describe_signal(
    "BFSK",
    fsk,
    fs
)