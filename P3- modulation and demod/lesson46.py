import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 4000
symbol_rate = 100
sps = fs // symbol_rate
num_symbols = 500


# -----------------------------
# BPSK
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
bpsk_symbols = 1 - 2 * bits

bpsk = np.repeat(
    bpsk_symbols,
    sps
).astype(complex)


# -----------------------------
# QPSK
# -----------------------------
bits = np.random.randint(0, 2, 2 * num_symbols)
pairs = bits.reshape(-1, 2)

qpsk_map = {
    (0, 0): np.exp(1j * np.pi / 4),
    (0, 1): np.exp(1j * 3 * np.pi / 4),
    (1, 1): np.exp(1j * 5 * np.pi / 4),
    (1, 0): np.exp(1j * 7 * np.pi / 4)
}

qpsk_symbols = np.array([
    qpsk_map[tuple(pair)]
    for pair in pairs
])

qpsk = np.repeat(qpsk_symbols, sps)


# -----------------------------
# 16-QAM
# -----------------------------
bits = np.random.randint(0, 2, 4 * num_symbols)
groups = bits.reshape(-1, 4)

gray_map = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): 1,
    (1, 0): 3
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

qam = np.repeat(qam_symbols, sps)


# -----------------------------
# BFSK
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)

f0 = 400
f1 = 800

fsk = []

for bit in bits:

    frequency = f1 if bit == 1 else f0

    n = np.arange(sps)

    symbol = np.exp(
        1j * 2 * np.pi * frequency * n / fs
    )

    fsk.extend(symbol)

fsk = np.array(fsk)


# -----------------------------
# Instantaneous frequency
# -----------------------------
signals = {
    "BPSK": bpsk,
    "QPSK": qpsk,
    "16-QAM": qam,
    "BFSK": fsk
}

plt.figure(figsize=(10, 6))

for name, x in signals.items():

    phase = np.unwrap(np.angle(x))

    inst_freq = (
        fs / (2 * np.pi)
    ) * np.diff(phase)

    # Ignore extreme numerical values
    inst_freq = inst_freq[
        np.abs(inst_freq) < fs / 2
    ]

    plt.hist(
        inst_freq,
        bins=80,
        alpha=0.45,
        label=name
    )

plt.xlabel("Instantaneous Frequency (Hz)")
plt.ylabel("Count")
plt.title("Instantaneous-Frequency Distributions")
plt.legend()
plt.grid()
plt.show()