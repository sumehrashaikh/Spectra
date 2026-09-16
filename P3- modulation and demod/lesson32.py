import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of symbols
num_symbols = 2000

# 4 bits per symbol
bits = np.random.randint(0, 2, 4 * num_symbols)

# Group into 4-bit symbols
bit_groups = bits.reshape(-1, 4)

# Gray-coded 2-bit mapping
gray_map = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): 1,
    (1, 0): 3
}

I = np.array([
    gray_map[tuple(pair[:2])]
    for pair in bit_groups
])

Q = np.array([
    gray_map[tuple(pair[2:])]
    for pair in bit_groups
])

# Normalize average symbol power to 1
symbols = I + 1j * Q
symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

# Add AWGN
snr_db = 18
snr_linear = 10 ** (snr_db / 10)

signal_power = np.mean(np.abs(symbols) ** 2)
noise_power = signal_power / snr_linear

noise = np.sqrt(noise_power / 2) * (
    np.random.randn(num_symbols)
    + 1j * np.random.randn(num_symbols)
)

received = symbols + noise

# Plot
plt.scatter(
    received.real,
    received.imag,
    s=8
)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")

plt.xlabel("I")
plt.ylabel("Q")
plt.title(f"16-QAM with Noise — SNR = {snr_db} dB")

plt.axis("equal")
plt.grid()
plt.show()