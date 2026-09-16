import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of symbols
num_symbols = 1000

# Generate random bits
bits = np.random.randint(0, 2, 2 * num_symbols)

# Group bits into pairs
bit_pairs = bits.reshape(-1, 2)

# Gray-coded QPSK mapping
mapping = {
    (0, 0): np.exp(1j * np.pi / 4),
    (0, 1): np.exp(1j * 3 * np.pi / 4),
    (1, 1): np.exp(1j * 5 * np.pi / 4),
    (1, 0): np.exp(1j * 7 * np.pi / 4)
}

symbols = np.array([
    mapping[tuple(pair)]
    for pair in bit_pairs
])

# Add AWGN
snr_db = 10
snr_linear = 10 ** (snr_db / 10)

signal_power = np.mean(np.abs(symbols) ** 2)
noise_power = signal_power / snr_linear

noise = np.sqrt(noise_power / 2) * (
    np.random.randn(num_symbols)
    + 1j * np.random.randn(num_symbols)
)

received = symbols + noise

# Plot constellation
plt.scatter(
    received.real,
    received.imag,
    s=10
)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")

plt.xlabel("I")
plt.ylabel("Q")
plt.title(f"QPSK with Noise — SNR = {snr_db} dB")

plt.axis("equal")
plt.grid()
plt.show()

# Ideal QPSK constellation
constellation = np.array([
    np.exp(1j * np.pi / 4),
    np.exp(1j * 3 * np.pi / 4),
    np.exp(1j * 5 * np.pi / 4),
    np.exp(1j * 7 * np.pi / 4)
])

# Find nearest constellation point
distances = np.abs(
    received[:, None] - constellation[None, :]
)

decisions = np.argmin(distances, axis=1)

# Map decisions back to bits
bit_table = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
])

recovered_bits = bit_table[decisions].reshape(-1)

# BER
errors = np.sum(bits != recovered_bits)
ber = errors / len(bits)

print("Total bits:", len(bits))
print("Bit errors:", errors)
print("BER:", ber)