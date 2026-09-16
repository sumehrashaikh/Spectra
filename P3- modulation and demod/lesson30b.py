import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of bits
num_bits = 1000

# Generate random bits
bits = np.random.randint(0, 2, num_bits)

# BPSK mapping
symbols = 1 - 2 * bits

# Add complex AWGN
snr_db = 8
snr_linear = 10 ** (snr_db / 10)

signal_power = np.mean(np.abs(symbols) ** 2)
noise_power = signal_power / snr_linear

noise = np.sqrt(noise_power / 2) * (
    np.random.randn(num_bits)
    + 1j * np.random.randn(num_bits)
)

received = symbols + noise

# Constellation
plt.scatter(
    received.real,
    received.imag,
    s=10
)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")

plt.xlabel("I")
plt.ylabel("Q")
plt.title(f"BPSK with Noise — SNR = {snr_db} dB")
plt.grid()
plt.show()