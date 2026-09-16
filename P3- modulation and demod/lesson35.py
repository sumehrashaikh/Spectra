import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# Parameters
# -----------------------------
fs = 4000
symbol_rate = 100
samples_per_symbol = fs // symbol_rate

f0 = 400
f1 = 800

offset = 25
snr_db = 10

num_bits = 100

# -----------------------------
# Generate bits
# -----------------------------
bits = np.random.randint(0, 2, num_bits)

# -----------------------------
# Generate complex BFSK
# -----------------------------
tx = []

for bit in bits:

    f = f1 if bit == 1 else f0

    n = np.arange(samples_per_symbol)

    symbol = np.exp(
        1j * 2 * np.pi * f * n / fs
    )

    tx.extend(symbol)

tx = np.array(tx)

# -----------------------------
# Apply frequency offset
# -----------------------------
n_total = np.arange(len(tx))

offset_signal = tx * np.exp(
    1j * 2 * np.pi * offset * n_total / fs
)

# -----------------------------
# Add AWGN
# -----------------------------
signal_power = np.mean(np.abs(offset_signal) ** 2)
snr_linear = 10 ** (snr_db / 10)

noise_power = signal_power / snr_linear

noise = np.sqrt(noise_power / 2) * (
    np.random.randn(len(tx))
    + 1j * np.random.randn(len(tx))
)

received = offset_signal + noise
# -----------------------------
# Frequency correction
# -----------------------------
correction = np.exp(
    -1j * 2 * np.pi * offset * n_total / fs
)

corrected = received * correction
# -----------------------------
# Demodulate using ORIGINAL
# frequencies
# -----------------------------
recovered_bits = []

n = np.arange(samples_per_symbol)

ref0 = np.exp(
    -1j * 2 * np.pi * f0 * n / fs
)

ref1 = np.exp(
    -1j * 2 * np.pi * f1 * n / fs
)

for i in range(num_bits):

    start = i * samples_per_symbol
    end = start + samples_per_symbol

    segment = corrected[start:end]

    energy0 = np.abs(np.sum(segment * ref0))
    energy1 = np.abs(np.sum(segment * ref1))

    recovered_bits.append(
        0 if energy0 > energy1 else 1
    )

recovered_bits = np.array(recovered_bits)

# -----------------------------
# BER
# -----------------------------
errors = np.sum(bits != recovered_bits)
ber = errors / len(bits)

print("Frequency offset:", offset, "Hz")
print("Bit errors:", errors)
print("BER:", ber)

# -----------------------------
# Plot a short section
# -----------------------------
t = np.arange(len(received)) / fs

plt.plot(
    t[:800],
    received.real[:800]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("BFSK with +25 Hz Frequency Offset")

plt.grid()
plt.show()