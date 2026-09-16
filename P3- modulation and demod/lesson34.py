import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# Signal parameters
# -----------------------------
fs = 4000               # Sampling frequency
symbol_rate = 100       # Symbols per second
samples_per_symbol = int(fs / symbol_rate)

f0 = 400                 # Frequency for bit 0
f1 = 800                 # Frequency for bit 1

num_bits = 20

# -----------------------------
# Generate random bits
# -----------------------------
bits = np.random.randint(0, 2, num_bits)

# -----------------------------
# Generate BFSK waveform
# -----------------------------
signal_bfsk = []

for bit in bits:

    frequency = f1 if bit == 1 else f0

    n = np.arange(samples_per_symbol)

    symbol = np.cos(
        2 * np.pi * frequency * n / fs
    )

    signal_bfsk.extend(symbol)

signal_bfsk = np.array(signal_bfsk)

# Time axis
t = np.arange(len(signal_bfsk)) / fs

# -----------------------------
# Add noise
# -----------------------------
snr_db = 10

signal_power = np.mean(signal_bfsk ** 2)
snr_linear = 10 ** (snr_db / 10)
noise_power = signal_power / snr_linear

noise = np.sqrt(noise_power) * np.random.randn(
    len(signal_bfsk)
)

received = signal_bfsk + noise

# -----------------------------
# Plot
# -----------------------------
plt.plot(t, received)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"BFSK Signal — SNR = {snr_db} dB")

plt.grid()
plt.show()

print("Transmitted bits:")
print(bits)
# -----------------------------
# BFSK demodulation
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

    segment = received[start:end]

    # Correlation magnitude
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

print("Recovered bits:")
print(recovered_bits)

print("Bit errors:", errors)
print("BER:", ber)