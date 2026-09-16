import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. Generate random bits
# -----------------------------
num_symbols = 2000

bits = np.random.randint(0, 2, 4 * num_symbols)
bit_groups = bits.reshape(-1, 4)

# -----------------------------
# 2. Gray-coded 16-QAM mapping
# -----------------------------
gray_map = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1):  1,
    (1, 0):  3
}

I = np.array([
    gray_map[tuple(pair[:2])]
    for pair in bit_groups
])

Q = np.array([
    gray_map[tuple(pair[2:])]
    for pair in bit_groups
])

symbols = I + 1j * Q

# Normalize average power
scale = np.sqrt(np.mean(np.abs(symbols) ** 2))
symbols = symbols / scale

# -----------------------------
# 3. Add AWGN
# -----------------------------
snr_db = 18
snr_linear = 10 ** (snr_db / 10)

signal_power = np.mean(np.abs(symbols) ** 2)
noise_power = signal_power / snr_linear

noise = np.sqrt(noise_power / 2) * (
    np.random.randn(num_symbols)
    + 1j * np.random.randn(num_symbols)
)

received = symbols + noise

# -----------------------------
# 4. Create ideal constellation
# -----------------------------
levels = np.array([-3, -1, 1, 3])

constellation = np.array([
    i + 1j * q
    for i in levels
    for q in levels
])

constellation = constellation / scale

# -----------------------------
# 5. Nearest-symbol detection
# -----------------------------
distances = np.abs(
    received[:, None] - constellation[None, :]
)

nearest_indices = np.argmin(distances, axis=1)

decisions = constellation[nearest_indices]

# -----------------------------
# 6. Convert detected symbols
#    back to bits
# -----------------------------
reverse_map = {
    -3: (0, 0),
    -1: (0, 1),
     1: (1, 1),
     3: (1, 0)
}

recovered = []

for symbol in decisions:

    i_level = int(np.round(symbol.real * scale))
    q_level = int(np.round(symbol.imag * scale))

    recovered.extend(
        reverse_map[i_level]
    )
    recovered.extend(
        reverse_map[q_level]
    )

recovered_bits = np.array(recovered)

# -----------------------------
# 7. BER
# -----------------------------
errors = np.sum(bits != recovered_bits)
ber = errors / len(bits)

print("Total bits:", len(bits))
print("Bit errors:", errors)
print("BER:", ber)

# -----------------------------
# 8. Plot
# -----------------------------
plt.scatter(
    received.real,
    received.imag,
    s=8
)

plt.xlabel("I")
plt.ylabel("Q")
plt.title(f"16-QAM Demodulation — SNR = {snr_db} dB")

plt.axis("equal")
plt.grid()
plt.show()