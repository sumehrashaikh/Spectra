import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 1000
symbol_rate = 100
samples_per_symbol = fs // symbol_rate

num_symbols = 1000
timing_offset = 4

# Generate bits and BPSK symbols
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

# Rectangular pulse shaping
tx = np.repeat(symbols, samples_per_symbol)

# Apply timing offset
rx = np.roll(tx, timing_offset)

# Receiver samples every symbol period,
# but assumes timing starts at sample 0.
sampled = rx[::samples_per_symbol]

# Make lengths match
sampled = sampled[:num_symbols]

# BPSK decision
recovered_bits = (sampled < 0).astype(int)

# BER
errors = np.sum(bits != recovered_bits)
ber = errors / len(bits)

print("Samples per symbol:", samples_per_symbol)
print("Timing offset:", timing_offset, "samples")
print("Bit errors:", errors)
print("BER:", ber)

# Show a short section
t = np.arange(len(rx)) / fs

plt.plot(
    t[:200],
    rx[:200],
    drawstyle="steps-post"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Symbol")
plt.title("BPSK with Timing Offset")
plt.grid()
plt.show()