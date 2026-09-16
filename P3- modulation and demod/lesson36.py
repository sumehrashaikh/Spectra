import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000
symbol_rate = 100
samples_per_symbol = fs // symbol_rate

num_symbols = 20

# Random BPSK symbols
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

# Repeat each symbol
tx = np.repeat(symbols, samples_per_symbol)

# Create a timing offset
timing_offset = 3

# Shift signal
rx = np.roll(tx, timing_offset)

# Time axis
t = np.arange(len(rx)) / fs

# Plot
plt.plot(t, rx, drawstyle="steps-post")

plt.xlabel("Time (seconds)")
plt.ylabel("Symbol value")
plt.title("BPSK with Timing Offset")

plt.grid()
plt.show()

print("Samples per symbol:", samples_per_symbol)
print("Timing offset:", timing_offset, "samples")