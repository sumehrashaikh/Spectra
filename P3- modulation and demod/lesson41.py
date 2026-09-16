import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# Parameters
# -----------------------------
num_symbols = 200
sps = 8

# Fractional timing offset
timing_offset = 2.5

# -----------------------------
# Generate BPSK symbols
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

# Rectangular waveform
tx = np.repeat(symbols, sps)

# -----------------------------
# Fractional delay using interpolation
# -----------------------------
sample_index = np.arange(len(tx))

shifted_index = sample_index - timing_offset

rx = np.interp(
    shifted_index,
    sample_index,
    tx,
    left=0,
    right=0
)

# -----------------------------
# Test timing phases
# -----------------------------
phases = np.linspace(0, sps - 1, 100)

error_values = []

for phase in phases:

    sample_positions = (
        np.arange(num_symbols) * sps + phase
    )

    valid = (
        sample_positions >= 0
    ) & (
        sample_positions < len(rx)
    )

    samples = np.interp(
        sample_positions[valid],
        np.arange(len(rx)),
        rx
    )

    # Simple timing-quality measure:
    # how close samples are to ±1
    error = np.mean(
        np.abs(np.abs(samples) - 1)
    )

    error_values.append(error)

# -----------------------------
# Plot timing quality
# -----------------------------
plt.plot(
    phases,
    error_values
)

plt.xlabel("Sampling phase")
plt.ylabel("Timing error measure")
plt.title("Timing Quality vs Sampling Phase")
plt.grid()
plt.show()

best_phase = phases[np.argmin(error_values)]

print("Applied timing offset:", timing_offset)
print("Best estimated phase:", best_phase)