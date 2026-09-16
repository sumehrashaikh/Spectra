import numpy as np
import matplotlib.pyplot as plt


def rrc_filter(beta, span, sps):
    t = np.arange(-span * sps, span * sps + 1, dtype=float) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):

        if np.isclose(ti, 0.0):
            h[i] = 1 - beta + 4 * beta / np.pi

        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )

        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti
                * np.cos(np.pi * ti * (1 + beta))
            )

            denominator = (
                np.pi * ti * (1 - (4 * beta * ti) ** 2)
            )

            h[i] = numerator / denominator

    h /= np.sqrt(np.sum(h ** 2))
    return h


# -----------------------------
# Parameters
# -----------------------------
np.random.seed(42)

num_symbols = 1000
sps = 8
beta = 0.35
span = 8
snr_db = 8

# Deliberate timing offset
timing_offset = 3

# -----------------------------
# Generate BPSK
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

# Upsample
upsampled = np.zeros(num_symbols * sps)
upsampled[::sps] = symbols

# RRC shaping
rrc = rrc_filter(beta, span, sps)

tx = np.convolve(
    upsampled,
    rrc,
    mode="same"
)

# Add noise
signal_power = np.mean(tx ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))

noise = np.sqrt(noise_power) * np.random.randn(len(tx))

rx = tx + noise

# Create timing offset
rx_shifted = np.roll(rx, timing_offset)

# -----------------------------
# Try every timing phase
# -----------------------------
ber_values = []

for phase in range(sps):

    sampled = rx_shifted[phase::sps]

    sampled = sampled[:num_symbols]

    recovered_bits = (sampled < 0).astype(int)

    ber = np.mean(recovered_bits != bits)

    ber_values.append(ber)

# Best timing phase
best_phase = np.argmin(ber_values)
best_ber = ber_values[best_phase]

print("True timing offset:", timing_offset)
print("Best timing phase:", best_phase)
print("Best BER:", best_ber)

# -----------------------------
# Plot BER for each phase
# -----------------------------
plt.plot(
    range(sps),
    ber_values,
    "o-"
)

plt.xlabel("Timing phase (sample)")
plt.ylabel("BER")
plt.title("BER vs Timing Phase")
plt.xticks(range(sps))
plt.grid()
plt.show()