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

num_symbols = 100
sps = 8
beta = 0.35
span = 8
snr_db = 8

# -----------------------------
# BPSK symbols
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

# -----------------------------
# Upsample
# -----------------------------
upsampled = np.zeros(num_symbols * sps)
upsampled[::sps] = symbols

# -----------------------------
# Transmit RRC filter
# -----------------------------
rrc = rrc_filter(beta, span, sps)

tx = np.convolve(
    upsampled,
    rrc,
    mode="same"
)

# -----------------------------
# Add AWGN
# -----------------------------
signal_power = np.mean(tx ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))

noise = np.sqrt(noise_power) * np.random.randn(len(tx))

rx = tx + noise

# -----------------------------
# Matched filter
# -----------------------------
rx_filtered = np.convolve(
    rx,
    rrc,
    mode="same"
)

# -----------------------------
# Plot comparison
# -----------------------------
t = np.arange(len(rx)) / sps

plt.plot(
    t[:200],
    rx[:200],
    alpha=0.5,
    label="Received + noise"
)

plt.plot(
    t[:200],
    rx_filtered[:200],
    label="After matched filter"
)

plt.xlabel("Symbol periods")
plt.ylabel("Amplitude")
plt.title(f"Matched Filtering — SNR = {snr_db} dB")
plt.legend()
plt.grid()
plt.show()