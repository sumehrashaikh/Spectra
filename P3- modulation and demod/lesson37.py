import numpy as np
import matplotlib.pyplot as plt


def rrc_filter(beta, span, sps):
    """
    Root Raised Cosine filter.

    beta : roll-off factor, 0 < beta <= 1
    span : filter length in symbols on EACH side
    sps  : samples per symbol
    """
    t = np.arange(-span * sps, span * sps + 1, dtype=float) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):

        # t = 0
        if np.isclose(ti, 0.0):
            h[i] = 1 - beta + (4 * beta / np.pi)

        # t = ±1 / (4*beta)
        elif beta > 0 and np.isclose(
            abs(ti), 1 / (4 * beta)
        ):
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * beta))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * beta))
                )
            )

        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                +
                4 * beta * ti
                * np.cos(np.pi * ti * (1 + beta))
            )

            denominator = (
                np.pi * ti
                * (1 - (4 * beta * ti) ** 2)
            )

            h[i] = numerator / denominator

    # Normalize filter energy
    h /= np.sqrt(np.sum(h ** 2))

    return h


# -----------------------------
# Parameters
# -----------------------------
np.random.seed(42)

num_symbols = 50
sps = 8                 # samples per symbol
beta = 0.35             # roll-off factor
span = 8                # filter span in symbols

# -----------------------------
# Generate BPSK symbols
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

# -----------------------------
# Upsample
# -----------------------------
upsampled = np.zeros(num_symbols * sps)
upsampled[::sps] = symbols

# -----------------------------
# RRC pulse shaping
# -----------------------------
rrc = rrc_filter(beta, span, sps)

shaped = np.convolve(
    upsampled,
    rrc,
    mode="same"
)

# -----------------------------
# Time axes
# -----------------------------
t_symbols = np.arange(num_symbols)

t_samples = np.arange(len(shaped)) / sps

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 5))

plt.step(
    t_symbols,
    symbols,
    where="post",
    label="Original symbols"
)

plt.plot(
    t_samples,
    shaped,
    label="RRC-shaped waveform"
)

plt.xlim(0, 20)
plt.xlabel("Symbol time")
plt.ylabel("Amplitude")
plt.title("BPSK: Rectangular Symbols vs RRC Pulse Shaping")
plt.legend()
plt.grid()
plt.show()