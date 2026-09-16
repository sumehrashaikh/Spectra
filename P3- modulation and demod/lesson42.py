import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

np.random.seed(42)

# -----------------------------
# Signal parameters
# -----------------------------
fs = 1000
symbol_rate = 100
sps = fs // symbol_rate
num_symbols = 500

# -----------------------------
# Generate BPSK
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
symbols = 1 - 2 * bits

x = np.repeat(symbols, sps)

# -----------------------------
# Add noise
# -----------------------------
snr_db = 10

signal_power = np.mean(x ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))

noise = np.sqrt(noise_power) * np.random.randn(len(x))

received = x + noise

# -----------------------------
# Detect transitions
# -----------------------------
difference = np.abs(np.diff(received))

# Remove average level
difference = difference - np.mean(difference)

# -----------------------------
# Autocorrelation
# -----------------------------
autocorr = np.correlate(
    difference,
    difference,
    mode="full"
)

autocorr = autocorr[len(difference) - 1:]

# Ignore zero-lag peak
search = autocorr[1:300]

# Find peaks
peaks, properties = find_peaks(
    search,
    distance=5,
    prominence=np.std(search)
)

peaks = peaks + 1

# -----------------------------
# Estimate samples/symbol
# -----------------------------
if len(peaks) > 0:

    # First strong autocorrelation peak
    estimated_sps = peaks[0]

    estimated_symbol_rate = fs / estimated_sps

    print(
        "Actual samples/symbol:",
        sps
    )

    print(
        "Estimated samples/symbol:",
        estimated_sps
    )

    print(
        "Actual symbol rate:",
        symbol_rate,
        "symbols/s"
    )

    print(
        "Estimated symbol rate:",
        estimated_symbol_rate,
        "symbols/s"
    )

else:
    print("Could not estimate symbol rate.")

# -----------------------------
# Plot autocorrelation
# -----------------------------
plt.plot(
    np.arange(len(search)),
    search
)

if len(peaks) > 0:
    plt.scatter(
        peaks,
        search[peaks - 1],
        marker="x",
        label="Detected peaks"
    )

plt.xlabel("Lag (samples)")
plt.ylabel("Autocorrelation")
plt.title("Symbol-Rate Estimation from Transition Spacing")
plt.xlim(0, 150)
plt.legend()
plt.grid()
plt.show()