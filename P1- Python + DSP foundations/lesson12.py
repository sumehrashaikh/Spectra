import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
cutoff = 100

# Design filter
sos = signal.butter(
    6,
    cutoff,
    btype="lowpass",
    fs=fs,
    output="sos"
)

# Calculate frequency response
freq, response = signal.sosfreqz(
    sos,
    worN=2000,
    fs=fs
)

# Magnitude in dB
magnitude_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))

# Plot
plt.plot(freq, magnitude_db)

plt.axvline(cutoff, linestyle="--", label="Cutoff = 100 Hz")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Low-Pass Filter Frequency Response")
plt.xlim(0, 300)
plt.ylim(-100, 5)
plt.legend()
plt.grid()
plt.show()