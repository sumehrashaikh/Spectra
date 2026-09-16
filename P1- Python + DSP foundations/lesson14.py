import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
duration = 2

t = np.arange(0, duration, 1 / fs)

# Frequency changes halfway through
x = np.where(
    t < 1,
    np.sin(2 * np.pi * 50 * t),
    np.sin(2 * np.pi * 120 * t)
)

# Spectrogram
freq, time, Sxx = signal.spectrogram(
    x,
    fs=fs,
    nperseg=256,
    noverlap=128
)

# Convert power to dB
Sxx_db = 10 * np.log10(Sxx + 1e-12)

# Plot
plt.pcolormesh(
    time,
    freq,
    Sxx_db,
    shading="gouraud"
)

plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (seconds)")
plt.title("Spectrogram")
plt.colorbar(label="Power (dB)")
plt.ylim(0, 250)
plt.show()