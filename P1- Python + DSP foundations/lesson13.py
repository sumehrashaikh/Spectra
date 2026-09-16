import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
duration = 2

t = np.arange(0, duration, 1 / fs)

# Desired signal
x = np.sin(2 * np.pi * 50 * t)

# Add random noise
noise = 0.5 * np.random.randn(len(t))

received = x + noise

# Power Spectral Density using Welch's method
freq, psd = signal.welch(
    received,
    fs=fs,
    nperseg=1024
)

# Convert power to dB
psd_db = 10 * np.log10(psd)

# Plot
plt.plot(freq, psd_db)

plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("Power Spectral Density")
plt.grid()
plt.show()