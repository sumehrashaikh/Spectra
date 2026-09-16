import numpy as np
import matplotlib.pyplot as plt

# Sampling settings
fs = 1000
duration = 1

# Time samples
t = np.arange(0, duration, 1 / fs)

# Two-frequency signal
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# FFT
X = np.fft.fft(x)

# Frequency axis
freq = np.fft.fftfreq(len(x), 1 / fs)

# Keep only positive frequencies
positive = freq >= 0

# Plot magnitude spectrum
plt.plot(freq[positive], np.abs(X[positive]))

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of 50 Hz + 120 Hz Signal")
plt.grid()
plt.show()