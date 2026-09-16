import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(42)

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Two signals
signal_1 = np.exp(1j * 2 * np.pi * 100 * t)
signal_2 = 0.6 * np.exp(1j * 2 * np.pi * 140 * t)

# Noise
noise = 0.25 * (
    np.random.randn(len(t))
    + 1j * np.random.randn(len(t))
)

received = signal_1 + signal_2 + noise

# FFT
X = np.fft.fftshift(np.fft.fft(received))
freq = np.fft.fftshift(
    np.fft.fftfreq(len(received), 1 / fs)
)

magnitude = np.abs(X)

# Find peaks
peaks, properties = signal.find_peaks(
    magnitude,
    prominence=100
)

# Keep only positive frequencies
valid = freq[peaks] > 0

peaks = peaks[valid]

# Plot
plt.plot(freq, magnitude)

plt.scatter(
    freq[peaks],
    magnitude[peaks],
    marker="x",
    label="Detected peaks"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Multiple Signal Peak Detection")
plt.xlim(50, 200)
plt.legend()
plt.grid()
plt.show()

print("Detected frequencies:")

for peak in peaks:
    print(f"{freq[peak]:.1f} Hz")