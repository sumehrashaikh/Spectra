import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(42)

fs = 2000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Create several nearby frequency components
x = (
    np.exp(1j * 2 * np.pi * 95 * t)
    + 0.8 * np.exp(1j * 2 * np.pi * 100 * t)
    + np.exp(1j * 2 * np.pi * 105 * t)
)

# Add noise
noise = 0.2 * (
    np.random.randn(len(t))
    + 1j * np.random.randn(len(t))
)

received = x + noise

# FFT
X = np.fft.fftshift(np.fft.fft(received))

freq = np.fft.fftshift(
    np.fft.fftfreq(len(received), 1 / fs)
)

magnitude = np.abs(X)

# Focus on positive frequencies
mask = (freq > 50) & (freq < 150)

freq_plot = freq[mask]
mag_plot = magnitude[mask]

# Threshold based on the strongest peak
threshold = 0.2 * np.max(mag_plot)

detected = mag_plot > threshold

# Find frequency range
detected_freqs = freq_plot[detected]

if len(detected_freqs) > 0:
    start_freq = detected_freqs.min()
    end_freq = detected_freqs.max()

    print(f"Estimated signal band: {start_freq:.1f} Hz → {end_freq:.1f} Hz")
    print(f"Estimated bandwidth: {end_freq - start_freq:.1f} Hz")
else:
    print("No signal detected")

# Plot
plt.plot(freq_plot, mag_plot)

plt.axhline(
    threshold,
    linestyle="--",
    label="Detection threshold"
)

if len(detected_freqs) > 0:
    plt.axvspan(
        start_freq,
        end_freq,
        alpha=0.3,
        label="Detected signal band"
    )

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Signal Bandwidth Detection")
plt.legend()
plt.grid()
plt.show()