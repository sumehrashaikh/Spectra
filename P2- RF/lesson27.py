import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Two complex signals
signal_1 = np.exp(1j * 2 * np.pi * 100 * t)
signal_2 = 0.6 * np.exp(1j * 2 * np.pi * 140 * t)

# Noise
noise = 0.25 * (
    np.random.randn(len(t))
    + 1j * np.random.randn(len(t))
)

# Received IQ
received = signal_1 + signal_2 + noise

# FFT
X = np.fft.fftshift(np.fft.fft(received))

freq = np.fft.fftshift(
    np.fft.fftfreq(len(received), 1 / fs)
)

# Magnitude
magnitude = np.abs(X)

# Plot
plt.plot(freq, magnitude)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Multiple Signals + Noise")

plt.xlim(50, 200)
plt.grid()
plt.show()