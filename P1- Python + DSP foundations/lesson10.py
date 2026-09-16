import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Complex IQ signal
iq = np.exp(1j * 2 * np.pi * 50 * t) + 0.5 * np.exp(1j * 2 * np.pi * 120 * t)

# FFT
X = np.fft.fft(iq)

# Shift zero frequency to the center
X_shifted = np.fft.fftshift(X)

# Frequency axis
freq = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))

# Plot
plt.plot(freq, np.abs(X_shifted))

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Complex IQ Signal")
plt.grid()
plt.show()