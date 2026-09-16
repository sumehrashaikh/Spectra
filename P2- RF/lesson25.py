import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Frequencies
f_expected = 100
f_received = 110

# Received IQ signal
iq = np.exp(1j * 2 * np.pi * f_received * t)

# Known frequency offset
offset = f_received - f_expected

# Frequency correction
correction = np.exp(-1j * 2 * np.pi * offset * t)

iq_corrected = iq * correction

# FFT of original and corrected signals
X_original = np.fft.fftshift(np.fft.fft(iq))
X_corrected = np.fft.fftshift(np.fft.fft(iq_corrected))

freq = np.fft.fftshift(
    np.fft.fftfreq(len(iq), 1 / fs)
)

# Plot
plt.plot(
    freq,
    np.abs(X_original),
    label="Before correction"
)

plt.plot(
    freq,
    np.abs(X_corrected),
    label="After correction"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Offset Correction")

plt.xlim(50, 150)
plt.legend()
plt.grid()
plt.show()