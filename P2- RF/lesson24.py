import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Expected frequency
f_expected = 100

# Actual received frequency
f_received = 110

# Received signal
x = np.exp(1j * 2 * np.pi * f_received * t)

# FFT
X = np.fft.fftshift(np.fft.fft(x))
freq = np.fft.fftshift(
    np.fft.fftfreq(len(x), 1 / fs)
)

# Plot spectrum
plt.plot(freq, np.abs(X))

plt.axvline(
    f_expected,
    linestyle="--",
    label="Expected = 100 Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Offset")
plt.xlim(50, 150)
plt.legend()
plt.grid()
plt.show()

print("Expected frequency:", f_expected, "Hz")
print("Received frequency:", f_received, "Hz")
print(
    "Frequency offset:",
    f_received - f_expected,
    "Hz"
)