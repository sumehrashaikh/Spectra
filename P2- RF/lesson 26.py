import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# True received frequency
f_received = 110

# Complex IQ signal
iq = np.exp(1j * 2 * np.pi * f_received * t)

# FFT
X = np.fft.fftshift(np.fft.fft(iq))
freq = np.fft.fftshift(
    np.fft.fftfreq(len(iq), 1 / fs)
)

# Find strongest frequency
peak_index = np.argmax(np.abs(X))
estimated_frequency = freq[peak_index]

print("True frequency:", f_received, "Hz")
print("Estimated frequency:", estimated_frequency, "Hz")

# Plot
plt.plot(freq, np.abs(X))

plt.axvline(
    estimated_frequency,
    linestyle="--",
    label=f"Estimated = {estimated_frequency:.1f} Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Automatic Frequency Estimation")

plt.xlim(50, 150)
plt.legend()
plt.grid()
plt.show()