import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Signal = 50 Hz + 200 Hz
wanted = np.sin(2 * np.pi * 50 * t)
noise = 0.5 * np.sin(2 * np.pi * 200 * t)
x = wanted + noise

# Low-pass filter
cutoff = 100

sos = signal.butter(
    6,
    cutoff,
    btype="lowpass",
    fs=fs,
    output="sos"
)

filtered = signal.sosfiltfilt(sos, x)

# FFT function
def get_spectrum(x):
    X = np.fft.rfft(x)
    magnitude = np.abs(X)
    freq = np.fft.rfftfreq(len(x), 1 / fs)
    return freq, magnitude

freq, original_fft = get_spectrum(x)
_, filtered_fft = get_spectrum(filtered)

# Plot
plt.plot(freq, original_fft, label="Original")
plt.plot(freq, filtered_fft, label="Filtered")

plt.xlim(0, 300)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Before and After Filtering")
plt.legend()
plt.grid()
plt.show()