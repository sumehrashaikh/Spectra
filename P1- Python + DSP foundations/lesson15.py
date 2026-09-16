import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Original sampling rate
fs_old = 1000

# New sampling rate
fs_new = 500

duration = 1

t = np.arange(0, duration, 1 / fs_old)

# Signal
x = (
    np.sin(2 * np.pi * 50 * t)
    + 0.4 * np.sin(2 * np.pi * 120 * t)
)

# Resample
num_samples = int(len(x) * fs_new / fs_old)

x_resampled = signal.resample(x, num_samples)

# New time axis
t_new = np.arange(len(x_resampled)) / fs_new

# Plot
plt.plot(t, x, label="Original")
plt.plot(t_new, x_resampled, "o-", markersize=2, label="Resampled")

plt.xlim(0, 0.1)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Resampling: 1000 Hz → 500 Hz")
plt.legend()
plt.grid()
plt.show()