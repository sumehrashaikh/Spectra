import numpy as np
import matplotlib.pyplot as plt

fs = 1000          # Sampling frequency: 1000 samples/sec
f = 50             # Signal frequency: 50 Hz
duration = 1       # 1 second

t = np.arange(0, duration, 1/fs)

x = np.sin(2 * np.pi * f * t)

plt.plot(t, x)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("50 Hz Sine Wave")
plt.grid()
plt.show()