import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 0.1

t = np.arange(0, duration, 1 / fs)

# Signal with large amplitude
x = 7 * np.sin(2 * np.pi * 50 * t)

# Normalize to [-1, 1]
x_norm = x / np.max(np.abs(x))

plt.plot(t, x, label="Original")
plt.plot(t, x_norm, label="Normalized")

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Signal Normalization")
plt.legend()
plt.grid()
plt.show()

print("Original max:", np.max(np.abs(x)))
print("Normalized max:", np.max(np.abs(x_norm)))