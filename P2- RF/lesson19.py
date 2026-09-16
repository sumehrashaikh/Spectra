import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 1000
duration = 2

t = np.arange(0, duration, 1 / fs)

# Noise
noise = 0.3 * np.random.randn(len(t))

# Signal burst from 0.8 to 1.2 seconds
burst = np.zeros_like(t)

mask = (t >= 0.8) & (t <= 1.2)

burst[mask] = np.sin(2 * np.pi * 50 * t[mask])

# Received signal
received = noise + burst

# Calculate short-window power
window_size = 100

power = np.convolve(
    received ** 2,
    np.ones(window_size) / window_size,
    mode="same"
)

# Threshold
threshold = 0.25

detected = power > threshold

# Plot
plt.plot(t, power, label="Estimated Power")
plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Average Power")
plt.title("Time-Domain Energy Detection")
plt.legend()
plt.grid()
plt.show()

# Detection time range
detected_indices = np.where(detected)[0]

if len(detected_indices) > 0:
    start = t[detected_indices[0]]
    end = t[detected_indices[-1]]

    print(f"Detected signal from {start:.3f} s to {end:.3f} s")
else:
    print("No signal detected")