import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 1000
duration = 2

t = np.arange(0, duration, 1 / fs)

# Noise
noise = 0.3 * np.random.randn(len(t))

# Signal burst
burst = np.zeros_like(t)

mask = (t >= 0.8) & (t <= 1.2)

burst[mask] = np.sin(2 * np.pi * 50 * t[mask])

# Received signal
received = noise + burst

# Sliding-window power
window_size = 100

power = np.convolve(
    received ** 2,
    np.ones(window_size) / window_size,
    mode="same"
)

# Estimate baseline noise power
noise_region = t < 0.5

noise_power = np.median(power[noise_region])

# Adaptive threshold
threshold = 3 * noise_power

# Detect
detected = power > threshold

print("Estimated noise power:", noise_power)
print("Adaptive threshold:", threshold)

# Plot
plt.plot(t, power, label="Estimated Power")

plt.axhline(
    threshold,
    linestyle="--",
    label="Adaptive Threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Average Power")
plt.title("Adaptive Energy Detection")
plt.legend()
plt.grid()
plt.show()

# Detection range
indices = np.where(detected)[0]

if len(indices) > 0:

    start = t[indices[0]]
    end = t[indices[-1]]

    print(f"Detected signal from {start:.3f} s to {end:.3f} s")

else:

    print("No signal detected")