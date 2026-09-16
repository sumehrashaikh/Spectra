import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

np.random.seed(42)

fs = 1000
duration = 2

t = np.arange(0, duration, 1 / fs)

# Noise
noise = 0.25 * np.random.randn(len(t))

# Signal burst
burst = np.zeros_like(t)

mask = (t >= 0.8) & (t <= 1.2)
burst[mask] = np.sin(2 * np.pi * 50 * t[mask])

received = noise + burst

# Short-window power
window_size = 50

power = np.convolve(
    received ** 2,
    np.ones(window_size) / window_size,
    mode="same"
)

# Initial detection
threshold = 0.20
detected = power > threshold

# Close small gaps
structure = np.ones(30)

cleaned = ndimage.binary_closing(
    detected,
    structure=structure
)

# Remove small isolated regions
cleaned = ndimage.binary_opening(
    cleaned,
    structure=structure
)

# Find segments
labels, num_labels = ndimage.label(cleaned)

segments = []

for label in range(1, num_labels + 1):

    indices = np.where(labels == label)[0]

    if len(indices) >= 20:
        segments.append(
            (indices[0], indices[-1])
        )

# Plot
plt.plot(t, power, label="Power")

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

for start, end in segments:

    plt.axvspan(
        t[start],
        t[end],
        alpha=0.3
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Average Power")
plt.title("Cleaned Signal Segmentation")
plt.legend()
plt.grid()
plt.show()

for start, end in segments:
    print(
        f"Segment: "
        f"{t[start]:.3f} s → {t[end]:.3f} s"
    )