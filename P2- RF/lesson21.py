import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create noisy power data
N = 1000

noise = np.random.exponential(scale=1.0, size=N)

power = noise.copy()

# Add two signals
power[300] += 8
power[700] += 5

# CFAR parameters
training_cells = 20
guard_cells = 4
threshold_factor = 4

detections = np.zeros(N, dtype=bool)
thresholds = np.full(N, np.nan)

for i in range(training_cells + guard_cells,
               N - training_cells - guard_cells):

    # Cells on the left and right
    left = power[
        i - guard_cells - training_cells:
        i - guard_cells
    ]

    right = power[
        i + guard_cells + 1:
        i + guard_cells + training_cells + 1
    ]

    # Estimate local noise
    noise_estimate = np.mean(
        np.concatenate((left, right))
    )

    # Adaptive threshold
    threshold = threshold_factor * noise_estimate

    thresholds[i] = threshold

    # Test current cell
    if power[i] > threshold:
        detections[i] = True

# Plot
plt.plot(power, label="Power")
plt.plot(thresholds, label="CFAR Threshold")

plt.scatter(
    np.where(detections),
    power[detections],
    marker="x",
    label="Detection"
)

plt.xlabel("Cell")
plt.ylabel("Power")
plt.title("Basic CA-CFAR Detection")
plt.legend()
plt.grid()
plt.show()

print("Detected cells:")
print(np.where(detections)[0])