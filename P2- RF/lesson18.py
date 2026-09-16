import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(42)

fs = 1000
duration = 2

t = np.arange(0, duration, 1 / fs)

# Noise
noise = 0.3 * np.random.randn(len(t))

# Signal exists only from 0.8 s to 1.2 s
burst = np.zeros_like(t)

mask = (t >= 0.8) & (t <= 1.2)

burst[mask] = np.sin(2 * np.pi * 50 * t[mask])

# Received signal
received = noise + burst

# Calculate PSD
freq, psd = signal.welch(
    received,
    fs=fs,
    nperseg=512
)

psd_db = 10 * np.log10(psd + 1e-12)

# Estimate noise floor
noise_floor = np.median(psd_db)

# Detection threshold
threshold = noise_floor + 6

print("Estimated noise floor:", noise_floor, "dB")
print("Detection threshold:", threshold, "dB")

# Detect frequencies above threshold
detected = psd_db > threshold

# Plot
plt.plot(freq, psd_db, label="PSD")
plt.axhline(
    noise_floor,
    linestyle="--",
    label="Noise floor"
)
plt.axhline(
    threshold,
    linestyle=":",
    label="Detection threshold"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("Noise Floor and Signal Detection")
plt.xlim(0, 200)
plt.legend()
plt.grid()
plt.show()

# Report detected frequencies
detected_freqs = freq[detected]

if len(detected_freqs) > 0:
    print("Signal detected around:",
          detected_freqs.min(),
          "to",
          detected_freqs.max(),
          "Hz")
else:
    print("No signal detected")