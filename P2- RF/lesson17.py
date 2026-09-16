import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 1000
duration = 1

t = np.arange(0, duration, 1 / fs)

# Clean signal
signal_clean = np.sin(2 * np.pi * 50 * t)

# Noise
noise = np.random.randn(len(t))

# Choose desired SNR
snr_db = 10

# Convert SNR from dB to linear
snr_linear = 10 ** (snr_db / 10)

# Signal power
signal_power = np.mean(signal_clean ** 2)

# Required noise power
noise_power = signal_power / snr_linear

# Scale noise to desired power
noise_scaled = noise * np.sqrt(noise_power / np.mean(noise ** 2))

# Received signal
received = signal_clean + noise_scaled

# Check actual SNR
measured_snr = (
    np.mean(signal_clean ** 2)
    / np.mean(noise_scaled ** 2)
)

measured_snr_db = 10 * np.log10(measured_snr)

print("Target SNR:", snr_db, "dB")
print("Measured SNR:", measured_snr_db, "dB")

# Plot
plt.plot(t, received)

plt.xlim(0, 0.15)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"Received Signal — SNR = {measured_snr_db:.2f} dB")
plt.grid()
plt.show()