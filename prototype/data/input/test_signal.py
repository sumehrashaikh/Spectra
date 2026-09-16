from pathlib import Path
import wave

import numpy as np


# ------------------------------------------------------------
# Output path
# ------------------------------------------------------------

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "test_signal.wav"
)


# ------------------------------------------------------------
# Signal parameters
# ------------------------------------------------------------

fs = 48000
duration = 2.0

t = np.arange(
    int(fs * duration)
) / fs


# ------------------------------------------------------------
# Create two test tones
# ------------------------------------------------------------

x = (
    0.7 * np.sin(2 * np.pi * 3000 * t)
    + 0.3 * np.sin(2 * np.pi * 7000 * t)
)


# ------------------------------------------------------------
# Normalize safely
# ------------------------------------------------------------

peak = np.max(np.abs(x))

if peak == 0:
    raise ValueError("Generated signal has zero amplitude.")

x = x / peak


# ------------------------------------------------------------
# Convert to 16-bit PCM
# ------------------------------------------------------------

x_int16 = np.int16(
    np.clip(x, -1, 1) * 32767
)


# ------------------------------------------------------------
# Write WAV
# ------------------------------------------------------------

with wave.open(
    str(OUTPUT_PATH),
    "wb"
) as wav:

    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(fs)

    wav.writeframes(
        x_int16.tobytes()
    )


print("WAV created successfully.")
print("Path:", OUTPUT_PATH)
print("Sample rate:", fs, "Hz")
print("Duration:", duration, "seconds")