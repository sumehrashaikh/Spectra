from pathlib import Path
import wave

import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

FS = 8000

SYMBOL_RATE = 100

SPS = FS // SYMBOL_RATE

NUM_SYMBOLS = 1000

SNR_DB = 15

OUTPUT = (
    Path(__file__).resolve().parent
    / "bpsk_iq.wav"
)

REFERENCE_OUTPUT = OUTPUT.with_name(
    f"{OUTPUT.stem}.reference.npz"
)


# ============================================================
# RANDOM BITS
# ============================================================

rng = np.random.default_rng(42)

bits = rng.integers(
    0,
    2,
    NUM_SYMBOLS
)

symbols = (
    1 - 2 * bits
).astype(float)


# ============================================================
# UPSAMPLE
# ============================================================

upsampled = np.repeat(
    symbols,
    SPS
)


# ============================================================
# SIMPLE RAISED-COSINE-LIKE SMOOTHING
# ============================================================

span = 8

x = np.linspace(
    -span / 2,
    span / 2,
    span * SPS + 1
)

alpha = 0.35

# Simple windowed-sinc pulse
pulse = np.sinc(
    x
)

pulse *= np.hanning(
    len(pulse)
)

pulse /= np.sum(
    pulse
)


shaped = np.convolve(
    upsampled,
    pulse,
    mode="same"
)


# ============================================================
# COMPLEX BASEBAND IQ
# ============================================================

iq = shaped.astype(
    np.complex128
)


# ============================================================
# ADD COMPLEX AWGN
# ============================================================

signal_power = np.mean(
    np.abs(iq) ** 2
)

snr_linear = 10 ** (
    SNR_DB / 10
)

noise_power = (
    signal_power
    / snr_linear
)

noise = np.sqrt(
    noise_power / 2
) * (
    rng.standard_normal(len(iq))
    + 1j
    * rng.standard_normal(len(iq))
)

iq_noisy = iq + noise


# ============================================================
# NORMALIZE
# ============================================================

scale = np.max(
    np.abs(iq_noisy)
)

iq_noisy /= (
    scale + 1e-12
)


# ============================================================
# WRITE STEREO WAV
# ============================================================

i_int16 = np.int16(
    np.clip(
        iq_noisy.real,
        -1,
        1
    ) * 32767
)

q_int16 = np.int16(
    np.clip(
        iq_noisy.imag,
        -1,
        1
    ) * 32767
)

stereo = np.column_stack(
    (
        i_int16,
        q_int16
    )
)

with wave.open(
    str(OUTPUT),
    "wb"
) as wav:

    wav.setnchannels(2)

    wav.setsampwidth(2)

    wav.setframerate(FS)

    wav.writeframes(
        stereo.tobytes()
    )

# Keep the exact source bits beside the synthetic WAV.  This makes BER tests
# reproducible without embedding test-only knowledge in the receiver.
np.savez_compressed(
    REFERENCE_OUTPUT,
    bits=bits.astype(np.uint8),
    sample_rate=np.array(FS),
    symbol_rate=np.array(SYMBOL_RATE),
    samples_per_symbol=np.array(SPS),
    snr_db=np.array(SNR_DB),
    seed=np.array(42),
)


print("BPSK IQ WAV created.")
print("----------------------")
print("File:", OUTPUT)
print("Sample rate:", FS, "Hz")
print("Symbol rate:", SYMBOL_RATE, "symbols/s")
print("Samples/symbol:", SPS)
print("SNR:", SNR_DB, "dB")
print("Symbols:", NUM_SYMBOLS)
print("Bit reference:", REFERENCE_OUTPUT)
