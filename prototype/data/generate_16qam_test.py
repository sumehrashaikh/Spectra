import numpy as np
from pathlib import Path
from scipy.io import wavfile


# ============================================================
# 16-QAM TEST SIGNAL GENERATOR
# ============================================================

sample_rate = 8000
symbol_rate = 100
samples_per_symbol = sample_rate // symbol_rate

num_symbols = 1000
snr_db = 18.0

rng = np.random.default_rng(123)


# ============================================================
# OUTPUT PATHS
# ============================================================

project_root = Path(__file__).resolve().parents[2]

output_dir = (
    project_root
    / "prototype"
    / "data"
    / "input"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

wav_path = output_dir / "qam16_iq.wav"
reference_path = output_dir / "qam16_iq.reference.npz"


# ============================================================
# GENERATE RANDOM BITS
# ============================================================

num_bits = num_symbols * 4

bits = rng.integers(
    0,
    2,
    size=num_bits,
    dtype=np.uint8
)

bit_groups = bits.reshape(
    num_symbols,
    4
)


# ============================================================
# GRAY-CODED 16-QAM MAPPING
#
# Two bits control I
# Two bits control Q
#
# Gray mapping:
#
# 00 -> -3
# 01 -> -1
# 11 -> +1
# 10 -> +3
# ============================================================

gray_map = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): 1,
    (1, 0): 3,
}


symbols = np.zeros(
    num_symbols,
    dtype=np.complex128
)


for index, group in enumerate(bit_groups):

    i_bits = (
        int(group[0]),
        int(group[1])
    )

    q_bits = (
        int(group[2]),
        int(group[3])
    )

    i_value = gray_map[i_bits]
    q_value = gray_map[q_bits]

    symbols[index] = (
        i_value
        + 1j * q_value
    )


# ============================================================
# NORMALIZE 16-QAM
#
# Average symbol energy of unnormalized
# square 16-QAM is 10.
# ============================================================

symbols = symbols / np.sqrt(10.0)


# ============================================================
# OVERSAMPLE
# ============================================================

baseband = np.repeat(
    symbols,
    samples_per_symbol
)


# ============================================================
# ADD COMPLEX AWGN
# ============================================================

signal_power = np.mean(
    np.abs(baseband) ** 2
)

snr_linear = 10 ** (
    snr_db / 10.0
)

noise_power = (
    signal_power
    / snr_linear
)

noise_std = np.sqrt(
    noise_power / 2.0
)

noise = (
    rng.normal(
        0,
        noise_std,
        len(baseband)
    )
    +
    1j
    * rng.normal(
        0,
        noise_std,
        len(baseband)
    )
)

iq_signal = baseband + noise


# ============================================================
# NORMALIZE FOR 16-BIT WAV
# ============================================================

peak = np.max(
    np.abs(iq_signal)
)

if peak <= 0:
    raise ValueError(
        "Generated signal has zero amplitude."
    )

scale = (
    0.90
    / peak
)

i_data = np.real(
    iq_signal
) * scale

q_data = np.imag(
    iq_signal
) * scale


# ============================================================
# CREATE STEREO WAV
#
# Channel 0 = I
# Channel 1 = Q
# ============================================================

stereo = np.column_stack(
    (
        i_data,
        q_data
    )
)

stereo_int16 = np.int16(
    np.clip(
        stereo,
        -1.0,
        1.0
    )
    * 32767
)


# ============================================================
# SAVE WAV
# ============================================================

wavfile.write(
    wav_path,
    sample_rate,
    stereo_int16
)


# ============================================================
# SAVE REFERENCE INFORMATION
# ============================================================

np.savez(
    reference_path,
    bits=bits,
    symbols=symbols,
    sample_rate=sample_rate,
    symbol_rate=symbol_rate,
    samples_per_symbol=samples_per_symbol,
    snr_db=snr_db
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("16-QAM test generated")
print("---------------------")
print(
    f"Sample rate: "
    f"{sample_rate} Hz"
)

print(
    f"Symbol rate: "
    f"{symbol_rate} symbols/s"
)

print(
    f"Samples/symbol: "
    f"{samples_per_symbol}"
)

print(
    f"Symbols: "
    f"{num_symbols}"
)

print(
    f"Bits: "
    f"{num_bits}"
)

print(
    f"SNR: "
    f"{snr_db:.1f} dB"
)

print(
    f"WAV: "
    f"{wav_path}"
)

print(
    f"Reference: "
    f"{reference_path}"
)