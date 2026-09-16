from pathlib import Path

import numpy as np
from scipy.io import wavfile


# -----------------------------
# BFSK test-signal parameters
# -----------------------------
SAMPLE_RATE = 8000
SYMBOL_RATE = 100
SAMPLES_PER_SYMBOL = SAMPLE_RATE // SYMBOL_RATE
NUM_SYMBOLS = 1000

FREQ_0 = 500.0
FREQ_1 = 700.0

SEED = 123


def generate_bfsk():
    rng = np.random.default_rng(SEED)

    # Random binary data
    bits = rng.integers(0, 2, NUM_SYMBOLS)

    # Frequency selected for each symbol
    frequencies = np.where(bits == 0, FREQ_0, FREQ_1)

    # Generate continuous-phase BFSK
    phase = np.zeros(NUM_SYMBOLS * SAMPLES_PER_SYMBOL)

    current_phase = 0.0
    index = 0

    for frequency in frequencies:
        t = np.arange(SAMPLES_PER_SYMBOL) / SAMPLE_RATE

        symbol_phase = current_phase + 2 * np.pi * frequency * t
        phase[index:index + SAMPLES_PER_SYMBOL] = symbol_phase

        current_phase = symbol_phase[-1] + 2 * np.pi * frequency / SAMPLE_RATE
        index += SAMPLES_PER_SYMBOL

    iq = np.exp(1j * phase)

    return iq, bits


def save_bfsk():
    iq, bits = generate_bfsk()

    output_dir = Path(__file__).resolve().parent / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to int16 WAV range
    i = np.real(iq)
    q = np.imag(iq)

    stereo = np.column_stack((i, q))
    stereo_int16 = np.int16(stereo * 32767)

    wav_path = output_dir / "bfsk_iq.wav"
    reference_path = output_dir / "bfsk_iq.reference.npz"

    wavfile.write(wav_path, SAMPLE_RATE, stereo_int16)

    np.savez(
        reference_path,
        bits=bits,
        sample_rate=SAMPLE_RATE,
        symbol_rate=SYMBOL_RATE,
        freq_0=FREQ_0,
        freq_1=FREQ_1,
    )

    print(f"Saved: {wav_path}")
    print(f"Saved: {reference_path}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Symbol rate: {SYMBOL_RATE} symbols/s")
    print(f"Samples/symbol: {SAMPLES_PER_SYMBOL}")
    print(f"Symbols: {NUM_SYMBOLS}")
    print(f"Frequencies: {FREQ_0} Hz / {FREQ_1} Hz")


if __name__ == "__main__":
    save_bfsk()