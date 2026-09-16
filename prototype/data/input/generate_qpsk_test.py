from pathlib import Path
import wave

import numpy as np


def save_wav_iq(path, i, q, sample_rate):
    """
    Save complex IQ as a stereo 16-bit WAV.
    Channel 0 = I
    Channel 1 = Q
    """

    i = np.asarray(i, dtype=float)
    q = np.asarray(q, dtype=float)

    peak = max(
        np.max(np.abs(i)),
        np.max(np.abs(q)),
        1e-12
    )

    scale = 0.85 / peak

    i = np.clip(i * scale, -1.0, 1.0)
    q = np.clip(q * scale, -1.0, 1.0)

    interleaved = np.empty(
        i.size * 2,
        dtype=np.int16
    )

    interleaved[0::2] = (
        i * 32767
    ).astype(np.int16)

    interleaved[1::2] = (
        q * 32767
    ).astype(np.int16)

    with wave.open(str(path), "wb") as wav:

        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(
            interleaved.tobytes()
        )


def main():

    rng = np.random.default_rng(42)

    # --------------------------------------------------
    # Signal parameters
    # --------------------------------------------------

    sample_rate = 8000
    symbol_rate = 100

    samples_per_symbol = (
        sample_rate // symbol_rate
    )

    num_symbols = 1000

    snr_db = 15.0

    # --------------------------------------------------
    # Generate random bits
    # --------------------------------------------------

    bits = rng.integers(
        0,
        2,
        size=num_symbols * 2,
        dtype=np.uint8
    )

    bit_pairs = bits.reshape(
        -1,
        2
    )

    # --------------------------------------------------
    # Gray-coded QPSK mapping
    #
    # 00 -> +1 + j
    # 01 -> -1 + j
    # 11 -> -1 - j
    # 10 -> +1 - j
    # --------------------------------------------------

    symbols = np.zeros(
        num_symbols,
        dtype=np.complex128
    )

    for index, pair in enumerate(bit_pairs):

        b0 = pair[0]
        b1 = pair[1]

        if b0 == 0 and b1 == 0:
            symbols[index] = 1 + 1j

        elif b0 == 0 and b1 == 1:
            symbols[index] = -1 + 1j

        elif b0 == 1 and b1 == 1:
            symbols[index] = -1 - 1j

        else:
            symbols[index] = 1 - 1j

    # Normalize symbol energy
    symbols /= np.sqrt(2)

    # --------------------------------------------------
    # Oversample
    # --------------------------------------------------

    baseband = np.repeat(
        symbols,
        samples_per_symbol
    )

    # --------------------------------------------------
    # Add AWGN
    # --------------------------------------------------

    signal_power = np.mean(
        np.abs(baseband) ** 2
    )

    snr_linear = 10 ** (
        snr_db / 10.0
    )

    noise_power = (
        signal_power / snr_linear
    )

    noise = (
        np.sqrt(noise_power / 2)
        *
        (
            rng.standard_normal(
                len(baseband)
            )
            +
            1j
            *
            rng.standard_normal(
                len(baseband)
            )
        )
    )

    received = baseband + noise

    # --------------------------------------------------
    # Save files
    # --------------------------------------------------

    output_dir = Path(
        __file__
    ).resolve().parent / "input"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    wav_path = (
        output_dir /
        "qpsk_iq.wav"
    )

    reference_path = (
        output_dir /
        "qpsk_iq.reference.npz"
    )

    save_wav_iq(
        wav_path,
        received.real,
        received.imag,
        sample_rate
    )

    np.savez(
        reference_path,
        bits=bits,
        symbols=symbols,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        samples_per_symbol=samples_per_symbol,
        snr_db=snr_db
    )

    print("QPSK test generated")
    print("--------------------")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Symbol rate: {symbol_rate} symbols/s")
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
        f"{len(bits)}"
    )
    print(
        f"SNR: "
        f"{snr_db:.1f} dB"
    )
    print(f"WAV: {wav_path}")
    print(
        f"Reference: "
        f"{reference_path}"
    )


if __name__ == "__main__":
    main()