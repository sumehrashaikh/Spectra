import numpy as np

from prototype.core.timing import (
    recover_symbol_timing,
    sample_symbols,
)

from prototype.core.bpsk import (
    demodulate_bpsk,
)

from prototype.modulation.demodulator import (
    demodulate_bfsk,
    demodulate_qpsk,
    qpsk_decision,
    demodulate_qam16,
    qam16_decision,
)
from prototype.modulation.classifier import (
    classify_modulation,
    classify_from_constellation,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

SAMPLE_RATE = 8000
SYMBOL_RATE = 100
SAMPLES_PER_SYMBOL = 80

NUM_SYMBOLS = 1000

SNR_VALUES = [
    20,
    15,
    10,
    5,
    0,
]

RNG = np.random.default_rng(2026)


# ============================================================
# BPSK GENERATOR
# ============================================================

def generate_bpsk():

    bits = RNG.integers(
        0,
        2,
        NUM_SYMBOLS,
        dtype=np.uint8
    )

    symbols = (
        2 * bits.astype(float)
        - 1
    ).astype(np.complex128)

    baseband = np.repeat(
        symbols,
        SAMPLES_PER_SYMBOL
    )

    return baseband, bits


# ============================================================
# QPSK GENERATOR
# ============================================================

def generate_qpsk():

    bits = RNG.integers(
        0,
        2,
        NUM_SYMBOLS * 2,
        dtype=np.uint8
    )

    groups = bits.reshape(
        NUM_SYMBOLS,
        2
    )

    symbols = np.zeros(
        NUM_SYMBOLS,
        dtype=np.complex128
    )

    for i, group in enumerate(groups):

        b0 = int(group[0])
        b1 = int(group[1])

        if (b0, b1) == (0, 0):
            symbols[i] = 1 + 1j

        elif (b0, b1) == (0, 1):
            symbols[i] = -1 + 1j

        elif (b0, b1) == (1, 1):
            symbols[i] = -1 - 1j

        else:
            symbols[i] = 1 - 1j

    symbols = (
        symbols
        / np.sqrt(2.0)
    )

    baseband = np.repeat(
        symbols,
        SAMPLES_PER_SYMBOL
    )

    return baseband, bits


# ============================================================
# 16-QAM GENERATOR
# ============================================================

def generate_16qam():

    bits = RNG.integers(
        0,
        2,
        NUM_SYMBOLS * 4,
        dtype=np.uint8
    )

    groups = bits.reshape(
        NUM_SYMBOLS,
        4
    )

    gray_map = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 1): 1,
        (1, 0): 3,
    }

    symbols = np.zeros(
        NUM_SYMBOLS,
        dtype=np.complex128
    )

    for i, group in enumerate(groups):

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

        symbols[i] = (
            i_value
            + 1j * q_value
        )

    symbols = (
        symbols
        / np.sqrt(10.0)
    )

    baseband = np.repeat(
        symbols,
        SAMPLES_PER_SYMBOL
    )

    return baseband, bits


# ============================================================
# BFSK GENERATOR
# ============================================================

def generate_bfsk():

    bits = RNG.integers(
        0,
        2,
        NUM_SYMBOLS,
        dtype=np.uint8,
    )

    freq_0 = 500.0
    freq_1 = 700.0

    frequencies = np.where(
        bits == 0,
        freq_0,
        freq_1,
    )

    total_samples = (
        NUM_SYMBOLS * SAMPLES_PER_SYMBOL
    )

    signal = np.zeros(
        total_samples,
        dtype=np.complex128,
    )

    phase = 0.0

    for symbol_index, frequency in enumerate(frequencies):

        start = (
            symbol_index
            * SAMPLES_PER_SYMBOL
        )

        end = start + SAMPLES_PER_SYMBOL

        t = (
            np.arange(SAMPLES_PER_SYMBOL)
            / SAMPLE_RATE
        )

        symbol_phase = (
            phase
            + 2
            * np.pi
            * frequency
            * t
        )

        signal[start:end] = np.exp(
            1j * symbol_phase
        )

        phase = (
            symbol_phase[-1]
            + 2
            * np.pi
            * frequency
            / SAMPLE_RATE
        )

    return signal, bits

# ============================================================
# ADD AWGN
# ============================================================

def add_awgn(signal, snr_db):

    signal_power = np.mean(
        np.abs(signal) ** 2
    )

    snr_linear = (
        10 ** (snr_db / 10.0)
    )

    noise_power = (
        signal_power
        / snr_linear
    )

    noise_std = np.sqrt(
        noise_power / 2.0
    )

    noise = (
        RNG.normal(
            0,
            noise_std,
            len(signal)
        )
        +
        1j
        * RNG.normal(
            0,
            noise_std,
            len(signal)
        )
    )

    return signal + noise


# ============================================================
# BER HELPER
# ============================================================

def calculate_direct_ber(
    recovered,
    reference
):

    count = min(
        len(recovered),
        len(reference)
    )

    if count == 0:
        return None

    recovered = recovered[:count]
    reference = reference[:count]

    errors = int(
        np.sum(
            recovered != reference
        )
    )

    return {
        "errors": errors,
        "count": count,
        "ber": errors / count,
    }


# ============================================================
# QPSK BER
# ============================================================

def calculate_qpsk_ber(
    symbols,
    reference_bits
):

    best = None

    for rotation_index in range(4):

        rotation = (
            rotation_index
            * np.pi
            / 2.0
        )

        rotated = (
            symbols
            * np.exp(
                -1j * rotation
            )
        )

        bits, _, _ = qpsk_decision(
            rotated
        )

        result = calculate_direct_ber(
            bits,
            reference_bits
        )

        if result is None:
            continue

        result["rotation"] = (
            rotation_index * 90
        )

        if (
            best is None
            or result["errors"]
            < best["errors"]
        ):
            best = result

    return best


# ============================================================
# 16-QAM BER
# ============================================================

def calculate_16qam_ber(
    symbols,
    reference_bits
):

    best = None

    for rotation_index in range(4):

        rotation = (
            rotation_index
            * np.pi
            / 2.0
        )

        rotated = (
            symbols
            * np.exp(
                -1j * rotation
            )
        )

        bits, _, _ = qam16_decision(
            rotated
        )

        result = calculate_direct_ber(
            bits,
            reference_bits
        )

        if result is None:
            continue

        result["rotation"] = (
            rotation_index * 90
        )

        if (
            best is None
            or result["errors"]
            < best["errors"]
        ):
            best = result

    return best


# ============================================================
# RUN ONE SIGNAL
# ============================================================

def run_test(
    modulation,
    clean_signal,
    reference_bits,
    snr_db
):

    noisy_signal = add_awgn(clean_signal, snr_db)

    detected_modulation = "Unknown"
    features = {}

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    try:
        if modulation == "BFSK":
            sps = SAMPLES_PER_SYMBOL
            symbol_rate = SYMBOL_RATE
            confidence = 1.0
            offset = 0
            symbol_samples = noisy_signal
        else:
            timing = recover_symbol_timing(
                noisy_signal,
                SAMPLE_RATE
            )
            sps = timing.estimated_sps
            symbol_rate = timing.symbol_rate
            confidence = timing.confidence
            offset = timing.timing_offset

            symbol_samples = sample_symbols(
                noisy_signal,
                timing
            )

    except Exception:
        return {
            "modulation": modulation,
            "snr": snr_db,
            "detected": detected_modulation,
            "classification_ok": False,
            "sps": None,
            "symbol_rate": None,
            "confidence": None,
            "ber": None,
            "errors": None,
            "bits": 0,
            "status": "TIMING FAILED",
        }

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    try:
        if modulation == "BFSK":
            detected_modulation, features = classify_modulation(
                noisy_signal,
                SAMPLE_RATE
            )
        else:
            detected_modulation, features = (
                classify_from_constellation(symbol_samples)
            )
    except Exception:
        detected_modulation = "Unknown"
        features = {}

    # --------------------------------------------------------
    # Demodulation / BER
    # --------------------------------------------------------

    if modulation == "BFSK":
        try:
            demod = demodulate_bfsk(
                noisy_signal,
                SAMPLE_RATE,
                sps,
                500.0,
                700.0,
            )

            ber_result = calculate_direct_ber(
                demod["bits"],
                reference_bits
            )

            if ber_result is None:
                raise ValueError("BFSK BER comparison returned no result")

        except Exception:
            return {
                "modulation": modulation,
                "snr": snr_db,
                "detected": detected_modulation,
                "classification_ok":
                    detected_modulation == modulation,
                "sps": sps,
                "symbol_rate": symbol_rate,
                "confidence": confidence,
                "ber": None,
                "errors": None,
                "bits": 0,
                "status": "DEMOD FAILED",
            }

    elif modulation == "BPSK":
        try:
            demod = demodulate_bpsk(symbol_samples)

            ber_result = calculate_bpsk_ber(
                demod.bits,
                reference_bits
            )

            if ber_result is None:
                raise ValueError("BPSK BER comparison returned no result")

        except Exception:
            return {
                "modulation": modulation,
                "snr": snr_db,
                "detected": detected_modulation,
                "classification_ok":
                    detected_modulation == modulation,
                "sps": sps,
                "symbol_rate": symbol_rate,
                "confidence": confidence,
                "ber": None,
                "errors": None,
                "bits": 0,
                "status": "DEMOD FAILED",
            }

    elif modulation == "QPSK":
        try:
            demod = demodulate_qpsk(
                noisy_signal,
                sps,
                offset
            )

            ber_result = calculate_qpsk_ber(
                demod["corrected_symbols"],
                reference_bits
            )

            if ber_result is None:
                raise ValueError("QPSK BER comparison returned no result")

        except Exception:
            return {
                "modulation": modulation,
                "snr": snr_db,
                "detected": detected_modulation,
                "classification_ok":
                    detected_modulation == modulation,
                "sps": sps,
                "symbol_rate": symbol_rate,
                "confidence": confidence,
                "ber": None,
                "errors": None,
                "bits": 0,
                "status": "DEMOD FAILED",
            }

    elif modulation == "16-QAM":
        try:
            demod = demodulate_qam16(
                noisy_signal,
                sps,
                offset
            )

            ber_result = calculate_16qam_ber(
                demod["symbols"],
                reference_bits
            )

            if ber_result is None:
                raise ValueError("16-QAM BER comparison returned no result")

        except Exception:
            return {
                "modulation": modulation,
                "snr": snr_db,
                "detected": detected_modulation,
                "classification_ok":
                    detected_modulation == modulation,
                "sps": sps,
                "symbol_rate": symbol_rate,
                "confidence": confidence,
                "ber": None,
                "errors": None,
                "bits": 0,
                "status": "DEMOD FAILED",
            }

    else:
        return {
            "modulation": modulation,
            "snr": snr_db,
            "detected": detected_modulation,
            "classification_ok": False,
            "sps": sps,
            "symbol_rate": symbol_rate,
            "confidence": confidence,
            "ber": None,
            "errors": None,
            "bits": 0,
            "status": "UNKNOWN MODULATION",
        }

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "modulation": modulation,
        "snr": snr_db,
        "detected": detected_modulation,
        "classification_ok":
            detected_modulation == modulation,
        "sps": sps,
        "symbol_rate": symbol_rate,
        "confidence": confidence,
        "ber": ber_result["ber"],
        "errors": ber_result["errors"],
        "bits": ber_result["count"],
        "status": "OK",
    }


# MAIN BENCHMARK
# ============================================================

def main():

    print()
    print("=" * 75)
    print("SIH SIGNAL ANALYZER - SNR ROBUSTNESS TEST")
    print("=" * 75)

    signals = {
        "BPSK": generate_bpsk(),
        "QPSK": generate_qpsk(),
        "16-QAM": generate_16qam(),
        "BFSK": generate_bfsk(),
    }

    results = []

    for modulation, (
        clean_signal,
        reference_bits
    ) in signals.items():

        print()
        print("-" * 75)
        print(modulation)
        print("-" * 75)

        for snr_db in SNR_VALUES:

            result = run_test(
                modulation,
                clean_signal,
                reference_bits,
                snr_db
            )

            results.append(result)

            confidence = result["confidence"]

            if confidence is not None:
                confidence_text = (
                    f"{confidence * 100:.1f}%"
                )
            else:
                confidence_text = "N/A"

            if result["ber"] is not None:
                ber_text = (
                    f"{result['ber']:.6g}"
                )
            else:
                ber_text = "N/A"

            print(
                f"SNR={snr_db:>2} dB | "
                f"Detected={result['detected']:<7} | "
                f"SPS="
                f"{result['sps'] if result['sps'] is not None else 'N/A'} | "
                f"Confidence={confidence_text:>6} | "
                f"BER={ber_text:<10} | "
                f"{result['status']}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    for modulation in signals:

        modulation_results = [
            r
            for r in results
            if r["modulation"] == modulation
        ]

        correct = sum(
            r["classification_ok"]
            for r in modulation_results
        )

        total = len(
            modulation_results
        )

        print(
            f"{modulation:<8} "
            f"classification: "
            f"{correct}/{total}"
        )

    print()
    print("Benchmark complete.")

def calculate_bpsk_ber(
    recovered_bits,
    reference_bits
):

    count = min(
        len(recovered_bits),
        len(reference_bits)
    )

    if count == 0:
        return None

    recovered = recovered_bits[:count]
    reference = reference_bits[:count]

    direct_errors = int(
        np.sum(
            recovered != reference
        )
    )

    inverted_errors = int(
        np.sum(
            (1 - recovered) != reference
        )
    )

    if inverted_errors < direct_errors:

        return {
            "errors": inverted_errors,
            "count": count,
            "ber": inverted_errors / count,
            "polarity": "inverted",
        }

    return {
        "errors": direct_errors,
        "count": count,
        "ber": direct_errors / count,
        "polarity": "direct",
    }



if __name__ == "__main__":
    main()

