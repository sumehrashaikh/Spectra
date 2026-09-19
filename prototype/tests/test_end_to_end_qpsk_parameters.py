import numpy as np

from prototype.core.signal import Signal
from prototype.core.synchronization import synchronize_signal
from prototype.classification.classifier import classify_signal
from prototype.demodulation.demodulator import (
    demodulate_signal,
    calculate_ber,
)
from prototype.parameters.extractor import extract_parameters
from prototype.parameters.symbol_rate import (
    estimate_symbol_rate,
    rrc_filter,
)


# ============================================================
# QPSK GENERATOR
# ============================================================

def generate_qpsk(
    num_symbols=1000,
    samples_per_symbol=80,
    seed=42,
):
    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        size=(num_symbols, 2),
    )

    constellation = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    # Gray mapping:
    #
    # 00 -> +1 + 1j
    # 01 -> -1 + 1j
    # 11 -> -1 - 1j
    # 10 -> +1 - 1j

    indices = bits[:, 0] * 2 + bits[:, 1]

    mapping = np.array([0, 1, 3, 2])

    symbols = constellation[mapping[indices]]

    return symbols, bits.reshape(-1)


# ============================================================
# RRC PULSE SHAPING
# ============================================================

def pulse_shape(symbols, samples_per_symbol):
    upsampled = np.zeros(
        len(symbols) * samples_per_symbol,
        dtype=np.complex128,
    )

    upsampled[::samples_per_symbol] = symbols

    rrc_taps = rrc_filter(
        samples_per_symbol,
        rolloff=0.35,
        span_symbols=8,
    )

    shaped = np.convolve(
        upsampled,
        rrc_taps,
        mode="same",
    )

    return shaped


# ============================================================
# APPLY IMPAIRMENTS
# ============================================================

def apply_impairments(
    samples,
    sample_rate,
    timing_offset=13,
    frequency_offset=37.0,
    phase_offset_deg=30.0,
):
    # Timing offset
    delayed = np.concatenate(
        [
            np.zeros(timing_offset, dtype=np.complex128),
            samples,
        ]
    )

    # Frequency offset
    n = np.arange(len(delayed))

    frequency_rotation = np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_offset
        * n
        / sample_rate
    )

    impaired = delayed * frequency_rotation

    # Phase offset
    phase_rotation = np.exp(
        1j * np.deg2rad(phase_offset_deg)
    )

    impaired *= phase_rotation

    return impaired


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("SPECTRA V2 END-TO-END QPSK PARAMETER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    sample_rate = 8000
    symbol_rate = 100
    samples_per_symbol = 80

    num_symbols = 1000

    timing_offset = 13
    frequency_offset = 37.0
    phase_offset_deg = 30.0

    # --------------------------------------------------------
    # 1. Generate QPSK
    # --------------------------------------------------------

    print("\n[1] Generate QPSK")

    symbols, reference_bits = generate_qpsk(
        num_symbols=num_symbols,
        samples_per_symbol=samples_per_symbol,
    )

    print(f"Symbols: {len(symbols)}")
    print(f"Reference bits: {len(reference_bits)}")

    # --------------------------------------------------------
    # 2. RRC pulse shaping
    # --------------------------------------------------------

    print("\n[2] RRC pulse shaping")

    shaped = pulse_shape(
        symbols,
        samples_per_symbol,
    )

    print(f"Shaped samples: {len(shaped)}")
    print(f"Samples per symbol: {samples_per_symbol}")

    # --------------------------------------------------------
    # 3. Apply impairments
    # --------------------------------------------------------

    print("\n[3] Apply impairments")

    impaired = apply_impairments(
        shaped,
        sample_rate=sample_rate,
        timing_offset=timing_offset,
        frequency_offset=frequency_offset,
        phase_offset_deg=phase_offset_deg,
    )

    print(f"Timing offset: {timing_offset} samples")
    print(f"Frequency offset: {frequency_offset} Hz")
    print(f"Phase offset: {phase_offset_deg} degrees")

    signal = Signal(
        samples=impaired,
        sample_rate=sample_rate,
    )

    # --------------------------------------------------------
    # 4. Parameter extraction
    # --------------------------------------------------------

    print("\n[4] Parameter extraction")

    parameters = extract_parameters(signal)

    print(
        f"Peak frequency: "
        f"{parameters.peak_frequency:.4f} Hz"
    )

    print(
        f"Center frequency: "
        f"{parameters.center_frequency:.4f} Hz"
    )

    print(
        f"Bandwidth: "
        f"{parameters.bandwidth:.4f} Hz"
    )

    print(
        f"RMS: "
        f"{parameters.rms:.6f}"
    )

    print(
        f"Peak: "
        f"{parameters.peak:.6f}"
    )

    print(
        f"Power: "
        f"{parameters.power:.6f}"
    )

    print(
        f"Noise power: "
        f"{parameters.noise_power:.8f}"
    )

    print(
        f"SNR: "
        f"{parameters.snr_db:.4f} dB"
    )

    # --------------------------------------------------------
    # 5. Symbol-rate estimation
    # --------------------------------------------------------

    print("\n[5] Symbol-rate estimation")

    symbol_rate_result = estimate_symbol_rate(
        signal.samples,
        sample_rate=sample_rate,
        min_symbol_rate=50,
        max_symbol_rate=500,
    )

    estimated_symbol_rate = (
        symbol_rate_result.symbol_rate
    )

    print(
        f"Actual symbol rate: "
        f"{symbol_rate} baud"
    )

    print(
        f"Estimated symbol rate: "
        f"{estimated_symbol_rate:.4f} baud"
    )

    print(
        f"Estimated SPS: "
        f"{symbol_rate_result.samples_per_symbol:.4f}"
    )

    print(
        f"Confidence: "
        f"{symbol_rate_result.confidence:.2f}%"
    )

    print(
        f"Method: "
        f"{symbol_rate_result.method}"
    )

    # --------------------------------------------------------
    # Validate symbol-rate estimation
    # --------------------------------------------------------

    symbol_rate_error = abs(
        estimated_symbol_rate - symbol_rate
    )

    print(
        f"Symbol-rate error: "
        f"{symbol_rate_error:.4f} baud"
    )

    # For this synthetic test, the estimator should recover
    # the known 100-baud signal.

    assert symbol_rate_error <= 5.0, (
        f"Symbol-rate estimation failed: "
        f"expected approximately {symbol_rate}, "
        f"got {estimated_symbol_rate}"
    )

    # --------------------------------------------------------
    # 6. Synchronization
    # --------------------------------------------------------

    print("\n[6] Synchronization")

    sync_result = synchronize_signal(
        signal,
        symbol_rate=estimated_symbol_rate,
        modulation_order=4,
    )

    print(
        f"Estimated frequency offset: "
        f"{sync_result.frequency_offset:.4f} Hz"
        if hasattr(sync_result, "frequency_offset")
        else "Carrier synchronization completed"
    )

    print(
        f"Timing offset: "
        f"{sync_result.timing_offset}"
    )

    print(
        f"SPS: "
        f"{sync_result.samples_per_symbol:.2f}"
    )

    print(
        f"Recovered symbols: "
        f"{len(sync_result.signal.samples)}"
    )

    # --------------------------------------------------------
    # 7. Classification
    # --------------------------------------------------------

    print("\n[7] Classification")

    classification = classify_signal(
        sync_result.signal,
        use_constellation=True,
        synchronized=True,
    )

    print(
        f"Detected modulation: "
        f"{classification.modulation}"
    )

    print(
        f"Confidence: "
        f"{classification.confidence:.2f}%"
    )

    print(
        f"Method: "
        f"{classification.method}"
    )

    assert classification.modulation == "QPSK", (
        f"Expected QPSK, "
        f"got {classification.modulation}"
    )

    # --------------------------------------------------------
    # 8. Demodulation
    # --------------------------------------------------------

    print("\n[8] Demodulation")

    demodulated = demodulate_signal(
        sync_result.signal,
        modulation="QPSK",
        synchronized=True,
    )

    print(
        f"Recovered symbols: "
        f"{demodulated.num_symbols}"
    )

    print(
        f"Recovered bits: "
        f"{demodulated.num_bits}"
    )

    print(
        f"Decision margin: "
        f"{demodulated.decision_margin:.4f}"
    )

    if "phase_ambiguity_rotation_deg" in demodulated.metadata:
        print(
            "QPSK ambiguity rotation: "
            f"{demodulated.metadata['phase_ambiguity_rotation_deg']:.1f}"
            " degrees"
        )

        # --------------------------------------------------------
    # 9. BER
    # --------------------------------------------------------

    print("\n[9] BER")

    ber_result = calculate_ber(
        demodulated.bits,
        reference_bits,
    )

    print(
        f"Compared bits: "
        f"{ber_result['compared_bits']}"
    )

    print(
        f"Bit errors: "
        f"{ber_result['direct_errors']}"
    )

    print(
        f"BER: "
        f"{ber_result['direct_ber']:.6f}"
    )

    assert ber_result["direct_ber"] == 0.0, (
        f"Expected BER 0, "
        f"got {ber_result['direct_ber']}"
    )
    # --------------------------------------------------------
    # 10. Validation
    # --------------------------------------------------------

    print("\n[10] Validation")

    print("✓ QPSK signal generated.")
    print("✓ RRC pulse shaping applied.")
    print("✓ Timing impairment applied.")
    print("✓ Frequency impairment applied.")
    print("✓ Phase impairment applied.")
    print("✓ Parameter extraction completed.")
    print("✓ Symbol-rate estimation completed.")
    print("✓ Symbol-rate estimation is within tolerance.")
    print("✓ Carrier synchronization completed.")
    print("✓ Timing synchronization completed.")
    print("✓ QPSK classification passed.")
    print("✓ QPSK demodulation completed.")
    print("✓ BER validation passed.")

    print("\n" + "=" * 70)
    print("✓ END-TO-END QPSK PARAMETER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()