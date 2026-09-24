import numpy as np

from prototype.core.signal import Signal
from prototype.detection.detector import (
    detect_signal,
    detect_candidates,
)
from prototype.core.isolator import isolate_signal
from prototype.parameters.extractor import extract_parameters
from prototype.parameters.symbol_rate import (
    estimate_symbol_rate,
    rrc_filter,
)
from prototype.core.synchronization import synchronize_signal
from prototype.classification.classifier import classify_signal
from prototype.demodulation.demodulator import (
    demodulate_signal,
    calculate_ber,
)


# ============================================================
# BPSK GENERATOR
# ============================================================

def generate_bpsk(
    num_symbols=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        num_symbols,
    )

    symbols = (
        2 * bits - 1
    ).astype(np.complex128)

    return symbols, bits


# ============================================================
# RRC PULSE SHAPING
# ============================================================

def pulse_shape(
    symbols,
    samples_per_symbol,
):
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
    delayed = np.concatenate(
        [
            np.zeros(
                timing_offset,
                dtype=np.complex128,
            ),
            samples,
        ]
    )

    n = np.arange(len(delayed))

    frequency_rotation = np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_offset
        * n
        / sample_rate
    )

    impaired = (
        delayed
        * frequency_rotation
    )

    phase_rotation = np.exp(
        1j * np.deg2rad(
            phase_offset_deg
        )
    )

    impaired *= phase_rotation

    return impaired


# ============================================================
# GENERATE INTERFERER
# ============================================================

def generate_interferer(
    num_samples,
    sample_rate,
    frequency=1500.0,
    amplitude=0.25,
):
    n = np.arange(num_samples)

    interferer = (
        amplitude
        * np.exp(
            1j
            * 2.0
            * np.pi
            * frequency
            * n
            / sample_rate
        )
    )

    return interferer


# ============================================================
# ADD NOISE
# ============================================================

def add_noise(
    samples,
    noise_amplitude=0.005,
    seed=123,
):
    rng = np.random.default_rng(seed)

    noise = (
        noise_amplitude
        * (
            rng.standard_normal(
                len(samples)
            )
            + 1j
            * rng.standard_normal(
                len(samples)
            )
        )
        / np.sqrt(2.0)
    )

    return samples + noise


# ============================================================
# MAIN TEST
# ============================================================

def test_test_end_to_end_bpsk_full():

    print("=" * 70)
    print("SPECTRA V2 FULL END-TO-END BPSK TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    sample_rate = 8000

    symbol_rate = 100
    samples_per_symbol = 80

    payload_symbols = 1000

    timing_offset = 13
    frequency_offset = 37.0
    phase_offset_deg = 30.0

    desired_frequency = 500.0
    interferer_frequency = 1500.0

    # --------------------------------------------------------
    # 1. Generate BPSK payload
    # --------------------------------------------------------

    print("\n[1] Generate BPSK payload")

    payload_symbols_data, reference_bits = (
        generate_bpsk(
            num_symbols=payload_symbols,
        )
    )

    print(
        f"Payload symbols: "
        f"{len(payload_symbols_data)}"
    )

    print(
        f"Payload reference bits: "
        f"{len(reference_bits)}"
    )

    # --------------------------------------------------------
    # 2. RRC pulse shaping
    # --------------------------------------------------------

    print("\n[2] RRC pulse shaping")

    shaped = pulse_shape(
        payload_symbols_data,
        samples_per_symbol,
    )

    print(
        f"Shaped samples: "
        f"{len(shaped)}"
    )

    print(
        f"Samples per symbol: "
        f"{samples_per_symbol}"
    )

    # --------------------------------------------------------
    # 3. Place desired signal at 500 Hz
    # --------------------------------------------------------

    print("\n[3] Place desired signal at 500 Hz")

    n = np.arange(len(shaped))

    desired_carrier = np.exp(
        1j
        * 2.0
        * np.pi
        * desired_frequency
        * n
        / sample_rate
    )

    desired_signal = (
        shaped
        * desired_carrier
    )

    print(
        f"Desired center frequency: "
        f"{desired_frequency:.1f} Hz"
    )

    # --------------------------------------------------------
    # 4. Apply impairments
    # --------------------------------------------------------

    print("\n[4] Apply impairments")

    impaired_desired = apply_impairments(
        desired_signal,
        sample_rate=sample_rate,
        timing_offset=timing_offset,
        frequency_offset=frequency_offset,
        phase_offset_deg=phase_offset_deg,
    )

    print(
        f"Timing offset: "
        f"{timing_offset} samples"
    )

    print(
        f"Additional frequency offset: "
        f"{frequency_offset:.1f} Hz"
    )

    print(
        f"Phase offset: "
        f"{phase_offset_deg:.1f} degrees"
    )

    # --------------------------------------------------------
    # 5. Add interferer
    # --------------------------------------------------------

    print("\n[5] Add interferer")

    interferer = generate_interferer(
        num_samples=len(impaired_desired),
        sample_rate=sample_rate,
        frequency=interferer_frequency,
        amplitude=0.25,
    )

    print(
        f"Interferer frequency: "
        f"{interferer_frequency:.1f} Hz"
    )

    print(
        "Interferer amplitude: 0.25"
    )

    # --------------------------------------------------------
    # 6. Add noise
    # --------------------------------------------------------

    print("\n[6] Add noise")

    received_samples = (
        impaired_desired
        + interferer
    )

    received_samples = add_noise(
        received_samples,
        noise_amplitude=0.005,
    )

    signal = Signal(
        samples=received_samples,
        sample_rate=sample_rate,
    )

    print(
        f"Input samples: "
        f"{signal.num_samples}"
    )

    print(
        f"Input duration: "
        f"{signal.duration:.4f} seconds"
    )

    # --------------------------------------------------------
    # 7. Detection
    # --------------------------------------------------------

    print("\n[7] Detection")

    detection_result = detect_signal(
        signal,
        threshold_db=6.0,
    )

    print(
        f"Signal detected: "
        f"{detection_result.detected}"
    )

    print(
        f"Number of candidates: "
        f"{len(detection_result.candidates)}"
    )

    assert detection_result.detected, (
        "Expected signal detection to succeed."
    )

    # --------------------------------------------------------
    # 8. Candidate selection
    # --------------------------------------------------------

    print("\n[8] Candidate selection")

    candidates = detect_candidates(
        signal,
        threshold_db=6.0,
    )

    print(
        f"Candidates returned: "
        f"{len(candidates)}"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"Candidate {index}: "
            f"center={candidate.center_frequency:.2f} Hz, "
            f"BW={candidate.bandwidth:.2f} Hz, "
            f"peak={candidate.peak_frequency:.2f} Hz, "
            f"confidence={candidate.confidence:.2f}"
        )

    assert len(candidates) > 0, (
        "Expected at least one signal candidate."
    )

    # Synthetic-test oracle:
    # select the candidate closest to the known
    # desired signal region after CFO.

    selected_candidate = min(
        candidates,
        key=lambda candidate: abs(
            candidate.center_frequency
            - (
                desired_frequency
                + frequency_offset
            )
        ),
    )

    print(
        "\nSelected candidate:"
    )

    print(
        f"Center frequency: "
        f"{selected_candidate.center_frequency:.2f} Hz"
    )

    print(
        f"Bandwidth: "
        f"{selected_candidate.bandwidth:.2f} Hz"
    )

    print(
        f"Peak frequency: "
        f"{selected_candidate.peak_frequency:.2f} Hz"
    )

    assert abs(
        selected_candidate.center_frequency
        - (
            desired_frequency
            + frequency_offset
        )
    ) < 150.0, (
        "Candidate selection did not identify "
        "the desired signal region."
    )

    # --------------------------------------------------------
    # 9. Isolation / DDC
    # --------------------------------------------------------

    print("\n[9] Isolation / DDC")

    isolation_result = isolate_signal(
        signal,
        center_frequency=(
            selected_candidate.center_frequency
        ),
        bandwidth=selected_candidate.bandwidth,
        filter_margin=1.25,
        filter_order=6,
    )

    isolated_signal = (
        isolation_result.signal
    )

    print(
        f"Isolation center: "
        f"{isolation_result.center_frequency:.2f} Hz"
    )

    print(
        f"Isolation bandwidth: "
        f"{isolation_result.bandwidth:.2f} Hz"
    )

    print(
        f"Low cutoff: "
        f"{isolation_result.low_cutoff:.2f} Hz"
    )

    print(
        f"High cutoff: "
        f"{isolation_result.high_cutoff:.2f} Hz"
    )

    print(
        f"Isolated RMS: "
        f"{isolated_signal.rms:.6f}"
    )

    # --------------------------------------------------------
    # 10. Parameter extraction
    # --------------------------------------------------------

    print("\n[10] Parameter extraction")

    parameters = extract_parameters(
        isolated_signal
    )

    print(
        f"Baseband peak frequency: "
        f"{parameters.peak_frequency:.4f} Hz"
    )

    print(
        f"Baseband center frequency: "
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

    assert abs(
        parameters.center_frequency
    ) < 100.0, (
        "Isolated signal was not successfully "
        "down-converted toward baseband."
    )

    # --------------------------------------------------------
    # 11. Symbol-rate estimation
    # --------------------------------------------------------

    print("\n[11] Symbol-rate estimation")

    symbol_rate_result = estimate_symbol_rate(
        isolated_signal.samples,
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

    symbol_rate_error = abs(
        estimated_symbol_rate
        - symbol_rate
    )

    print(
        f"Symbol-rate error: "
        f"{symbol_rate_error:.4f} baud"
    )

    assert symbol_rate_error <= 5.0, (
        f"Symbol-rate estimation failed: "
        f"expected approximately {symbol_rate}, "
        f"got {estimated_symbol_rate}"
    )

    # --------------------------------------------------------
    # 12. Synchronization
    # --------------------------------------------------------

    print("\n[12] Synchronization")

    sync_result = synchronize_signal(
        isolated_signal,
        symbol_rate=estimated_symbol_rate,
        modulation_order=2,
    )

    synchronized_symbols = (
        sync_result.signal.samples
    )

    


    estimated_phase_offset = float(
    sync_result.signal.metadata.get(
        "phase_offset",
        0.0,
       )
    )

    phase_correction = np.exp(
         -1j * estimated_phase_offset
     )

    corrected_symbols = (
        synchronized_symbols
       * phase_correction
     )

    print(
        f"Estimated frequency offset: "
        f"{sync_result.signal.metadata.get('frequency_offset', 'N/A')}"
    )

    print(
        f"Estimated phase offset: "
        f"{sync_result.signal.metadata.get('phase_offset', 'N/A')}"
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
        f"{len(synchronized_symbols)}"
    )

    assert len(
        synchronized_symbols
    ) > 900, (
        "Synchronization returned too few "
        "BPSK symbols."
    )

    # --------------------------------------------------------
    # 13. Classification
    # --------------------------------------------------------

    print("\n[13] Classification")

    synchronized_signal = Signal(
        samples=corrected_symbols,
        sample_rate=sync_result.signal.sample_rate,
        metadata=sync_result.signal.metadata.copy(),
    )

    classification = classify_signal(
        synchronized_signal,
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

    assert classification.modulation == "BPSK", (
        f"Expected BPSK, "
        f"got {classification.modulation}"
    )

    # --------------------------------------------------------
    # 14. V2 BPSK DEMODULATION
    # --------------------------------------------------------

    print("\n[14] V2 BPSK DEMODULATION")

    demodulation_result = demodulate_signal(
        synchronized_signal,
        modulation="BPSK",
        synchronized=True,
    )

    recovered_bits = (
        demodulation_result.bits
    )

    print(
        f"Recovered bits: "
        f"{len(recovered_bits)}"
    )

    print(
        f"Decision margin: "
        f"{demodulation_result.decision_margin:.6f}"
    )

    # --------------------------------------------------------
    # 15. BER
    # --------------------------------------------------------

    print("\n[15] BER")

    ber_result = calculate_ber(
        recovered_bits,
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
    # 16. Final validation
    # --------------------------------------------------------

    print("\n[16] Validation")

    print("✓ BPSK payload generated.")
    print("✓ RRC pulse shaping applied.")
    print("✓ Desired signal placed at 500 Hz.")
    print("✓ Timing impairment applied.")
    print("✓ Frequency impairment applied.")
    print("✓ Phase impairment applied.")
    print("✓ Interferer added at 1500 Hz.")
    print("✓ Noise added.")
    print("✓ Signal detection completed.")
    print("✓ Candidate selection completed.")
    print("✓ Desired signal candidate selected.")
    print("✓ Signal isolation completed.")
    print("✓ DDC completed.")
    print("✓ Parameter extraction completed.")
    print("✓ Symbol-rate estimation completed.")
    print("✓ Symbol-rate estimation is within tolerance.")
    print("✓ Carrier synchronization completed.")
    print("✓ Timing synchronization completed.")
    print("✓ BPSK classification passed.")
    print("✓ BPSK demodulation completed.")
    print("✓ Payload BER validation passed.")

    print("\n" + "=" * 70)
    print("✓ FULL END-TO-END BPSK TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
