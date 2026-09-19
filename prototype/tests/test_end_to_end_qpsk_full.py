import numpy as np
from prototype.demodulation.qpsk_sync import qpsk_symbol_decision
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
from prototype.demodulation.qpsk_sync import (
    resolve_qpsk_phase,
    rotate_qpsk,
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
# CREATE KNOWN QPSK PREAMBLE
# ============================================================

def generate_qpsk_preamble(
    num_symbols=64,
    seed=2026,
):
    """
    Generate a deterministic, non-periodic QPSK preamble.

    The receiver knows this sequence in advance and uses it
    to resolve QPSK's 90-degree phase ambiguity.

    A pseudo-random sequence is used instead of a repeating
    four-symbol pattern so that phase rotation cannot be
    confused with a symbol-position shift.
    """

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

    mapping = np.array(
        [0, 1, 3, 2]
    )

    preamble = constellation[
        mapping[indices]
    ]

    return preamble

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
    # Timing offset
    delayed = np.concatenate(
        [
            np.zeros(
                timing_offset,
                dtype=np.complex128,
            ),
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
# GENERATE INTERFERER
# ============================================================

def generate_interferer(
    num_samples,
    sample_rate,
    frequency=1500.0,
    amplitude=0.25,
):
    n = np.arange(num_samples)

    interferer = amplitude * np.exp(
        1j
        * 2.0
        * np.pi
        * frequency
        * n
        / sample_rate
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

    noise = noise_amplitude * (
        rng.standard_normal(len(samples))
        + 1j * rng.standard_normal(len(samples))
    ) / np.sqrt(2.0)

    return samples + noise


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("SPECTRA V2 FULL END-TO-END QPSK TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    sample_rate = 8000

    symbol_rate = 100
    samples_per_symbol = 80

    preamble_symbols = 64
    payload_symbols = 1000

    timing_offset = 13
    frequency_offset = 37.0
    phase_offset_deg = 30.0

    desired_frequency = 500.0
    interferer_frequency = 1500.0

    # --------------------------------------------------------
    # 1. Generate known preamble
    # --------------------------------------------------------

    print("\n[1] Generate QPSK preamble")

    preamble = generate_qpsk_preamble(
        preamble_symbols
    )

    print(
        f"Preamble symbols: "
        f"{len(preamble)}"
    )

    # --------------------------------------------------------
    # 2. Generate QPSK payload
    # --------------------------------------------------------

    print("\n[2] Generate QPSK payload")

    payload_symbols, reference_bits = generate_qpsk(
        num_symbols=payload_symbols,
        samples_per_symbol=samples_per_symbol,
    )

    print(
        f"Payload symbols: "
        f"{len(payload_symbols)}"
    )

    print(
        f"Payload reference bits: "
        f"{len(reference_bits)}"
    )

    # --------------------------------------------------------
    # 3. Combine preamble + payload
    # --------------------------------------------------------

    print("\n[3] Build frame")

    transmit_symbols = np.concatenate(
        [
            preamble,
            payload_symbols,
        ]
    )

    print(
        f"Total symbols: "
        f"{len(transmit_symbols)}"
    )

    print(
        f"Preamble: "
        f"{len(preamble)} symbols"
    )

    print(
        f"Payload: "
        f"{len(payload_symbols)} symbols"
    )

    # --------------------------------------------------------
    # 4. RRC pulse shaping
    # --------------------------------------------------------

    print("\n[4] RRC pulse shaping")

    shaped = pulse_shape(
        transmit_symbols,
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
    # 5. Place desired signal at 500 Hz
    # --------------------------------------------------------

    print("\n[5] Place desired signal at 500 Hz")

    n = np.arange(len(shaped))

    desired_carrier = np.exp(
        1j
        * 2.0
        * np.pi
        * desired_frequency
        * n
        / sample_rate
    )

    desired_signal = shaped * desired_carrier

    print(
        f"Desired center frequency: "
        f"{desired_frequency:.1f} Hz"
    )

    # --------------------------------------------------------
    # 6. Apply impairments
    # --------------------------------------------------------

    print("\n[6] Apply impairments")

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
    # 7. Add interferer
    # --------------------------------------------------------

    print("\n[7] Add interferer")

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
    # 8. Add noise
    # --------------------------------------------------------

    print("\n[8] Add noise")

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
    # 9. Detection
    # --------------------------------------------------------

    print("\n[9] Detection")

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
    # 10. Candidate selection
    # --------------------------------------------------------

    print("\n[10] Candidate selection")

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

    # --------------------------------------------------------
    # Select candidate closest to expected desired region.
    #
    # This is a synthetic-test oracle only.
    # --------------------------------------------------------

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
    # 11. Isolation / DDC
    # --------------------------------------------------------

    print("\n[11] Isolation / DDC")

    isolation_result = isolate_signal(
        signal,
        center_frequency=selected_candidate.center_frequency,
        bandwidth=selected_candidate.bandwidth,
        filter_margin=1.25,
        filter_order=6,
    )

    isolated_signal = isolation_result.signal

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
    # 12. Parameter extraction
    # --------------------------------------------------------

    print("\n[12] Parameter extraction")

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
    # 13. Symbol-rate estimation
    # --------------------------------------------------------

    print("\n[13] Symbol-rate estimation")

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
    # 14. Synchronization
    # --------------------------------------------------------

    print("\n[14] Synchronization")

    sync_result = synchronize_signal(
        isolated_signal,
        symbol_rate=estimated_symbol_rate,
        modulation_order=4,
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
        f"{len(sync_result.signal.samples)}"
    )

    # --------------------------------------------------------
    # 15. Preamble phase ambiguity resolution
    # --------------------------------------------------------

    print(
        "\n[15] QPSK preamble phase resolution"
    )

    synchronized_symbols = (
        sync_result.signal.samples
    )

    preamble_count = len(preamble)

    assert len(synchronized_symbols) >= (
        preamble_count
    ), (
        "Synchronization returned fewer symbols "
        "than the preamble length."
    )

    received_preamble = (
        synchronized_symbols[:preamble_count]
    )

    print("\nPreamble rotation diagnostics:")

    for rotation_deg in (0.0, 90.0, 180.0, 270.0):

        rotated = rotate_qpsk(
            received_preamble,
            rotation_deg,
        )

        received_mag = np.abs(rotated)
        reference_mag = np.abs(preamble)

        valid = (
            (received_mag > 1e-12)
            & (reference_mag > 1e-12)
        )

        normalized_received = (
            rotated[valid]
            / received_mag[valid]
        )

        normalized_reference = (
            preamble[valid]
            / reference_mag[valid]
        )

        error = float(
            np.mean(
                np.abs(
                    normalized_received
                    - normalized_reference
                )
            )
        )

        print(
            f"  {rotation_deg:6.1f}° "
            f"→ normalized error = {error:.8f}"
        )

    phase_result = resolve_qpsk_phase(
        received_preamble,
        preamble,
    )

    phase_rotation_deg = (
        phase_result["rotation_deg"]
    )

    print(
        f"Resolved phase correction: "
        f"{phase_rotation_deg:.1f} degrees"
    )

    print(
        f"Confidence: "
        f"{phase_result['confidence']:.2f}%"
    )

    print(
        f"Mean preamble error: "
        f"{phase_result['mean_error']:.8f}"
    )

    # Apply the resolved phase correction to the
    # complete synchronized frame.

    corrected_symbols = rotate_qpsk(
        synchronized_symbols,
        phase_rotation_deg,
    )

    corrected_signal = Signal(
        samples=corrected_symbols,
        sample_rate=sync_result.signal.sample_rate,
        metadata=sync_result.signal.metadata.copy(),
    )

    corrected_signal.add_metadata(
        qpsk_phase_correction_deg=phase_rotation_deg,
        qpsk_phase_resolution_confidence=(
            phase_result["confidence"]
        ),
    )

    # Validate the preamble after correction.
    #
    # The synchronized signal has an arbitrary amplitude because
    # of RRC pulse shaping and isolation filtering. Therefore,
    # phase validation must compare normalized constellation
    # directions rather than raw amplitudes.

    corrected_preamble = (
        corrected_symbols[:preamble_count]
    )

    corrected_magnitude = np.abs(
        corrected_preamble
    )

    reference_magnitude = np.abs(
        preamble
    )

    valid = (
        (corrected_magnitude > 1e-12)
        & (reference_magnitude > 1e-12)
    )

    normalized_corrected = (
        corrected_preamble[valid]
        / corrected_magnitude[valid]
    )

    normalized_reference = (
        preamble[valid]
        / reference_magnitude[valid]
    )

    normalized_preamble_error = float(
        np.mean(
            np.abs(
                normalized_corrected
                - normalized_reference
            )
        )
    )

    print(
        f"Corrected normalized preamble error: "
        f"{normalized_preamble_error:.8f}"
    )

    assert normalized_preamble_error < 0.25, (
        "QPSK preamble phase resolution failed."
    )

    # --------------------------------------------------------
    # 16. Classification
    # --------------------------------------------------------

    print("\n[16] Classification")

    classification = classify_signal(
        corrected_signal,
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
    # 17. Remove preamble before payload demodulation
    # --------------------------------------------------------

    print("\n[17] Extract payload")

    payload_start = preamble_count

    synchronized_payload = (
        corrected_symbols[payload_start:]
    )

    print(
        f"Payload symbols available: "
        f"{len(synchronized_payload)}"
    )

    payload_signal = Signal(
        samples=synchronized_payload,
        sample_rate=corrected_signal.sample_rate,
        metadata=corrected_signal.metadata.copy(),
    )

    print("\nQPSK payload rotation diagnostics:")

    for rotation_deg in (0.0, 90.0, 180.0, 270.0):
        rotated_payload = rotate_qpsk(
            synchronized_payload,
            rotation_deg,
        )

        test_bits = qpsk_symbol_decision(
            rotated_payload
        )

        test_ber = calculate_ber(
            test_bits,
            reference_bits,
        )

        print(
            f"  {rotation_deg:6.1f}° → "
            f"BER = {test_ber['direct_ber']:.6f}, "
            f"errors = {test_ber['direct_errors']}"
        )


    print("\nPayload constellation statistics:")

    payload_magnitudes = np.abs(synchronized_payload)

    print(
        f"Mean magnitude: "
        f"{np.mean(payload_magnitudes):.6f}"
    )

    print(
        f"Std magnitude: "
        f"{np.std(payload_magnitudes):.6f}"
    )

    print(
        f"Mean I: "
        f"{np.mean(synchronized_payload.real):.6f}"
    )

    print(
        f"Mean Q: "
        f"{np.mean(synchronized_payload.imag):.6f}"
    )

    print("\nQPSK payload phase diagnostics:")

    for index in [0, 100, 250, 500, 750, 999]:
        received = synchronized_payload[index]
        reference = payload_symbols[index]

        phase_error = np.angle(
            received * np.conj(reference),
            deg=True,
        )

        print(
            f"  Symbol {index:4d}: "
            f"received={received.real:+.4f}{received.imag:+.4f}j, "
            f"reference={reference.real:+.4f}{reference.imag:+.4f}j, "
            f"phase error={phase_error:+.2f}°"
        )

    # --------------------------------------------------------
    # 18. Demodulation
    # --------------------------------------------------------

    print("\n[18] DIRECT QPSK DEMODULATION TEST")

    direct_bits = qpsk_symbol_decision(
        synchronized_payload
    )

    direct_ber = calculate_ber(
        direct_bits,
        reference_bits
    )

    print(
        f"Direct recovered bits: {len(direct_bits)}"
    )

    print(
        f"Direct bit errors: "
        f"{direct_ber['direct_errors']}"
    )

    print(
        f"Direct BER: "
        f"{direct_ber['direct_ber']:.6f}"
    )

    print("\nFirst 10 symbols / bits:")

    for i in range(10):
        received_symbol = synchronized_payload[i]
        reference_symbol = payload_symbols[i]

        received_bits = direct_bits[
            i * 2:(i + 1) * 2
        ]

        reference_bit_pair = reference_bits[
            i * 2:(i + 1) * 2
        ]

        print(
            f"{i:2d}: "
            f"RX={received_symbol.real:+.4f}"
            f"{received_symbol.imag:+.4f}j "
            f"REF={reference_symbol.real:+.4f}"
            f"{reference_symbol.imag:+.4f}j "
            f"RX bits={received_bits.tolist()} "
            f"REF bits={reference_bit_pair.tolist()}"
        )

    # --------------------------------------------------------
    # 19. BER
    # --------------------------------------------------------

    print("\n[19] BER")

    ber_result = calculate_ber(
        direct_bits,
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
    # 20. Final validation
    # --------------------------------------------------------

    print("\n[20] Validation")

    print("✓ QPSK preamble generated.")
    print("✓ QPSK payload generated.")
    print("✓ Frame constructed.")
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
    print("✓ QPSK preamble phase ambiguity resolved.")
    print("✓ QPSK classification passed.")
    print("✓ Payload extracted.")
    print("✓ QPSK demodulation completed.")
    print("✓ Payload BER validation passed.")

    print("\n" + "=" * 70)
    print("✓ FULL END-TO-END QPSK TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()