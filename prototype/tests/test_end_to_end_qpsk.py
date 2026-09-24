import numpy as np
from prototype.core.signal import Signal
from prototype.parameters.symbol_rate import rrc_filter
from prototype.core.synchronization import synchronize_signal
from prototype.classification.classifier import classify_signal
from prototype.demodulation.demodulator import (
    demodulate_signal,
    calculate_ber,
)


def generate_qpsk(
    num_symbols=1000,
    samples_per_symbol=80,
    seed=42,
):
    """
    Generate QPSK symbols using the same Gray-coded mapping
    expected by the V2/V1 QPSK demodulator.

    Bit mapping:
        00 -> +1 + 1j
        01 -> -1 + 1j
        11 -> -1 - 1j
        10 -> +1 - 1j
    """

    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        size=(num_symbols, 2),
    )

    # Gray-coded QPSK constellation.
    #
    # 00 -> +1 + 1j
    # 01 -> -1 + 1j
    # 11 -> -1 - 1j
    # 10 -> +1 - 1j
    constellation = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    # Convert each bit pair to the corresponding constellation index.
    #
    # 00 -> 0
    # 01 -> 1
    # 10 -> 2
    # 11 -> 3
    #
    # The constellation array above is arranged according to the
    # Gray-coded sequence required by the demodulator.
    indices = (
        bits[:, 0] * 2
        + bits[:, 1]
    )

    # Rearrange binary indices into Gray-coded constellation order:
    #
    # binary 00 -> constellation 0
    # binary 01 -> constellation 1
    # binary 10 -> constellation 3
    # binary 11 -> constellation 2
    mapping = np.array(
        [
            0,
            1,
            3,
            2,
        ]
    )

    symbols = constellation[
        mapping[indices]
    ]

    return symbols, bits.reshape(-1)


def apply_timing_offset(
    samples,
    offset,
):
    """
    Add a deterministic sample delay.
    """

    if offset <= 0:
        return samples

    return np.concatenate(
        [
            np.zeros(
                offset,
                dtype=np.complex128,
            ),
            samples,
        ]
    )


def apply_frequency_offset(
    samples,
    sample_rate,
    frequency_offset,
):
    """
    Apply carrier-frequency offset.
    """

    n = np.arange(
        len(samples)
    )

    rotation = np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_offset
        * n
        / sample_rate
    )

    return samples * rotation


def apply_phase_offset(
    samples,
    phase_offset_deg,
):
    """
    Apply a constant carrier-phase rotation.
    """

    phase = np.deg2rad(
        phase_offset_deg
    )

    return samples * np.exp(
        1j * phase
    )


def test_test_end_to_end_qpsk():
    print("=" * 70)
    print("SPECTRA V2 END-TO-END QPSK TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    sample_rate = 8000
    symbol_rate = 100
    samples_per_symbol = 80

    timing_offset = 13
    frequency_offset = 37.0
    phase_offset_deg = 30.0

    # ---------------------------------------------------------
    # Generate QPSK
    # ---------------------------------------------------------

    print("\n[1] Generate QPSK")

    symbols, reference_bits = generate_qpsk(
        num_symbols=1000,
        samples_per_symbol=samples_per_symbol,
    )

    print(
        f"Symbols: {len(symbols)}"
    )

    print(
        f"Reference bits: "
        f"{len(reference_bits)}"
    )

    # ---------------------------------------------------------
    # Pulse shaping
    # ---------------------------------------------------------

    print("\n[2] RRC pulse shaping")

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

    # ---------------------------------------------------------
    # Impairments
    # ---------------------------------------------------------

    print("\n[3] Apply impairments")

    impaired = apply_timing_offset(
        shaped,
        timing_offset,
    )

    impaired = apply_frequency_offset(
        impaired,
        sample_rate,
        frequency_offset,
    )

    impaired = apply_phase_offset(
        impaired,
        phase_offset_deg,
    )

    print(
        f"Timing offset: "
        f"{timing_offset} samples"
    )

    print(
        f"Frequency offset: "
        f"{frequency_offset} Hz"
    )

    print(
        f"Phase offset: "
        f"{phase_offset_deg} degrees"
    )

    # ---------------------------------------------------------
    # Create V2 Signal
    # ---------------------------------------------------------

    signal = Signal(
        samples=impaired,
        sample_rate=sample_rate,
        metadata={
            "test": "e2e_qpsk",
            "known_modulation": "QPSK",
            "known_symbol_rate": symbol_rate,
        },
    )

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------

    print("\n[4] Synchronization")

    sync_result = synchronize_signal(
        signal,
        symbol_rate=symbol_rate,
        modulation_order=4,
    )

    print(
        f"Estimated frequency offset: "
        f"{sync_result.frequency_offset:.4f} Hz"
    )

    print(
        f"Frequency error: "
        f"{abs(sync_result.frequency_offset - frequency_offset):.4f} Hz"
    )

    print(
        f"Estimated phase offset: "
        f"{np.rad2deg(sync_result.phase_offset):.4f} degrees"
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
        f"{sync_result.signal.num_samples}"
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    print("\n[5] Classification")

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

    if classification.modulation != "QPSK":
        raise AssertionError(
            "Expected QPSK, "
            f"got {classification.modulation}"
        )

    # ---------------------------------------------------------
    # Demodulation
    # ---------------------------------------------------------

    print("\n[6] Demodulation")

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

    print(
        f"QPSK ambiguity rotation: "
        f"{demodulated.metadata['phase_ambiguity_rotation_deg']:.1f} degrees"
    )

    demodulated_bits = demodulated.bits

    # ---------------------------------------------------------
    # BER
    # ---------------------------------------------------------

    print("\n[7] BER")

    ber = calculate_ber(
        demodulated_bits,
        reference_bits,
    )

    print(
        f"Compared bits: "
        f"{ber['compared_bits']}"
    )

    print(
        f"Bit errors: "
        f"{ber['direct_errors']}"
    )

    print(
        f"BER: "
        f"{ber['direct_ber']:.6f}"
    )

    if ber["direct_ber"] != 0.0:
        raise AssertionError(
            f"Expected BER 0, "
            f"got {ber['direct_ber']}"
        )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    print("\n[8] Validation")

    print(
        "✓ QPSK signal generated."
    )

    print(
        "✓ RRC pulse shaping applied."
    )

    print(
        "✓ Timing impairment applied."
    )

    print(
        "✓ Frequency impairment applied."
    )

    print(
        "✓ Phase impairment applied."
    )

    print(
        "✓ Carrier synchronization completed."
    )

    print(
        "✓ Timing synchronization completed."
    )

    print(
        "✓ QPSK classification passed."
    )

    print(
        "✓ QPSK demodulation completed."
    )

    print(
        "✓ BER validation passed."
    )

    print("\n" + "=" * 70)

    print(
        "✓ END-TO-END QPSK TEST PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
