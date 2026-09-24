import numpy as np

from prototype.core.signal import Signal
from prototype.detection.detector import detect_signal
from prototype.core.isolator import isolate_signal


def generate_bpsk(
    sample_rate=8000,
    symbol_rate=100,
    duration=2.0,
    frequency=500.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = int(duration * symbol_rate)

    bits = rng.integers(0, 2, num_symbols)
    symbols = 2 * bits - 1

    baseband = np.repeat(
        symbols,
        samples_per_symbol,
    ).astype(np.complex128)

    t = np.arange(len(baseband)) / sample_rate
    carrier = np.exp(1j * 2 * np.pi * frequency * t)

    return baseband * carrier


def generate_test_signal():
    sample_rate = 8000

    # Main BPSK signal at 500 Hz
    bpsk = generate_bpsk(
        sample_rate=sample_rate,
        symbol_rate=100,
        duration=2.0,
        frequency=500.0,
        seed=42,
    )

    # Strong interferer at 1500 Hz
    t = np.arange(len(bpsk)) / sample_rate
    interferer = 0.8 * np.exp(
        1j * 2 * np.pi * 1500 * t
    )

    combined = bpsk + interferer

    return Signal(
        samples=combined,
        sample_rate=sample_rate,
        metadata={
            "test": "BPSK + 1500 Hz interferer",
            "bpsk_frequency": 500.0,
            "interferer_frequency": 1500.0,
        },
    )


def select_candidate(detection_result, target_frequency):
    """Select the candidate closest to the target frequency."""

    if not detection_result.candidates:
        raise RuntimeError(
            "No detection candidates available."
        )

    return min(
        detection_result.candidates,
        key=lambda candidate: abs(
            candidate.center_frequency - target_frequency
        ),
    )


def test_test_detection_isolation():
    print("=" * 70)
    print("V2 DETECTION → CANDIDATE SELECTION → ISOLATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Generate input signal
    # ---------------------------------------------------------

    signal = generate_test_signal()

    print("\n[1] Input signal")
    print(f"Sample rate : {signal.sample_rate} Hz")
    print(f"Samples     : {signal.num_samples}")
    print(f"Duration    : {signal.duration:.2f} s")

    # ---------------------------------------------------------
    # 2. Detection
    # ---------------------------------------------------------

    print("\n[2] Running detection...")

    detection = detect_signal(signal)

    print(f"Detected    : {detection.detected}")
    print(f"Candidates  : {len(detection.candidates)}")

    if not detection.detected:
        raise RuntimeError(
            "Detection failed."
        )

    for i, candidate in enumerate(
        detection.candidates,
        start=1,
    ):
        print(
            f"Candidate {i}: "
            f"center={candidate.center_frequency:.2f} Hz, "
            f"bandwidth={candidate.bandwidth:.2f} Hz, "
            f"peak={candidate.peak_frequency:.2f} Hz, "
            f"confidence={candidate.confidence:.1f}%"
        )

    # ---------------------------------------------------------
    # 3. Candidate selection
    # ---------------------------------------------------------

    print("\n[3] Selecting 500 Hz candidate...")

    selected = select_candidate(
        detection,
        target_frequency=500.0,
    )

    print(
        f"Selected candidate: "
        f"center={selected.center_frequency:.2f} Hz, "
        f"bandwidth={selected.bandwidth:.2f} Hz"
    )

    # Make sure the correct candidate was selected.
    if abs(selected.center_frequency - 500.0) > 100.0:
        raise AssertionError(
            "Candidate selection failed: "
            "500 Hz candidate was not selected."
        )

    print("✓ Correct candidate selected.")

    # ---------------------------------------------------------
    # 4. Isolation / DDC
    # ---------------------------------------------------------

    print("\n[4] Running isolation / DDC...")

    isolation_result = isolate_signal(
        signal,
        center_frequency=selected.center_frequency,
        bandwidth=selected.bandwidth,
    )

    # IsolationResult contains the actual Signal here.
    isolated = isolation_result.signal

    print(
        f"Output sample rate : "
        f"{isolated.sample_rate:.2f} Hz"
    )

    print(
        f"Output samples     : "
        f"{isolated.num_samples}"
    )

    print(
        f"Output RMS         : "
        f"{isolated.rms:.6f}"
    )

    print(
        f"Output peak        : "
        f"{isolated.peak:.6f}"
    )

    print("\nIsolation result:")

    print(
        f"  Center frequency : "
        f"{isolation_result.center_frequency:.2f} Hz"
    )

    print(
        f"  Bandwidth        : "
        f"{isolation_result.bandwidth:.2f} Hz"
    )

    print(
        f"  Low cutoff       : "
        f"{isolation_result.low_cutoff:.2f} Hz"
    )

    print(
        f"  High cutoff      : "
        f"{isolation_result.high_cutoff:.2f} Hz"
    )

    # ---------------------------------------------------------
    # 5. Validate DDC
    # ---------------------------------------------------------

    print("\n[5] Baseband validation")

    spectrum = np.fft.fftshift(
        np.fft.fft(isolated.samples)
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            isolated.num_samples,
            d=1 / isolated.sample_rate,
        )
    )

    peak_index = np.argmax(np.abs(spectrum))

    output_peak_frequency = frequencies[peak_index]

    print(
        f"Output dominant frequency: "
        f"{output_peak_frequency:.2f} Hz"
    )

    # The selected 500 Hz signal should have been
    # shifted to approximately 0 Hz.
    if abs(output_peak_frequency) > 100.0:
        raise AssertionError(
            "Isolation/DDC failed: "
            "selected signal was not moved "
            "close to baseband."
        )

    print(
        "✓ Selected signal successfully "
        "moved to baseband."
    )

    # ---------------------------------------------------------
    # 6. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "✓ DETECTION → SELECTION → ISOLATION "
        "PIPELINE PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
