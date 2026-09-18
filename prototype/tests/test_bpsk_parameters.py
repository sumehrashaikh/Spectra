import numpy as np

from prototype.core.signal import Signal
from prototype.detection.detector import detect_signal
from prototype.core.isolator import isolate_signal
from prototype.parameters.extractor import extract_parameters


def generate_bpsk(
    sample_rate=8000,
    symbol_rate=100,
    duration=2.0,
    frequency=500.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    samples_per_symbol = int(
        sample_rate / symbol_rate
    )

    num_symbols = int(
        duration * symbol_rate
    )

    bits = rng.integers(
        0,
        2,
        num_symbols,
    )

    symbols = 2 * bits - 1

    baseband = np.repeat(
        symbols,
        samples_per_symbol,
    ).astype(np.complex128)

    t = np.arange(
        len(baseband)
    ) / sample_rate

    carrier = np.exp(
        1j
        * 2
        * np.pi
        * frequency
        * t
    )

    return baseband * carrier


def main():

    print("=" * 70)
    print("V2 BPSK PARAMETER EXTRACTION INTEGRATION TEST")
    print("=" * 70)

    sample_rate = 8000

    # ---------------------------------------------------------
    # 1. Generate communication signal
    # ---------------------------------------------------------

    bpsk = generate_bpsk(
        sample_rate=sample_rate,
        symbol_rate=100,
        duration=2.0,
        frequency=500.0,
        seed=42,
    )

    t = np.arange(
        len(bpsk)
    ) / sample_rate

    # Strong interferer.
    interferer = (
        0.8
        * np.exp(
            1j
            * 2
            * np.pi
            * 1500
            * t
        )
    )

    combined = bpsk + interferer

    signal = Signal(
        samples=combined,
        sample_rate=sample_rate,
        metadata={
            "test": "BPSK + 1500 Hz interferer",
            "known_bpsk_frequency": 500.0,
            "known_symbol_rate": 100.0,
            "known_interferer_frequency": 1500.0,
        },
    )

    print("\n[1] Input")

    print(
        f"Sample rate : "
        f"{signal.sample_rate:.2f} Hz"
    )

    print(
        f"Samples     : "
        f"{signal.num_samples}"
    )

    print(
        f"Duration    : "
        f"{signal.duration:.2f} s"
    )

    # ---------------------------------------------------------
    # 2. Detection
    # ---------------------------------------------------------

    print("\n[2] Detection")

    detection = detect_signal(signal)

    print(
        f"Detected   : "
        f"{detection.detected}"
    )

    print(
        f"Candidates : "
        f"{len(detection.candidates)}"
    )

    if not detection.detected:
        raise AssertionError(
            "Signal detection failed."
        )

    # Find candidate closest to known BPSK frequency.
    selected = min(
        detection.candidates,
        key=lambda candidate: abs(
            candidate.center_frequency
            - 500.0
        ),
    )

    print(
        f"Selected center frequency : "
        f"{selected.center_frequency:.2f} Hz"
    )

    print(
        f"Selected bandwidth        : "
        f"{selected.bandwidth:.2f} Hz"
    )

    # ---------------------------------------------------------
    # 3. Isolation / DDC
    # ---------------------------------------------------------

    print("\n[3] Isolation / DDC")

    isolation = isolate_signal(
        signal,
        center_frequency=selected.center_frequency,
        bandwidth=selected.bandwidth,
    )

    isolated = isolation.signal

    print(
        f"Isolated center : "
        f"{isolation.center_frequency:.2f} Hz"
    )

    print(
        f"Isolated BW     : "
        f"{isolation.bandwidth:.2f} Hz"
    )

    print(
        f"Low cutoff      : "
        f"{isolation.low_cutoff:.2f} Hz"
    )

    print(
        f"High cutoff     : "
        f"{isolation.high_cutoff:.2f} Hz"
    )

    # ---------------------------------------------------------
    # 4. Parameter extraction
    # ---------------------------------------------------------

    print("\n[4] Parameter extraction")

    parameters = extract_parameters(
        isolated
    )

    print(
        f"Peak frequency   : "
        f"{parameters.peak_frequency:.2f} Hz"
    )

    print(
        f"Center frequency : "
        f"{parameters.center_frequency:.2f} Hz"
    )

    print(
        f"Bandwidth        : "
        f"{parameters.bandwidth:.2f} Hz"
    )

    print(
        f"RMS              : "
        f"{parameters.rms:.6f}"
    )

    print(
        f"Peak             : "
        f"{parameters.peak:.6f}"
    )

    print(
        f"Power            : "
        f"{parameters.power:.6f}"
    )

    print(
        f"Noise power      : "
        f"{parameters.noise_power:.6e}"
    )

    print(
        f"SNR              : "
        f"{parameters.snr_db:.2f} dB"
    )

    # ---------------------------------------------------------
    # 5. Validation
    # ---------------------------------------------------------

    print("\n[5] Validation")

    baseband_frequency_error = abs(
        parameters.peak_frequency
    )

    print(
        f"Baseband peak-frequency error : "
        f"{baseband_frequency_error:.2f} Hz"
    )

    if baseband_frequency_error > 50:
        raise AssertionError(
            "Isolated signal was not correctly "
            "represented as a baseband signal."
        )

    center_frequency_error = abs(
        parameters.center_frequency
    )

    print(
        f"Baseband center-frequency error : "
        f"{center_frequency_error:.2f} Hz"
    )

    if center_frequency_error > 50:
        raise AssertionError(
            "Baseband center frequency is too far "
            "from 0 Hz."
        )

    if parameters.rms <= 0:
        raise AssertionError(
            "Invalid RMS."
        )

    if parameters.peak <= 0:
        raise AssertionError(
            "Invalid peak."
        )

    if parameters.power <= 0:
        raise AssertionError(
            "Invalid signal power."
        )

    if parameters.bandwidth < 0:
        raise AssertionError(
            "Bandwidth cannot be negative."
        )

    print(
        "✓ Baseband frequency extraction passed."
    )

    print(
        "✓ Baseband center-frequency extraction passed."
    )

    print(
        "✓ Power measurements passed."
    )

    print(
        "✓ Bandwidth extraction passed."
    )

    print("\n" + "=" * 70)
    print(
        "✓ BPSK PARAMETER EXTRACTION "
        "INTEGRATION TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()