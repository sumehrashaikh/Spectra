import numpy as np

from prototype.core.signal import Signal
from prototype.parameters.extractor import extract_parameters


def test_test_parameter_extractor():

    print("=" * 70)
    print("V2 PARAMETER EXTRACTION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Generate clean 500 Hz complex tone
    # ---------------------------------------------------------

    sample_rate = 8000
    frequency = 500
    duration = 2.0

    num_samples = int(
        sample_rate * duration
    )

    t = np.arange(
        num_samples
    ) / sample_rate

    samples = np.exp(
        1j
        * 2
        * np.pi
        * frequency
        * t
    )

    signal = Signal(
        samples=samples,
        sample_rate=sample_rate,
        metadata={
            "test": "clean 500 Hz tone",
        },
    )

    print("\n[1] Input signal")

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
    # Extract parameters
    # ---------------------------------------------------------

    print("\n[2] Extracting parameters...")

    parameters = extract_parameters(
        signal
    )

    print(
        f"Peak frequency  : "
        f"{parameters.peak_frequency:.2f} Hz"
    )

    print(
        f"Center frequency: "
        f"{parameters.center_frequency:.2f} Hz"
    )

    print(
        f"Bandwidth       : "
        f"{parameters.bandwidth:.2f} Hz"
    )

    print(
        f"RMS             : "
        f"{parameters.rms:.6f}"
    )

    print(
        f"Peak            : "
        f"{parameters.peak:.6f}"
    )

    print(
        f"Power           : "
        f"{parameters.power:.6f}"
    )

    print(
        f"Noise power     : "
        f"{parameters.noise_power:.6e}"
    )

    print(
        f"SNR             : "
        f"{parameters.snr_db:.2f} dB"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n[3] Validation")

    frequency_error = abs(
        parameters.peak_frequency
        - frequency
    )

    print(
        f"Frequency error : "
        f"{frequency_error:.2f} Hz"
    )

    if frequency_error > 5:
        raise AssertionError(
            "Peak-frequency estimation failed."
        )

    if parameters.rms < 0.9:
        raise AssertionError(
            "RMS estimation failed."
        )

    if parameters.peak < 0.9:
        raise AssertionError(
            "Peak estimation failed."
        )

    if parameters.power < 0.9:
        raise AssertionError(
            "Power estimation failed."
        )

    if parameters.bandwidth < 0:
        raise AssertionError(
            "Bandwidth cannot be negative."
        )

    print(
        "✓ Peak frequency estimation passed."
    )

    print(
        "✓ RMS estimation passed."
    )

    print(
        "✓ Peak estimation passed."
    )

    print(
        "✓ Power estimation passed."
    )

    print(
        "✓ Bandwidth estimation passed."
    )

    print("\n" + "=" * 70)
    print(
        "✓ PARAMETER EXTRACTION TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
