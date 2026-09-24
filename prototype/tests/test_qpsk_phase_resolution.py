import numpy as np

from prototype.demodulation.qpsk_sync import (
    resolve_qpsk_phase,
    rotate_qpsk,
)


def test_test_qpsk_phase_resolution():

    print("=" * 70)
    print("SPECTRA V2 QPSK PHASE AMBIGUITY TEST")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Generate known QPSK reference symbols
    # --------------------------------------------------------

    print("\n[1] Generate reference QPSK")

    bits = rng.integers(
        0,
        2,
        size=(1000, 2),
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

    indices = bits[:, 0] * 2 + bits[:, 1]

    mapping = np.array(
        [0, 1, 3, 2]
    )

    reference_symbols = constellation[
        mapping[indices]
    ]

    print(
        f"Reference symbols: "
        f"{len(reference_symbols)}"
    )

    # --------------------------------------------------------
    # Apply a known ambiguity
    # --------------------------------------------------------

    print("\n[2] Apply 90-degree ambiguity")

    received_symbols = rotate_qpsk(
        reference_symbols,
        90.0,
    )

    print("Injected ambiguity: 90.0 degrees")

    # --------------------------------------------------------
    # Resolve ambiguity
    # --------------------------------------------------------

    print("\n[3] Resolve phase ambiguity")

    result = resolve_qpsk_phase(
        received_symbols,
        reference_symbols,
    )

    print(
        f"Estimated correction: "
        f"{result['rotation_deg']:.1f} degrees"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"Mean error: "
        f"{result['mean_error']:.8f}"
    )

    print("\nCandidates:")

    for candidate in result["candidates"]:
        print(
            f"  "
            f"{candidate['rotation_deg']:.0f}°"
            f" → error="
            f"{candidate['mean_error']:.8f}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    #
    # The received signal was rotated +90°.
    #
    # To undo that rotation, the resolver should determine
    # the equivalent correcting orientation.
    #
    # Our rotate_qpsk() function applies the selected
    # rotation directly, so the expected correction is
    # 270° (= -90°).
    #

    assert result["rotation_deg"] == 270.0, (
        "Expected a 270-degree correction "
        "for a 90-degree received rotation."
    )

    corrected = rotate_qpsk(
        received_symbols,
        result["rotation_deg"],
    )

    corrected_error = np.mean(
        np.abs(
            corrected
            - reference_symbols
        )
    )

    print(
        f"\nCorrected mean error: "
        f"{corrected_error:.10f}"
    )

    assert corrected_error < 1e-10, (
        "QPSK phase ambiguity correction failed."
    )

    print("\n[4] Validation")

    print("✓ QPSK reference generated.")
    print("✓ 90-degree phase ambiguity applied.")
    print("✓ Four ambiguity rotations evaluated.")
    print("✓ Correct phase orientation selected.")
    print("✓ Phase ambiguity successfully resolved.")

    print("\n" + "=" * 70)
    print("✓ QPSK PHASE AMBIGUITY TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
