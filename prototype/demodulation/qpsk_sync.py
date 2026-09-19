import numpy as np


QPSK_AMBIGUITY_ROTATIONS_DEG = (
    0.0,
    90.0,
    180.0,
    270.0,
)


def rotate_qpsk(
    samples: np.ndarray,
    rotation_deg: float,
) -> np.ndarray:
    """
    Rotate QPSK samples counter-clockwise by rotation_deg.
    """
    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot rotate empty QPSK samples."
        )

    rotation = np.exp(
        1j * np.deg2rad(rotation_deg)
    )

    return samples * rotation

def qpsk_symbol_decision(samples):
    """
    Make hard QPSK decisions using the Spectra QPSK
    Gray-coded bit mapping.

    Mapping:

        00 -> +I +Q
        01 -> -I +Q
        10 -> +I -Q
        11 -> -I -Q
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if samples.size == 0:
        raise ValueError(
            "Cannot make QPSK decisions from empty samples."
        )

    i_negative = samples.real < 0
    q_negative = samples.imag < 0

    symbol_bits = np.empty(
        (samples.size, 2),
        dtype=np.int8,
    )

    # QPSK Gray mapping:
    #
    # +I +Q -> 00
    # -I +Q -> 01
    # +I -Q -> 10
    # -I -Q -> 11

    symbol_bits[:, 0] = q_negative.astype(np.int8)
    symbol_bits[:, 1] = i_negative.astype(np.int8)

    return symbol_bits.reshape(-1)

def resolve_qpsk_phase(
    received_symbols: np.ndarray,
    reference_symbols: np.ndarray,
) -> dict:
    """
    Resolve QPSK's 90-degree phase ambiguity.

    The reference symbols represent a known preamble.

    The function tests four possible CORRECTIONS:

        0°
        -90°
        -180°
        -270°

    The returned rotation is directly usable with
    rotate_qpsk(received_symbols, rotation_deg).

    Therefore, if the received signal is +90° rotated
    relative to the reference, the resolver returns
    -90° represented as 270°.
    """

    received_symbols = np.asarray(
        received_symbols,
        dtype=np.complex128,
    )

    reference_symbols = np.asarray(
        reference_symbols,
        dtype=np.complex128,
    )

    if received_symbols.size == 0:
        raise ValueError(
            "Received symbols cannot be empty."
        )

    if reference_symbols.size == 0:
        raise ValueError(
            "Reference symbols cannot be empty."
        )

    count = min(
        received_symbols.size,
        reference_symbols.size,
    )

    received = received_symbols[:count]
    reference = reference_symbols[:count]

    reference_magnitude = np.abs(reference)

    valid = reference_magnitude > 1e-12

    if not np.any(valid):
        raise ValueError(
            "Reference symbols contain no valid symbols."
        )

    received = received[valid]
    reference = reference[valid]

    reference_normalized = (
        reference
        / np.maximum(
            np.abs(reference),
            1e-12,
        )
    )

    results = []

    for rotation_deg in QPSK_AMBIGUITY_ROTATIONS_DEG:

        corrected = rotate_qpsk(
            received,
            rotation_deg,
        )

        corrected_normalized = (
            corrected
            / np.maximum(
                np.abs(corrected),
                1e-12,
            )
        )

        error = np.abs(
            corrected_normalized
            - reference_normalized
        )

        mean_error = float(
            np.mean(error)
        )

        results.append(
            {
                "rotation_deg": rotation_deg,
                "mean_error": mean_error,
            }
        )

    best = min(
        results,
        key=lambda item: item["mean_error"],
    )

    # Maximum possible normalized complex-vector error
    # is approximately 2. Convert the geometric error
    # into a 0-100 confidence value.
    confidence = float(
        np.clip(
            100.0
            * (
                1.0
                - best["mean_error"] / 2.0
            ),
            0.0,
            100.0,
        )
    )

    return {
        "rotation_deg": best["rotation_deg"],
        "confidence": confidence,
        "mean_error": best["mean_error"],
        "candidates": results,
    }