import numpy as np

from prototype.core.analyzer import (
    basic_stats,
    compute_spectrum,
)


def analyze_selected_signal(
    samples,
    sample_rate
):
    """
    Analyze an already-isolated signal.
    """

    if samples is None:
        raise ValueError(
            "No selected signal is available."
        )

    if len(samples) == 0:
        raise ValueError(
            "Selected signal is empty."
        )

    stats = basic_stats(
        samples,
        sample_rate
    )

    (
        frequency,
        magnitude,
        magnitude_db
    ) = compute_spectrum(
        samples,
        sample_rate
    )

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    peak_index = np.argmax(
        magnitude
    )

    dominant_frequency = float(
        frequency[peak_index]
    )

    # --------------------------------------------------------
    # Estimate local -3 dB bandwidth
    # --------------------------------------------------------

    peak_db = magnitude_db[
        peak_index
    ]

    threshold = peak_db - 3.0

    above = (
        magnitude_db >= threshold
    )

    indices = np.where(
        above
    )[0]

    if len(indices) > 0:

        lower = float(
            frequency[indices[0]]
        )

        upper = float(
            frequency[indices[-1]]
        )

        bandwidth = upper - lower

    else:

        lower = None
        upper = None
        bandwidth = None

    return {
        **stats,

        "frequency":
            frequency,

        "magnitude":
            magnitude,

        "magnitude_db":
            magnitude_db,

        "dominant_frequency":
            dominant_frequency,

        "lower_frequency":
            lower,

        "upper_frequency":
            upper,

        "bandwidth":
            bandwidth,
    }