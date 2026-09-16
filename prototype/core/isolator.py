import numpy as np
from scipy.signal import butter, sosfiltfilt


def isolate_signal(
    samples,
    sample_rate,
    center_frequency,
    bandwidth,
    guard_factor=1.5
):
    """
    Isolate an automatically detected signal.

    Handles both:
    - ordinary real RF signals
    - complex baseband IQ signals
    """

    if samples is None or len(samples) == 0:
        raise ValueError("Signal is empty.")

    nyquist = sample_rate / 2

    # --------------------------------------------------------
    # COMPLEX IQ BASEBAND
    # --------------------------------------------------------

    if np.iscomplexobj(samples):

        # Baseband signal centered near DC.
        if abs(center_frequency) < sample_rate * 0.05:

            # ``bandwidth`` describes the detected occupied *two-sided*
            # interval.  A low-pass cutoff must first reach the furthest
            # detected edge from DC, then leave transition/pulse-shaping
            # room outside that edge.  The old expression used only a
            # scaled half-bandwidth as the cutoff, which could clip a
            # slightly offset IQ signal and its timing information.
            detected_half_bandwidth = max(
                float(bandwidth) / 2.0,
                sample_rate / 10000,
            )

            furthest_detected_edge = (
                abs(float(center_frequency))
                + detected_half_bandwidth
            )

            # Add a generous, scale-aware guard on either side of the
            # detected interval.  This is based solely on the detector's
            # occupied-bandwidth estimate; it does not assume a symbol rate
            # or samples-per-symbol value.
            guard_hz = (
                max(float(guard_factor), 0.0)
                * detected_half_bandwidth
            )

            cutoff = max(
                furthest_detected_edge + guard_hz,
                sample_rate / 2000
            )

            cutoff = min(
                cutoff,
                nyquist * 0.9
            )

            normalized = (
                cutoff / nyquist
            )

            sos = butter(
                6,
                normalized,
                btype="lowpass",
                output="sos"
            )

            i_filtered = sosfiltfilt(
                sos,
                samples.real
            )

            q_filtered = sosfiltfilt(
                sos,
                samples.imag
            )

            filtered = (
                i_filtered
                + 1j * q_filtered
            )

            return filtered, {
                "mode": "IQ low-pass",
                "filter_low": -cutoff,
                "filter_high": cutoff,
                "center_frequency":
                    float(center_frequency),
                "bandwidth":
                    float(2 * cutoff),
            }

    # --------------------------------------------------------
    # GENERAL BAND-PASS
    # --------------------------------------------------------

    half_bandwidth = max(
        bandwidth * guard_factor / 2,
        sample_rate / 10000
    )

    low = (
        center_frequency
        - half_bandwidth
    )

    high = (
        center_frequency
        + half_bandwidth
    )

    low = max(
        low,
        1.0
    )

    high = min(
        high,
        nyquist - 1.0
    )

    if low >= high:

        raise ValueError(
            "Invalid filter bandwidth."
        )

    low_norm = low / nyquist
    high_norm = high / nyquist

    sos = butter(
        6,
        [
            low_norm,
            high_norm
        ],
        btype="bandpass",
        output="sos"
    )

    if np.iscomplexobj(samples):

        i_filtered = sosfiltfilt(
            sos,
            samples.real
        )

        q_filtered = sosfiltfilt(
            sos,
            samples.imag
        )

        filtered = (
            i_filtered
            + 1j * q_filtered
        )

    else:

        filtered = sosfiltfilt(
            sos,
            samples
        )

    return filtered, {
        "mode": "Band-pass",
        "filter_low":
            float(low),
        "filter_high":
            float(high),
        "center_frequency":
            float(center_frequency),
        "bandwidth":
            float(high - low),
    }
