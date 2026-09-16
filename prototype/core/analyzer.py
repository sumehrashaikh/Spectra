import numpy as np
from scipy.signal import find_peaks, welch


# ============================================================
# BASIC STATISTICS
# ============================================================

def basic_stats(samples, sample_rate):

    if samples is None or len(samples) == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be positive."
        )

    magnitude = np.abs(samples)
    power = magnitude ** 2

    return {
        "sample_rate": float(sample_rate),
        "num_samples": int(len(samples)),
        "duration": float(
            len(samples) / sample_rate
        ),
        "rms": float(
            np.sqrt(np.mean(power))
        ),
        "peak": float(
            np.max(magnitude)
        ),
    }


# ============================================================
# SPECTRUM
# ============================================================

def compute_spectrum(samples, sample_rate):

    if len(samples) == 0:
        raise ValueError("Signal is empty.")

    x = np.asarray(samples)

    # --------------------------------------------------------
    # Complex IQ
    # --------------------------------------------------------

    if np.iscomplexobj(x):

        nperseg = min(
            4096,
            len(x)
        )

        frequencies, psd = welch(
            x,
            fs=sample_rate,
            nperseg=nperseg,
            return_onesided=False,
            scaling="density"
        )

        frequencies = np.fft.fftshift(
            frequencies
        )

        psd = np.fft.fftshift(
            psd
        )

        magnitude = np.sqrt(
            np.maximum(psd, 0)
        )

    # --------------------------------------------------------
    # Real-valued WAV
    # --------------------------------------------------------

    else:

        x = x.astype(float)

        n = len(x)

        window = np.hanning(n)

        spectrum = np.fft.rfft(
            x * window
        )

        frequencies = np.fft.rfftfreq(
            n,
            d=1 / sample_rate
        )

        magnitude = (
            2.0
            * np.abs(spectrum)
            / np.sum(window)
        )

    magnitude_db = (
        20
        * np.log10(
            magnitude + 1e-12
        )
    )

    return (
        frequencies,
        magnitude,
        magnitude_db
    )


# ============================================================
# NOISE FLOOR
# ============================================================

def estimate_noise_floor(magnitude_db):

    if len(magnitude_db) == 0:
        raise ValueError(
            "Spectrum is empty."
        )

    sorted_db = np.sort(
        magnitude_db
    )

    # Lower 70% gives a robust background estimate.
    cutoff = max(
        1,
        int(0.70 * len(sorted_db))
    )

    background = sorted_db[
        :cutoff
    ]

    floor = np.median(
        background
    )

    dynamic_range = (
        np.max(magnitude_db)
        - floor
    )

    # Very clean synthetic data may have a
    # numerical FFT floor instead of physical noise.
    if dynamic_range > 100:

        return {
            "noise_floor_db": None,
            "measurement_floor_db":
                float(floor),
            "noise_limited": False,
        }

    return {
        "noise_floor_db":
            float(floor),
        "measurement_floor_db":
            float(floor),
        "noise_limited": True,
    }


# ============================================================
# IQ BASEBAND REGION
# ============================================================

def detect_iq_baseband_region(
    frequency,
    magnitude_db,
    sample_rate
):
    """
    Detect significant spectral regions in complex IQ data.

    Unlike the previous version, this does NOT assume that an IQ
    signal must be centered around DC.

    It detects actual spectral peaks and builds occupied regions
    around those peaks.
    """

    if len(frequency) == 0:
        return [], 0.0

    # --------------------------------------------------------
    # Estimate noise / measurement floor
    # --------------------------------------------------------

    noise_info = estimate_noise_floor(magnitude_db)

    noise_floor = noise_info["noise_floor_db"]

    if noise_floor is None:
        peak_level = float(np.max(magnitude_db))

        # Relative threshold for very clean synthetic signals
        threshold = peak_level - 35.0
    else:
        threshold = noise_floor + 6.0

    # --------------------------------------------------------
    # Smooth spectrum
    # --------------------------------------------------------

    window = 9

    kernel = np.ones(window) / window

    smoothed = np.convolve(
        magnitude_db,
        kernel,
        mode="same"
    )

    # --------------------------------------------------------
    # Frequency resolution
    # --------------------------------------------------------

    if len(frequency) > 1:
        resolution = abs(
            frequency[1] - frequency[0]
        )
    else:
        resolution = 1.0

    # --------------------------------------------------------
    # Detect spectral peaks
    # --------------------------------------------------------

    distance_bins = max(
        1,
        int(50.0 / resolution)
    )

    peaks, properties = find_peaks(
        smoothed,
        height=threshold,
        distance=distance_bins,
        prominence=3.0
    )

    if len(peaks) == 0:
        return [], float(threshold)

    # --------------------------------------------------------
    # Sort peaks by strength
    # --------------------------------------------------------

    peaks = sorted(
        peaks,
        key=lambda p: smoothed[p],
        reverse=True
    )

    # Keep only meaningful strongest peaks
    max_peaks = min(
        len(peaks),
        12
    )

    peaks = peaks[:max_peaks]

    # --------------------------------------------------------
    # Build spectral regions around peaks
    # --------------------------------------------------------

    regions = []

    for peak in peaks:

        peak_level = smoothed[peak]

        # Region threshold relative to this peak
        local_threshold = max(
            threshold,
            peak_level - 20.0
        )

        left = peak
        right = peak

        while (
            left > 0
            and smoothed[left - 1] >= local_threshold
        ):
            left -= 1

        while (
            right < len(smoothed) - 1
            and smoothed[right + 1] >= local_threshold
        ):
            right += 1

        lower = float(
            frequency[left]
        )

        upper = float(
            frequency[right]
        )

        bandwidth = abs(
            upper - lower
        )

        # Ignore essentially zero-width numerical spikes
        if bandwidth < resolution:
            bandwidth = resolution

        regions.append({
            "peak_frequency":
                float(frequency[peak]),

            "lower_frequency":
                lower,

            "upper_frequency":
                upper,

            "bandwidth":
                float(bandwidth),

            "peak_magnitude_db":
                float(magnitude_db[peak]),
        })

    # --------------------------------------------------------
    # Merge overlapping regions
    # --------------------------------------------------------

    regions.sort(
        key=lambda r: r["lower_frequency"]
    )

    merged = []

    for region in regions:

        if not merged:
            merged.append(region)
            continue

        previous = merged[-1]

        if (
            region["lower_frequency"]
            <= previous["upper_frequency"]
        ):

            previous["upper_frequency"] = max(
                previous["upper_frequency"],
                region["upper_frequency"]
            )

            previous["lower_frequency"] = min(
                previous["lower_frequency"],
                region["lower_frequency"]
            )

            previous["bandwidth"] = (
                previous["upper_frequency"]
                - previous["lower_frequency"]
            )

            if (
                region["peak_magnitude_db"]
                > previous["peak_magnitude_db"]
            ):
                previous["peak_frequency"] = (
                    region["peak_frequency"]
                )

                previous["peak_magnitude_db"] = (
                    region["peak_magnitude_db"]
                )

        else:
            merged.append(region)

    # --------------------------------------------------------
    # Sort strongest signals first
    # --------------------------------------------------------

    merged.sort(
        key=lambda r: r["peak_magnitude_db"],
        reverse=True
    )

    return merged, float(threshold)

# ============================================================
# NARROW PEAK DETECTOR
# ============================================================

def detect_real_signal_peaks(
    frequency,
    magnitude,
    magnitude_db,
    noise_floor_db,
):

    if noise_floor_db is None:

        threshold = (
            np.max(magnitude_db)
            - 40.0
        )

    else:

        threshold = (
            noise_floor_db + 10.0
        )

    if len(frequency) > 1:

        resolution = (
            frequency[1]
            - frequency[0]
        )

    else:

        resolution = 1.0

    distance_bins = max(
        1,
        int(20.0 / resolution)
    )

    peaks, _ = find_peaks(
        magnitude_db,
        height=threshold,
        distance=distance_bins,
        prominence=3.0
    )

    return peaks, float(threshold)


# ============================================================
# REAL SIGNAL REGIONS
# ============================================================

def build_signal_regions(
    frequency,
    magnitude_db,
    peaks,
    threshold_db
):

    regions = []

    for peak in peaks:

        peak_level = magnitude_db[
            peak
        ]

        local_threshold = (
            peak_level - 20.0
        )

        left = peak
        right = peak

        while (
            left > 0
            and magnitude_db[left - 1]
            >= local_threshold
        ):
            left -= 1

        while (
            right < len(magnitude_db) - 1
            and magnitude_db[right + 1]
            >= local_threshold
        ):
            right += 1

        regions.append({
            "peak_frequency":
                float(frequency[peak]),

            "lower_frequency":
                float(frequency[left]),

            "upper_frequency":
                float(frequency[right]),

            "bandwidth":
                float(
                    frequency[right]
                    - frequency[left]
                ),

            "peak_magnitude_db":
                float(
                    magnitude_db[peak]
                ),
        })

    return regions


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_signal(
    samples,
    sample_rate
):

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

    is_iq = np.iscomplexobj(
        samples
    )

    noise_info = estimate_noise_floor(
        magnitude_db
    )

    noise_floor_db = (
        noise_info["noise_floor_db"]
    )

    # --------------------------------------------------------
    # IQ SIGNAL
    # --------------------------------------------------------

    if is_iq:

        regions, threshold_db = (
            detect_iq_baseband_region(
                frequency,
                magnitude_db,
                sample_rate
            )
        )

    # --------------------------------------------------------
    # REAL WAV
    # --------------------------------------------------------

    else:

        peaks, threshold_db = (
            detect_real_signal_peaks(
                frequency,
                magnitude,
                magnitude_db,
                noise_floor_db
            )
        )

        regions = build_signal_regions(
            frequency,
            magnitude_db,
            peaks,
            threshold_db
        )

    # --------------------------------------------------------
    # Detected signals
    # --------------------------------------------------------

    detected_signals = []

    for index, region in enumerate(
        regions,
        start=1
    ):

        detected_signals.append({
            "id": index,
            "frequency":
                region["peak_frequency"],
            "lower_frequency":
                region["lower_frequency"],
            "upper_frequency":
                region["upper_frequency"],
            "bandwidth":
                region["bandwidth"],
            "peak_magnitude_db":
                region["peak_magnitude_db"],
        })

    signal_detected = (
        len(detected_signals) > 0
    )

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    if signal_detected:

        dominant_frequency = max(
            detected_signals,
            key=lambda s:
                s["peak_magnitude_db"]
        )["frequency"]

    else:

        dominant_frequency = None

    # --------------------------------------------------------
    # Primary region
    # --------------------------------------------------------

    if signal_detected:

        primary = max(
            detected_signals,
            key=lambda s:
                s["peak_magnitude_db"]
        )

        lower_frequency = (
            primary["lower_frequency"]
        )

        upper_frequency = (
            primary["upper_frequency"]
        )

        bandwidth = (
            primary["bandwidth"]
        )

    else:

        lower_frequency = None
        upper_frequency = None
        bandwidth = None

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr_db = None

    if (
        not noise_info["noise_limited"]
    ):

        snr_db = None

    elif (
        signal_detected
        and noise_floor_db is not None
    ):

        signal_mask = np.zeros(
            len(frequency),
            dtype=bool
        )

        for region in detected_signals:

            signal_mask |= (
                (
                    frequency
                    >= region[
                        "lower_frequency"
                    ]
                )
                &
                (
                    frequency
                    <= region[
                        "upper_frequency"
                    ]
                )
            )

        signal_power = np.sum(
            magnitude[
                signal_mask
            ] ** 2
        )

        noise_power = np.sum(
            magnitude[
                ~signal_mask
            ] ** 2
        )

        if (
            signal_power > 0
            and noise_power > 1e-20
        ):

            snr_db = float(
                10
                * np.log10(
                    signal_power
                    / noise_power
                )
            )

    return {
        **stats,

        "frequency":
            frequency,

        "magnitude":
            magnitude,

        "magnitude_db":
            magnitude_db,

        "is_iq":
            is_iq,

        "noise_floor_db":
            noise_floor_db,

        "measurement_floor_db":
            noise_info[
                "measurement_floor_db"
            ],

        "noise_limited":
            noise_info[
                "noise_limited"
            ],

        "threshold_db":
            threshold_db,

        "signal_detected":
            signal_detected,

        "dominant_frequency":
            dominant_frequency,

        "lower_frequency":
            lower_frequency,

        "upper_frequency":
            upper_frequency,

        "bandwidth":
            bandwidth,

        "snr_db":
            snr_db,

        "detected_signals":
            detected_signals,
    }