from pyexpat import features

import numpy as np


def _phase_frequency_hz(samples, sample_rate):
    if len(samples) < 2:
        return np.array([])

    phase = np.unwrap(np.angle(samples))

    dphase = np.diff(phase)

    frequency_hz = (
        dphase
        * sample_rate
        / (2 * np.pi)
    )

    return frequency_hz


def extract_modulation_features(samples, sample_rate):

    if samples is None or len(samples) == 0:
        raise ValueError("Signal is empty.")

    samples = np.asarray(samples)

    amplitude = np.abs(samples)

    mean_amplitude = np.mean(amplitude)

    amplitude_std = np.std(amplitude)

    amplitude_cv = (
        amplitude_std
        / (mean_amplitude + 1e-12)
    )

    # --------------------------------------------------------
    # Normalize phase
    # --------------------------------------------------------

    normalized = (
        samples
        / (amplitude + 1e-12)
    )

    phase = np.angle(normalized)

    r2 = np.abs(
        np.mean(
            np.exp(1j * 2 * phase)
        )
    )

    r4 = np.abs(
        np.mean(
            np.exp(1j * 4 * phase)
        )
    )

    # --------------------------------------------------------
    # Instantaneous frequency
    # --------------------------------------------------------

    frequency_hz = _phase_frequency_hz(
        samples,
        sample_rate
    )

    if len(frequency_hz) > 10:

        lower = np.percentile(
            frequency_hz,
            5
        )

        upper = np.percentile(
            frequency_hz,
            95
        )

        robust_frequency = (
            frequency_hz[
                (frequency_hz >= lower)
                &
                (frequency_hz <= upper)
            ]
        )

        frequency_std = float(
            np.std(robust_frequency)
        )

        frequency_median = float(
            np.median(robust_frequency)
        )

    else:

        frequency_std = 0.0
        frequency_median = 0.0

    # --------------------------------------------------------
    # Amplitude distribution
    # --------------------------------------------------------

    amplitude_p10 = np.percentile(
        amplitude,
        10
    )

    amplitude_p50 = np.percentile(
        amplitude,
        50
    )

    amplitude_p90 = np.percentile(
        amplitude,
        90
    )

    robust_amplitude_spread = (
        amplitude_p90
        - amplitude_p10
    ) / (
        amplitude_p50
        + 1e-12
    )

    # --------------------------------------------------------
    # Amplitude histogram
    # --------------------------------------------------------

    if amplitude_p50 > 1e-12:

        normalized_amplitude = np.clip(
            amplitude / amplitude_p50,
            0.0,
            3.0
        )

        histogram, _ = np.histogram(
            normalized_amplitude,
            bins=30,
            range=(0.0, 3.0)
        )

        histogram_threshold = (
            np.max(histogram)
            * 0.15
        )

        meaningful_bins = int(
            np.sum(
                histogram
                > histogram_threshold
            )
        )

    else:

        meaningful_bins = 0

    # --------------------------------------------------------
    # NEW: amplitude level estimate
    #
    # 16-QAM has four amplitude magnitudes
    # along each I/Q axis:
    #
    # -3, -1, +1, +3
    #
    # Therefore the symbol amplitudes form
    # several distinct amplitude groups.
    # --------------------------------------------------------

    amplitude_std_ratio = (
        amplitude_std
        / (
            mean_amplitude
            + 1e-12
        )
    )

    return {
        "amplitude_cv":
            float(amplitude_cv),

        "amplitude_p10":
            float(amplitude_p10),

        "amplitude_p50":
            float(amplitude_p50),

        "amplitude_p90":
            float(amplitude_p90),

        "robust_amplitude_spread":
            float(robust_amplitude_spread),

        "meaningful_amplitude_bins":
            int(meaningful_bins),

        "amplitude_std_ratio":
            float(amplitude_std_ratio),

        "r2_phase_coherence":
            float(r2),

        "r4_phase_coherence":
            float(r4),

        "instantaneous_frequency_std_hz":
            float(frequency_std),

        "instantaneous_frequency_median_hz":
            float(frequency_median),
    }


def classify_modulation(samples, sample_rate):

    features = extract_modulation_features(
        samples,
        sample_rate
    )

    cv = features[
        "amplitude_cv"
    ]

    robust_spread = features[
        "robust_amplitude_spread"
    ]

    meaningful_bins = features[
        "meaningful_amplitude_bins"
    ]

    r2 = features[
        "r2_phase_coherence"
    ]

    r4 = features[
        "r4_phase_coherence"
    ]

    freq_std = features[
        "instantaneous_frequency_std_hz"
    ]

    # ========================================================
    # BFSK
    # ========================================================
    #
    # BFSK has nearly constant amplitude but its instantaneous
    # frequency switches between two levels.  This is a stronger
    # discriminator than the amplitude features used by 16-QAM.
    #
    # IMPORTANT:
    # Check BFSK BEFORE 16-QAM. At low SNR, frequency-switching
    # signals can acquire a broad amplitude distribution and may
    # otherwise be mistaken for 16-QAM.
    # ========================================================

    if (
        cv < 0.35
        and freq_std >= 80
        and r2 < 0.55
        and r4 < 0.55
    ):

        return (
            "BFSK",
            features
        )

    # ========================================================
    # 16-QAM
    # ========================================================
    #
    # 16-QAM:
    # - amplitude varies significantly
    # - phase is not concentrated into 2 or 4 states
    # - amplitude distribution is broader
    #
    # Strong frequency variation is characteristic of BFSK.
    # Exclude it here so noisy BFSK is not swallowed by the
    # broad 16-QAM amplitude rule.
    # ========================================================

    if (
        cv >= 0.20
        and robust_spread >= 0.45
        and meaningful_bins >= 5
        and r2 < 0.55
        and r4 < 0.85
        and freq_std < 80
    ):

        return (
            "16-QAM",
            features
        )


    # ========================================================
    # BPSK
    # ========================================================

    if r2 >= 0.55:

        return (
            "BPSK",
            features
        )

    # ========================================================
    # QPSK
    # ========================================================

    if r4 >= 0.55:

        return (
            "QPSK",
            features
        )

    # ========================================================
    # Unknown
    # ========================================================

    return (
        "Unknown",
        features
    )

# ============================================================
# SYMBOL-LEVEL CONSTELLATION FEATURES
# ============================================================


def _cluster_levels(values, max_levels):
    """
    Estimate distinct amplitude levels along one axis
    using robust 1-D k-means.

    Returns:
        centers: sorted cluster centers
        labels: cluster assignment for each value
        counts: number of values in each cluster
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return (
            np.array([]),
            np.array([], dtype=int),
            np.array([])
        )

    scale = np.std(values)

    if scale < 1e-12:
        normalized = values.copy()
    else:
        normalized = values / scale

    n = len(normalized)
    k = min(max_levels, n)

    initializations = []

    # Evenly spaced initialization
    initializations.append(
        np.linspace(
            np.min(normalized),
            np.max(normalized),
            k
        )
    )

    # Quantile initialization
    quantiles = np.linspace(0, 1, k)

    initializations.append(
        np.quantile(
            normalized,
            quantiles
        )
    )

    # Percentile initialization
    if k == 2:
        initializations.append(
            np.array([
                np.percentile(normalized, 25),
                np.percentile(normalized, 75)
            ])
        )

    elif k == 4:
        initializations.append(
            np.array([
                np.percentile(normalized, 12.5),
                np.percentile(normalized, 37.5),
                np.percentile(normalized, 62.5),
                np.percentile(normalized, 87.5)
            ])
        )

    best_centers = None
    best_labels = None
    best_error = np.inf

    for initial_centers in initializations:

        centers = np.asarray(
            initial_centers,
            dtype=float
        ).copy()

        for _ in range(100):

            distances = np.abs(
                normalized[:, None]
                - centers[None, :]
            )

            labels = np.argmin(
                distances,
                axis=1
            )

            new_centers = centers.copy()

            for i in range(k):

                cluster = normalized[
                    labels == i
                ]

                if len(cluster) > 0:
                    new_centers[i] = np.mean(
                        cluster
                    )

            new_centers = np.sort(
                new_centers
            )

            if np.allclose(
                centers,
                new_centers,
                atol=1e-8
            ):
                centers = new_centers
                break

            centers = new_centers

        distances = np.abs(
            normalized[:, None]
            - centers[None, :]
        )

        labels = np.argmin(
            distances,
            axis=1
        )

        error = np.mean(
            (
                normalized
                - centers[labels]
            ) ** 2
        )

        if error < best_error:
            best_error = error
            best_centers = centers.copy()
            best_labels = labels.copy()

    counts = np.array([
        np.sum(best_labels == i)
        for i in range(len(best_centers))
    ])

    valid = counts > 0

    centers = best_centers[valid]

    distances = np.abs(
        normalized[:, None]
        - centers[None, :]
    )

    labels = np.argmin(
        distances,
        axis=1
    )

    counts = np.array([
        np.sum(labels == i)
        for i in range(len(centers))
    ])

    return centers, labels, counts


def _axis_constellation_score(values):
    """
    Determine amplitude-level structure on one axis.

    Calculates:
        - 2-level clustering error
        - 4-level clustering error
        - level spacing
        - cluster occupancy
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {
            "levels": 0,
            "centers": np.array([]),
            "counts": np.array([]),
            "occupancy": np.array([]),
            "separation": 0.0,
            "error_2": np.inf,
            "error_4": np.inf,
            "level_spacing_ratio": 0.0,
        }

    scale = np.std(values)

    if scale < 1e-12:
        normalized = values.copy()
    else:
        normalized = values / scale

    def clustering_error(k):

        centers, labels, counts = _cluster_levels(
            normalized,
            k
        )

        if len(centers) == 0:
            return np.inf

        error = np.mean(
            (
                normalized
                - centers[labels]
            ) ** 2
        )

        return float(error)

    error_2 = clustering_error(2)
    error_4 = clustering_error(4)

    centers, labels, counts = _cluster_levels(
        normalized,
        4
    )

    valid = counts > 0

    centers = centers[valid]
    counts = counts[valid]

    # ---------------------------------------------------------
    # Four-level spacing
    # ---------------------------------------------------------

    if len(centers) >= 4:

        gaps = np.diff(centers)

        if np.max(gaps) > 1e-12:
            level_spacing_ratio = (
                np.min(gaps)
                / np.max(gaps)
            )
        else:
            level_spacing_ratio = 0.0

    else:
        level_spacing_ratio = 0.0

    # ---------------------------------------------------------
    # Minimum separation
    # ---------------------------------------------------------

    if len(centers) <= 1:
        separation = 0.0
    else:
        gaps = np.diff(centers)
        separation = float(
            np.min(gaps)
        )

    occupancy = counts / np.sum(counts)

    return {
        "levels": len(centers),
        "centers": centers,
        "counts": counts,
        "occupancy": occupancy,
        "separation": separation,
        "error_2": error_2,
        "error_4": error_4,
        "level_spacing_ratio":
            float(level_spacing_ratio),
    }


def _two_level_structure(values):
    """
    Measure how well an axis is represented by
    two approximately symmetric amplitude levels.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {
            "centers": np.array([]),
            "occupancy": np.array([]),
            "symmetry": 0.0,
        }

    scale = np.std(values)

    if scale < 1e-12:
        normalized = values.copy()
    else:
        normalized = values / scale

    centers, labels, counts = _cluster_levels(
        normalized,
        2
    )

    if len(centers) != 2:

        return {
            "centers": centers,
            "occupancy": (
                counts / np.sum(counts)
                if len(counts)
                else np.array([])
            ),
            "symmetry": 0.0,
        }

    occupancy = counts / np.sum(counts)

    # Symmetry around zero.
    symmetry = 1.0 - (
        abs(
            abs(centers[0])
            - abs(centers[1])
        )
        / max(
            abs(centers[0]),
            abs(centers[1]),
            1e-12
        )
    )

    return {
        "centers": centers,
        "occupancy": occupancy,
        "symmetry": float(
            np.clip(
                symmetry,
                0.0,
                1.0
            )
        ),
    }


def extract_constellation_features(symbols):
    """
    Extract geometry-based features from
    symbol-rate samples.
    """
    symbols = np.asarray(
        symbols,
        dtype=complex
    )

    if len(symbols) == 0:

        return {
            "axis_ratio": 0.0,
            "axis_energy_ratio": 0.0,

            "i_error_2": np.inf,
            "i_error_4": np.inf,
            "q_error_2": np.inf,
            "q_error_4": np.inf,

            "i_levels": 0,
            "q_levels": 0,

            "i_centers": np.array([]),
            "q_centers": np.array([]),

            "i_counts": np.array([]),
            "q_counts": np.array([]),

            "i_occupancy": np.array([]),
            "q_occupancy": np.array([]),

            "i_separation": 0.0,
            "q_separation": 0.0,

            "i_spacing_ratio": 0.0,
            "q_spacing_ratio": 0.0,

            "i_two_centers": np.array([]),
            "q_two_centers": np.array([]),

            "i_two_occupancy": np.array([]),
            "q_two_occupancy": np.array([]),

            "i_two_symmetry": 0.0,
            "q_two_symmetry": 0.0,
        }

    # ---------------------------------------------------------
    # Normalize constellation energy
    # ---------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            np.abs(symbols) ** 2
        )
    )

    if rms > 1e-12:
        symbols = symbols / rms

    i_values = np.real(symbols)
    q_values = np.imag(symbols)

    i_energy = np.mean(
        i_values ** 2
    )

    q_energy = np.mean(
        q_values ** 2
    )

    axis_ratio = (
        min(i_energy, q_energy)
        / max(
            i_energy,
            q_energy,
            1e-12
        )
    )

    # ---------------------------------------------------------
    # Four-level analysis
    # ---------------------------------------------------------

    i_info = _axis_constellation_score(
        i_values
    )

    q_info = _axis_constellation_score(
        q_values
    )

    # ---------------------------------------------------------
    # Two-level analysis
    # ---------------------------------------------------------

    i_two = _two_level_structure(
        i_values
    )

    q_two = _two_level_structure(
        q_values
    )

    return {

        # Existing names
        "axis_ratio": float(axis_ratio),
        "axis_energy_ratio": float(axis_ratio),

        "i_error_2":
            float(i_info["error_2"]),

        "i_error_4":
            float(i_info["error_4"]),

        "q_error_2":
            float(q_info["error_2"]),

        "q_error_4":
            float(q_info["error_4"]),

        # Four-level information
        "i_levels":
            i_info["levels"],

        "q_levels":
            q_info["levels"],

        "i_centers":
            i_info["centers"],

        "q_centers":
            q_info["centers"],

        "i_counts":
            i_info["counts"],

        "q_counts":
            q_info["counts"],

        "i_occupancy":
            i_info["occupancy"],

        "q_occupancy":
            q_info["occupancy"],

        "i_separation":
            i_info["separation"],

        "q_separation":
            q_info["separation"],

        "i_spacing_ratio":
            i_info["level_spacing_ratio"],

        "q_spacing_ratio":
            q_info["level_spacing_ratio"],

        # Two-level information
        "i_two_centers":
            i_two["centers"],

        "q_two_centers":
            q_two["centers"],

        "i_two_occupancy":
            i_two["occupancy"],

        "q_two_occupancy":
            q_two["occupancy"],

        "i_two_symmetry":
            i_two["symmetry"],

        "q_two_symmetry":
            q_two["symmetry"],
    }


def classify_from_constellation(symbols):
    """
    Classify BPSK, QPSK or 16-QAM using
    constellation geometry.

    BPSK:
        Energy concentrated on one axis.

    QPSK:
        Two symmetric levels on I and Q.

    16-QAM:
        Four distinct levels on I and Q.
    """

    features = extract_constellation_features(
        symbols
    )

    axis_ratio = features[
        "axis_ratio"
    ]

    # =========================================================
    # BPSK
    # =========================================================

    if axis_ratio < 0.20:

        return "BPSK", features

    # =========================================================
    # 16-QAM
    # =========================================================

    i_four_levels = (
        features["i_levels"] == 4
    )

    q_four_levels = (
        features["q_levels"] == 4
    )

    i_populated = (
        len(features["i_occupancy"]) == 4
        and np.all(
            features["i_occupancy"] > 0.05
        )
    )

    q_populated = (
        len(features["q_occupancy"]) == 4
        and np.all(
            features["q_occupancy"] > 0.05
        )
    )

    i_spacing_good = (
        features["i_spacing_ratio"] > 0.45
    )

    q_spacing_good = (
        features["q_spacing_ratio"] > 0.45
    )

    # A genuine 16-QAM constellation needs
    # four levels on both axes and the 2-level
    # model should leave significant error.

    # Compare the 2-level and 4-level models.
    #
    # In noisy QPSK, the clustering algorithm may create
    # four artificial levels, but the 2-level model is still
    # significantly better than the 4-level model.
    #
    # Genuine 16-QAM should benefit much more from the
    # four-level representation.

    i_four_level_advantage = (
        features["i_error_4"]
        / max(features["i_error_2"], 1e-12)
    )

    q_four_level_advantage = (
        features["q_error_4"]
        / max(features["q_error_2"], 1e-12)
    )

    genuine_four_level_structure = (
        i_four_level_advantage < 0.25
        and q_four_level_advantage < 0.25
    )

    if (
        axis_ratio > 0.45
        and i_four_levels
        and q_four_levels
        and i_populated
        and q_populated
        and i_spacing_good
        and q_spacing_good
        and features["i_error_2"] > 0.05
        and features["q_error_2"] > 0.05
        and genuine_four_level_structure
    ):
        return "16-QAM", features

    # =========================================================
    # QPSK
    # =========================================================

    i_two_occupancy = features[
        "i_two_occupancy"
    ]

    q_two_occupancy = features[
        "q_two_occupancy"
    ]

    i_two_populated = (
        len(i_two_occupancy) == 2
        and np.all(
            i_two_occupancy > 0.20
        )
    )

    q_two_populated = (
        len(q_two_occupancy) == 2
        and np.all(
            q_two_occupancy > 0.20
        )
    )

    i_symmetry_good = (
        features["i_two_symmetry"] > 0.85
    )

    q_symmetry_good = (
        features["q_two_symmetry"] > 0.85
    )

    # IMPORTANT:
    # Do NOT require i_levels == 2 here.
    #
    # Noisy QPSK can be split into four artificial
    # clusters by the 4-level k-means model.
    #
    # Instead, the dedicated 2-level model tells us
    # whether the underlying constellation has two
    # symmetric levels.

    if (
        axis_ratio > 0.45
        and i_two_populated
        and q_two_populated
        and i_symmetry_good
        and q_symmetry_good
    ):

        return "QPSK", features

    # =========================================================
    # Unknown
    # =========================================================

    return "Unknown", features