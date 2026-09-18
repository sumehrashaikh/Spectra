from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signal import Signal


@dataclass
class SignalCandidate:
    """Represents one detected signal region."""

    center_frequency: float
    bandwidth: float
    peak_frequency: float
    peak_power: float
    confidence: float
    start_frequency: float
    end_frequency: float
    start_index: int
    end_index: int

    def summary(self) -> dict:
        return {
            "center_frequency": self.center_frequency,
            "bandwidth": self.bandwidth,
            "peak_frequency": self.peak_frequency,
            "peak_power": self.peak_power,
            "confidence": self.confidence,
            "start_frequency": self.start_frequency,
            "end_frequency": self.end_frequency,
        }


@dataclass
class DetectionResult:
    """Result of signal detection."""

    detected: bool
    peak_frequency: float
    bandwidth: float
    peak_power: float
    noise_power: float
    threshold: float
    confidence: float
    frequency_axis: np.ndarray
    power_spectrum: np.ndarray
    signal_mask: np.ndarray
    candidates: list[SignalCandidate] | None = None

    def summary(self) -> dict:
        return {
            "detected": self.detected,
            "peak_frequency": self.peak_frequency,
            "bandwidth": self.bandwidth,
            "peak_power": self.peak_power,
            "noise_power": self.noise_power,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "candidate_count": len(self.candidates or []),
        }


def compute_spectrum(
    signal: Signal,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a two-sided FFT power spectrum."""

    if not isinstance(signal, Signal):
        raise TypeError(
            "compute_spectrum() requires a Signal object."
        )

    samples = signal.samples
    n = signal.num_samples
    sample_rate = signal.sample_rate

    spectrum = np.fft.fftshift(
        np.fft.fft(samples)
    )

    power_spectrum = (
        np.abs(spectrum) ** 2
    ) / (n ** 2)

    frequency_axis = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1.0 / sample_rate,
        )
    )

    return frequency_axis, power_spectrum


def estimate_detection_threshold(
    power_spectrum: np.ndarray,
    threshold_db: float = 10.0,
) -> tuple[float, float]:
    """
    Estimate the spectral noise floor and detection threshold.

    The threshold is intentionally higher than the original
    6 dB value because candidate detection operates on a
    smoothed spectrum.
    """

    power_spectrum = np.asarray(
        power_spectrum,
        dtype=float,
    )

    if power_spectrum.size == 0:
        raise ValueError(
            "Power spectrum is empty."
        )

    if np.any(power_spectrum < 0):
        raise ValueError(
            "Power spectrum cannot contain negative values."
        )

    noise_power = float(
        np.median(power_spectrum)
    )

    peak_power = float(
        np.max(power_spectrum)
    )

    numerical_floor = peak_power * 1e-10

    noise_power = max(
        noise_power,
        numerical_floor,
        np.finfo(float).eps,
    )

    threshold = noise_power * (
        10.0 ** (threshold_db / 10.0)
    )

    return noise_power, float(threshold)


def smooth_spectrum(
    power_spectrum: np.ndarray,
    window_bins: int = 41,
) -> np.ndarray:
    """
    Smooth the power spectrum using a moving-average window.

    Smoothing reduces individual FFT-bin fluctuations while
    preserving broader signal envelopes.
    """

    power_spectrum = np.asarray(
        power_spectrum,
        dtype=float,
    )

    if power_spectrum.size == 0:
        raise ValueError(
            "Power spectrum is empty."
        )

    window_bins = max(
        3,
        int(window_bins),
    )

    if window_bins % 2 == 0:
        window_bins += 1

    if window_bins >= power_spectrum.size:
        window_bins = (
            power_spectrum.size
            if power_spectrum.size % 2 == 1
            else power_spectrum.size - 1
        )

    if window_bins < 3:
        return power_spectrum.copy()

    kernel = np.ones(
        window_bins,
        dtype=float,
    ) / window_bins

    return np.convolve(
        power_spectrum,
        kernel,
        mode="same",
    )


def find_signal_peaks(
    smoothed_power: np.ndarray,
    noise_power: float,
    prominence_db: float = 8.0,
    min_peak_distance_bins: int = 80,
) -> np.ndarray:
    """
    Find significant peaks in the smoothed spectrum.
    """

    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for peak detection."
        ) from exc

    prominence = (
        noise_power
        * 10.0 ** (
            prominence_db / 10.0
        )
    )

    peaks, _ = find_peaks(
        smoothed_power,
        prominence=prominence,
        distance=max(
            1,
            int(min_peak_distance_bins),
        ),
    )

    return peaks


def expand_peak_region(
    smoothed_power: np.ndarray,
    peak_index: int,
    threshold: float,
) -> tuple[int, int]:
    """
    Expand a peak through the smoothed spectral envelope
    until it falls below the detection threshold.
    """

    left = int(peak_index)

    while (
        left > 0
        and smoothed_power[left - 1] >= threshold
    ):
        left -= 1

    right = int(peak_index)

    while (
        right < len(smoothed_power) - 1
        and smoothed_power[right + 1] >= threshold
    ):
        right += 1

    return left, right


def merge_regions(
    regions: list[tuple[int, int]],
    merge_gap_bins: int = 20,
) -> list[tuple[int, int]]:
    """
    Merge nearby spectral regions belonging to the same
    signal envelope.
    """

    if not regions:
        return []

    regions = sorted(regions)

    merged: list[list[int]] = [
        [
            regions[0][0],
            regions[0][1],
        ]
    ]

    for start, end in regions[1:]:
        previous = merged[-1]

        gap = (
            start
            - previous[1]
            - 1
        )

        if gap <= merge_gap_bins:
            previous[1] = max(
                previous[1],
                end,
            )
        else:
            merged.append(
                [start, end]
            )

    return [
        (start, end)
        for start, end in merged
    ]


def calculate_confidence(
    peak_power: float,
    noise_power: float,
) -> float:
    """Calculate confidence from peak-to-noise ratio."""

    if noise_power <= 0:
        return 100.0

    ratio = peak_power / noise_power

    if ratio <= 1:
        return 0.0

    snr_db = 10.0 * np.log10(ratio)

    confidence = (
        snr_db / 30.0
    ) * 100.0

    return float(
        np.clip(
            confidence,
            0.0,
            100.0,
        )
    )


def calculate_region_metrics(
    region: np.ndarray,
    noise_power: float,
) -> tuple[float, float, float]:
    """
    Calculate energy-related metrics for a candidate region.

    Returns:

        region_energy:
            Total spectral power contained in the region.

        energy_ratio:
            Region energy relative to the estimated noise
            power per FFT bin.

        mean_power_ratio:
            Average region power relative to the noise floor.
    """

    if region.size == 0:
        return 0.0, 0.0, 0.0

    region_energy = float(
        np.sum(region)
    )

    safe_noise = max(
        noise_power,
        np.finfo(float).eps,
    )

    energy_ratio = (
        region_energy / safe_noise
    )

    mean_power = (
        region_energy / region.size
    )

    mean_power_ratio = (
        mean_power / safe_noise
    )

    return (
        region_energy,
        energy_ratio,
        mean_power_ratio,
    )


def candidate_has_meaningful_energy(
    region: np.ndarray,
    noise_power: float,
    min_energy_ratio: float = 20.0,
    min_mean_power_ratio: float = 1.5,
) -> bool:
    """
    Determine whether a candidate contains meaningful spectral energy.

    A candidate must contain sufficient total energy and its
    average power must remain above the estimated noise floor.
    """

    (
        _region_energy,
        energy_ratio,
        mean_power_ratio,
    ) = calculate_region_metrics(
        region,
        noise_power,
    )

    if energy_ratio < min_energy_ratio:
        return False

    if mean_power_ratio < min_mean_power_ratio:
        return False

    return True


def candidates_overlap(
    first: SignalCandidate,
    second: SignalCandidate,
    overlap_ratio: float = 0.50,
) -> bool:
    """
    Determine whether two candidates overlap substantially.
    """

    first_start = first.start_frequency
    first_end = first.end_frequency

    second_start = second.start_frequency
    second_end = second.end_frequency

    overlap_start = max(
        first_start,
        second_start,
    )

    overlap_end = min(
        first_end,
        second_end,
    )

    overlap = (
        overlap_end - overlap_start
    )

    if overlap <= 0:
        return False

    first_width = max(
        first_end - first_start,
        np.finfo(float).eps,
    )

    second_width = max(
        second_end - second_start,
        np.finfo(float).eps,
    )

    first_overlap_ratio = (
        overlap / first_width
    )

    second_overlap_ratio = (
        overlap / second_width
    )

    return (
        first_overlap_ratio >= overlap_ratio
        or second_overlap_ratio >= overlap_ratio
    )


def is_symmetric_spectral_artifact(
    candidate: SignalCandidate,
    stronger: SignalCandidate,
    candidates: list[SignalCandidate],
    symmetry_tolerance: float = 0.08,
    max_relative_power: float = 0.20,
    max_distance_multiplier: float = 1.75,
) -> bool:
    """
    Determine whether a candidate is likely a symmetric spectral
    artifact generated by a stronger signal.

    The detector specifically looks for a pair of weak candidates
    positioned approximately symmetrically around a stronger
    candidate.

    Example:

        weak A       strong signal       weak B
           |               |                |
         -86.5            500            1086.5

    Such paired lobes can occur around digitally modulated
    signals and should not automatically be interpreted as
    independent transmissions.

    A candidate is considered an artifact only when:

    1. It is substantially weaker than the stronger candidate.
    2. Another candidate exists on the opposite side.
    3. The two weak candidates are approximately symmetric.
    4. Their powers are reasonably similar.
    5. Their frequency separation is related to the stronger
       signal's bandwidth.
    """

    if stronger.peak_power <= 0:
        return False

    relative_power = (
        candidate.peak_power
        / stronger.peak_power
    )

    if relative_power > max_relative_power:
        return False

    distance = abs(
        candidate.center_frequency
        - stronger.center_frequency
    )

    if distance <= 0:
        return False

    maximum_distance = (
        max(
            stronger.bandwidth,
            np.finfo(float).eps,
        )
        * max_distance_multiplier
    )

    if distance > maximum_distance:
        return False

    for other in candidates:
        if other is candidate:
            continue

        if other is stronger:
            continue

        other_distance = (
            other.center_frequency
            - stronger.center_frequency
        )

        if other_distance == 0:
            continue

        # The candidate and the other candidate must lie
        # on opposite sides of the stronger signal.
        if (
            candidate.center_frequency
            - stronger.center_frequency
        ) * other_distance >= 0:
            continue

        # Compare their distances from the stronger signal.
        distance_difference = abs(
            abs(
                candidate.center_frequency
                - stronger.center_frequency
            )
            - abs(
                other.center_frequency
                - stronger.center_frequency
            )
        )

        average_distance = (
            abs(
                candidate.center_frequency
                - stronger.center_frequency
            )
            + abs(
                other.center_frequency
                - stronger.center_frequency
            )
        ) / 2.0

        if average_distance <= 0:
            continue

        symmetry_error = (
            distance_difference
            / average_distance
        )

        if symmetry_error > symmetry_tolerance:
            continue

        # The two weak lobes should have comparable power.
        power_ratio = (
            min(
                candidate.peak_power,
                other.peak_power,
            )
            / max(
                candidate.peak_power,
                other.peak_power,
            )
        )

        if power_ratio < 0.60:
            continue

        return True

    return False


def suppress_spectral_artifacts(
    candidates: list[SignalCandidate],
    power_spectrum: np.ndarray,
    noise_power: float,
    artifact_relative_power: float = 0.20,
) -> list[SignalCandidate]:
    """
    Suppress likely spectral artifacts while preserving
    independently located signals.

    Three mechanisms are used:

    1. Substantial overlap with a stronger candidate.
    2. Symmetric weak spectral pairs around a stronger candidate.
    3. Extremely weak candidates below the configured
       artifact-relative-power level when they are also
       spectrally related to stronger candidates.

    A separated candidate is preserved even when it is much
    weaker than the strongest signal.
    """

    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda candidate: candidate.peak_power,
        reverse=True,
    )

    kept: list[SignalCandidate] = []

    for candidate in ordered:
        reject = False

        for stronger in kept:
            if stronger.peak_power <= 0:
                continue

            relative_power = (
                candidate.peak_power
                / stronger.peak_power
            )

            # -------------------------------------------------
            # Case 1:
            # Candidate substantially overlaps a stronger
            # spectral region.
            # -------------------------------------------------
            if (
                relative_power < artifact_relative_power
                and candidates_overlap(
                    candidate,
                    stronger,
                    overlap_ratio=0.50,
                )
            ):
                reject = True
                break

            # -------------------------------------------------
            # Case 2:
            # Candidate forms a symmetric weak pair around
            # a stronger signal.
            # -------------------------------------------------
            if is_symmetric_spectral_artifact(
                candidate=candidate,
                stronger=stronger,
                candidates=ordered,
            ):
                reject = True
                break

        if not reject:
            kept.append(candidate)

    return kept


def create_candidates(
    frequency_axis: np.ndarray,
    power_spectrum: np.ndarray,
    noise_power: float,
    threshold: float,
    min_bandwidth_hz: float = 1.0,
    smoothing_window_bins: int = 41,
    prominence_db: float = 8.0,
    min_peak_distance_bins: int = 80,
    merge_gap_bins: int = 20,
) -> list[SignalCandidate]:
    """
    Detect multiple spectral signal candidates.

    Detection is performed on a smoothed spectral envelope,
    while candidate measurements are taken from the original
    spectrum.

    Candidate validation uses regional energy and spectral
    relationships rather than a single global peak-power
    threshold.
    """

    frequency_axis = np.asarray(
        frequency_axis,
        dtype=float,
    )

    power_spectrum = np.asarray(
        power_spectrum,
        dtype=float,
    )

    if len(frequency_axis) != len(power_spectrum):
        raise ValueError(
            "Frequency axis and power spectrum must have "
            "the same length."
        )

    if len(frequency_axis) < 2:
        return []

    frequency_resolution = float(
        abs(
            frequency_axis[1]
            - frequency_axis[0]
        )
    )

    smoothed_power = smooth_spectrum(
        power_spectrum,
        window_bins=smoothing_window_bins,
    )

    peaks = find_signal_peaks(
        smoothed_power,
        noise_power,
        prominence_db=prominence_db,
        min_peak_distance_bins=min_peak_distance_bins,
    )

    raw_regions: list[tuple[int, int]] = []

    for peak_index in peaks:
        start_index, end_index = (
            expand_peak_region(
                smoothed_power,
                int(peak_index),
                threshold,
            )
        )

        raw_regions.append(
            (
                start_index,
                end_index,
            )
        )

    regions = merge_regions(
        raw_regions,
        merge_gap_bins=merge_gap_bins,
    )

    candidates: list[SignalCandidate] = []

    for start_index, end_index in regions:

        region = power_spectrum[
            start_index:end_index + 1
        ]

        if region.size == 0:
            continue

        peak_offset = int(
            np.argmax(region)
        )

        peak_index = (
            start_index
            + peak_offset
        )

        peak_power = float(
            power_spectrum[peak_index]
        )

        peak_frequency = float(
            frequency_axis[peak_index]
        )

        start_frequency = float(
            frequency_axis[start_index]
        )

        end_frequency = float(
            frequency_axis[end_index]
        )

        bandwidth = (
            end_frequency
            - start_frequency
        )

        if bandwidth <= 0:
            bandwidth = frequency_resolution

        if bandwidth < min_bandwidth_hz:
            continue

        (
            _region_energy,
            energy_ratio,
            mean_power_ratio,
        ) = calculate_region_metrics(
            region,
            noise_power,
        )

        if energy_ratio < 20.0:
            continue

        if mean_power_ratio < 1.5:
            continue

        center_frequency = (
            start_frequency
            + end_frequency
        ) / 2.0

        confidence = calculate_confidence(
            peak_power,
            noise_power,
        )

        candidates.append(
            SignalCandidate(
                center_frequency=float(
                    center_frequency
                ),
                bandwidth=float(
                    bandwidth
                ),
                peak_frequency=float(
                    peak_frequency
                ),
                peak_power=float(
                    peak_power
                ),
                confidence=float(
                    confidence
                ),
                start_frequency=float(
                    start_frequency
                ),
                end_frequency=float(
                    end_frequency
                ),
                start_index=start_index,
                end_index=end_index,
            )
        )

    candidates.sort(
        key=lambda candidate: candidate.peak_power,
        reverse=True,
    )

    candidates = suppress_spectral_artifacts(
        candidates=candidates,
        power_spectrum=power_spectrum,
        noise_power=noise_power,
        artifact_relative_power=0.20,
    )

    candidates.sort(
        key=lambda candidate: candidate.peak_power,
        reverse=True,
    )

    return candidates


def detect_candidates(
    signal: Signal,
    threshold_db: float = 10.0,
    min_bandwidth_hz: float = 1.0,
    smoothing_window_bins: int = 41,
    prominence_db: float = 8.0,
    min_peak_distance_bins: int = 80,
    merge_gap_bins: int = 20,
) -> list[SignalCandidate]:
    """Detect multiple spectral signal candidates."""

    if not isinstance(signal, Signal):
        raise TypeError(
            "detect_candidates() requires a Signal object."
        )

    frequency_axis, power_spectrum = (
        compute_spectrum(signal)
    )

    noise_power, threshold = (
        estimate_detection_threshold(
            power_spectrum,
            threshold_db=threshold_db,
        )
    )

    return create_candidates(
        frequency_axis=frequency_axis,
        power_spectrum=power_spectrum,
        noise_power=noise_power,
        threshold=threshold,
        min_bandwidth_hz=min_bandwidth_hz,
        smoothing_window_bins=smoothing_window_bins,
        prominence_db=prominence_db,
        min_peak_distance_bins=min_peak_distance_bins,
        merge_gap_bins=merge_gap_bins,
    )


def detect_signal(
    signal: Signal,
    threshold_db: float = 10.0,
    smoothing_window_bins: int = 41,
    prominence_db: float = 8.0,
    min_peak_distance_bins: int = 80,
    merge_gap_bins: int = 20,
) -> DetectionResult:
    """
    Detect coherent spectral regions.

    The strongest valid candidate is used as the primary
    detection.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "detect_signal() requires a Signal object."
        )

    frequency_axis, power_spectrum = (
        compute_spectrum(signal)
    )

    noise_power, threshold = (
        estimate_detection_threshold(
            power_spectrum,
            threshold_db=threshold_db,
        )
    )

    candidates = create_candidates(
        frequency_axis=frequency_axis,
        power_spectrum=power_spectrum,
        noise_power=noise_power,
        threshold=threshold,
        smoothing_window_bins=smoothing_window_bins,
        prominence_db=prominence_db,
        min_peak_distance_bins=min_peak_distance_bins,
        merge_gap_bins=merge_gap_bins,
    )

    signal_mask = np.zeros_like(
        power_spectrum,
        dtype=bool,
    )

    for candidate in candidates:
        signal_mask[
            candidate.start_index : candidate.end_index + 1
        ] = True

    if not candidates:
        return DetectionResult(
            detected=False,
            peak_frequency=0.0,
            bandwidth=0.0,
            peak_power=0.0,
            noise_power=noise_power,
            threshold=threshold,
            confidence=0.0,
            frequency_axis=frequency_axis,
            power_spectrum=power_spectrum,
            signal_mask=signal_mask,
            candidates=[],
        )

    primary = candidates[0]

    return DetectionResult(
        detected=True,
        peak_frequency=primary.peak_frequency,
        bandwidth=primary.bandwidth,
        peak_power=primary.peak_power,
        noise_power=noise_power,
        threshold=threshold,
        confidence=primary.confidence,
        frequency_axis=frequency_axis,
        power_spectrum=power_spectrum,
        signal_mask=signal_mask,
        candidates=candidates,
    )