"""Non-data-aided symbol-timing recovery helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingRecovery:
    """Timing information needed by a later symbol sampler/demodulator."""

    estimated_sps: float
    symbol_rate: float
    confidence: float
    timing_offset: int
    eye_opening: float

    def sample_indices(self, sample_count):
        """Return valid symbol-center indices for a recording length."""
        return np.arange(self.timing_offset, sample_count, int(self.estimated_sps))


def _principal_bpsk_axis(samples):
    """Project IQ onto its highest-energy real axis without assuming I or Q."""
    x = np.asarray(samples)
    if not np.iscomplexobj(x):
        return np.asarray(x, dtype=float)
    iq = np.column_stack((x.real, x.imag))
    iq -= np.mean(iq, axis=0)
    _, _, vectors = np.linalg.svd(iq, full_matrices=False)
    return iq @ vectors[0]


def recover_symbol_timing(samples, sample_rate, min_sps=2, max_sps=None):
    """Recover a symbol period and its best sampling phase.

    Folded envelope and transition-energy templates provide a non-data-aided
    clock metric. True timing periods recur at integer multiples; choosing the
    shortest strongly supported period resolves 2x/3x superperiod ambiguity.
    The phase is then chosen using a robust BPSK eye-opening metric.
    """
    if samples is None or len(samples) < 32:
        raise ValueError("Signal is too short for timing estimation.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    signal = _principal_bpsk_axis(samples)
    signal = signal - np.mean(signal)
    scale = np.std(signal)
    if scale < 1e-12:
        raise ValueError("Signal has insufficient variation.")
    signal = signal[:min(len(signal), 200000)] / scale

    smoothing_width = min(9, max(3, len(signal) // 32))
    smoothed = np.convolve(signal, np.ones(smoothing_width) / smoothing_width, mode="same")
    measures = (np.abs(smoothed), np.diff(smoothed) ** 2)
    n_samples = len(measures[1])
    min_sps = max(2, int(min_sps))
    max_sps = min(n_samples // 8 if max_sps is None else int(max_sps), n_samples // 2, 1000)
    if max_sps < min_sps:
        raise ValueError("Search range is too small for timing estimation.")

    candidates = np.arange(min_sps, max_sps + 1)
    quality = np.zeros(len(candidates), dtype=float)
    total_variances = [np.var(measure) for measure in measures]
    if max(total_variances) < 1e-15:
        raise ValueError("Signal has insufficient transition variation.")

    for i, sps in enumerate(candidates):
        component_quality = []
        for measure, total_variance in zip(measures, total_variances):
            if total_variance < 1e-15:
                component_quality.append(0.0)
                continue
            rows = len(measure) // sps
            folded = measure[:rows * sps].reshape(rows, sps)
            phase_mean = folded.mean(axis=0)
            residual = np.var(folded - phase_mean)
            explained = max(0.0, np.var(phase_mean) - residual / rows)
            component_quality.append(explained / total_variance)
        quality[i] = 0.8 * component_quality[0] + 0.2 * component_quality[1]

    harmonic_support = np.array([
        np.mean(quality[(candidates % sps) == 0]) for sps in candidates
    ])
    strongest_quality = np.max(quality)
    strongest_support = np.max(harmonic_support)
    fundamental = candidates[
        (quality >= 0.70 * strongest_quality)
        & (harmonic_support >= 0.75 * strongest_support)
    ]
    estimated_sps = int(fundamental[0]) if len(fundamental) else int(candidates[np.argmax(harmonic_support)])

    # At a BPSK symbol center |symbol| is both large and consistent. At a
    # transition it is smaller and varies with adjacent data bits.
    phase_scores = np.empty(estimated_sps, dtype=float)
    for offset in range(estimated_sps):
        magnitudes = np.abs(signal[offset::estimated_sps])
        median = np.median(magnitudes)
        mad = np.median(np.abs(magnitudes - median)) + 1e-12
        phase_scores[offset] = median / mad
    timing_offset = int(np.argmax(phase_scores))
    eye_opening = float(phase_scores[timing_offset])

    direct_quality = quality[candidates == estimated_sps][0]
    support_quality = harmonic_support[candidates == estimated_sps][0]
    period_confidence = np.sqrt(max(0.0, direct_quality / (strongest_quality + 1e-12)))
    support_confidence = support_quality / (strongest_support + 1e-12)
    sorted_eye = np.sort(phase_scores)
    eye_separation = (sorted_eye[-1] - sorted_eye[-2]) / (sorted_eye[-1] + 1e-12)
    confidence = float(np.clip(0.55 * period_confidence + 0.35 * support_confidence + 0.10 * eye_separation, 0.0, 1.0))

    return TimingRecovery(float(estimated_sps), float(sample_rate / estimated_sps), confidence, timing_offset, eye_opening)


def estimate_samples_per_symbol(samples, sample_rate, min_sps=2, max_sps=None, return_details=False):
    """Backward-compatible API; set ``return_details`` to receive the phase."""
    recovery = recover_symbol_timing(samples, sample_rate, min_sps, max_sps)
    if return_details:
        return recovery
    return recovery.estimated_sps, recovery.symbol_rate, recovery.confidence


def sample_symbols(samples, timing):
    """Extract symbol-center IQ samples from a :class:`TimingRecovery`."""
    return np.asarray(samples)[timing.sample_indices(len(samples))]
