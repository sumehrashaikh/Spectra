"""Correlation and synchronization-word detection utilities."""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import correlate
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for the dsp.correlation module.") from exc


def xcorr(
    a: np.ndarray,
    b: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation of ``a`` and ``b``.

    Returns (lags, correlation) where correlation[k] = sum_n a[n] * conj(b[n - k]).
    Complex inputs are supported (conjugated second argument).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        raise ValueError("xcorr inputs must be non-empty.")
    if a.size < b.size:
        a, b = b, a

    full = correlate(a, b, mode="full", method="fft")
    lags = np.arange(-(b.size - 1), a.size)

    if max_lag is not None:
        max_lag = int(max_lag)
        mask = np.abs(lags) <= max_lag
        return lags[mask], full[mask]
    return lags, full


def normalized_xcorr(
    a: np.ndarray,
    b: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding normalized cross-correlation in [-1, 1].

    Each output value is the correlation of ``b`` with the local
    segment of ``a`` normalized by both local energies, making the
    metric amplitude-independent (suitable for sync-word detection).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    lags, raw = xcorr(a, b, max_lag=max_lag)

    b_energy = np.sum(np.abs(b) ** 2)
    if b_energy <= 0:
        raise ValueError("Template b has zero energy.")

    normalized = np.zeros_like(raw, dtype=np.float64)
    offset = b.size - 1  # index of lag 0 in 'full' output

    for k, lag in enumerate(lags):
        start = lag
        end = start + b.size
        seg_start = max(start, 0)
        seg_end = min(end, a.size)
        if seg_end - seg_start < b.size:
            continue
        segment = a[seg_start:seg_end]
        seg_energy = np.sum(np.abs(segment) ** 2)
        if seg_energy <= 0:
            continue
        normalized[k] = np.abs(raw[k]) / np.sqrt(seg_energy * b_energy)

    return lags, normalized


def autocorrelation(
    samples: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Time-averaged autocorrelation |R(tau)| for lags 0..max_lag."""
    samples = np.asarray(samples)
    if samples.size < max_lag + 1:
        raise ValueError("Signal too short for the requested max_lag.")
    lags, values = xcorr(samples, samples, max_lag=max_lag)
    order = np.argsort(lags)
    lags = lags[order]
    values = values[order]
    n = samples.size
    counts = n - np.abs(lags)
    return np.abs(values) / np.maximum(counts, 1)


def find_sync_word(
    samples: np.ndarray,
    preamble: np.ndarray,
    threshold: float = 0.6,
    max_matches: int = 16,
) -> list[dict[str, float | int]]:
    """
    Locate occurrences of a known preamble/sync sequence via normalized
    correlation.

    Returns a list of matches ``{"index", "metric", "lag"}`` sorted by
    signal position, where ``index`` is the sample index where the
    preamble starts. ``metric`` is the normalized correlation in [0, 1].
    """
    samples = np.asarray(samples)
    preamble = np.asarray(preamble)
    if preamble.size == 0:
        raise ValueError("Preamble template is empty.")
    if samples.size < preamble.size:
        return []

    lags, normalized = normalized_xcorr(samples, preamble, max_lag=None)

    usable = np.flatnonzero(normalized > 0)
    matches: list[dict[str, float | int]] = []
    if usable.size == 0:
        return []

    candidates = np.flatnonzero(normalized >= threshold)
    # Non-maximum suppression with a one-preamble-length guard band.
    guard = preamble.size
    suppressed = np.zeros(candidates.size, dtype=bool)
    order = candidates[np.argsort(normalized[candidates])[::-1]]

    for index in order:
        if suppressed[np.searchsorted(candidates, index)]:
            continue
        matches.append(
            {
                "lag": int(lags[index]),
                "index": int(lags[index]),
                "metric": float(normalized[index]),
            }
        )
        suppressed |= np.abs(candidates - index) < guard
        if len(matches) >= max_matches:
            break

    matches.sort(key=lambda m: int(m["index"]))
    return matches
