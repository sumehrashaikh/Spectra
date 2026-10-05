from dataclasses import dataclass

import numpy as np

from prototype.core.signal import Signal
from prototype.core.synchronizer import (
    SynchronizationResult as TimingSynchronizationResult,
    synchronize_signal as synchronize_timing,
)
from prototype.core.carrier_sync import (
    CarrierRecoveryResult,
    recover_carrier,
)
from prototype.parameters.symbol_rate import rrc_filter


@dataclass
class FullSynchronizationResult:
    """Complete timing + carrier synchronization result."""

    signal: Signal

    symbol_rate: float
    samples_per_symbol: float

    timing_offset: int
    timing_confidence: float

    frequency_offset: float
    frequency_confidence: float

    phase_offset: float
    phase_confidence: float


_QAM16_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)


def _qam16_grid() -> np.ndarray:

    return np.array(
        [complex(a, b) for a in _QAM16_LEVELS for b in _QAM16_LEVELS]
    )


def _normalized_lattice(lattice: np.ndarray | None) -> np.ndarray:

    """Reference constellation, scaled to unit RMS (16-QAM by default).

    Normalizing makes one score comparable across constellations whose
    ideal points carry different average power (QPSK at radius 1 vs the
    16-QAM grid at unit RMS).
    """

    grid = _qam16_grid() if lattice is None else np.asarray(
        lattice, dtype=np.complex128
    ).reshape(-1)

    rms = float(np.sqrt(np.mean(np.abs(grid) ** 2)))

    if rms < 1e-12:
        raise ValueError("lattice must carry energy")

    return grid / rms


#: Symbols used by the (phase, step) grid search scoring.  The criterion is
#: a mean over symbols, so a spread-out subset estimates it just as well: the
#: search is O(steps x phases x symbols) and a long capture would otherwise
#: make the grid search dominate the analysis.
_LATTICE_SCORE_SUBSET = 512


def lattice_fit_score(
    symbols: np.ndarray,
    lattice: np.ndarray | None = None,
) -> float:

    """Trimmed mean distance to the nearest ``lattice`` point.

    Public because callers that choose between two synchronization runs
    need the same criterion the synchronizer optimizes.
    """

    return _lattice_fit_score(symbols, lattice)


def _search_score(symbols: np.ndarray, lattice: np.ndarray) -> float:

    """Lattice-fit score of a spread-out subset, for the timing search."""

    step = max(1, int(np.ceil(symbols.size / _LATTICE_SCORE_SUBSET)))

    return _lattice_fit_score(symbols[::step], lattice)


def _lattice_fit_score(
    symbols: np.ndarray,
    lattice: np.ndarray | None = None,
) -> float:

    """Trimmed mean distance to the nearest reference-lattice point."""

    rms = float(np.sqrt(np.mean(np.abs(symbols) ** 2)))

    if rms < 1e-12:
        return float("inf")

    normalized = np.asarray(symbols) / rms

    try:
        grid = _normalized_lattice(lattice)
    except ValueError:
        return float("inf")

    distances = np.min(
        np.abs(normalized[:, None] - grid[None, :]),
        axis=1,
    )

    keep = max(1, int(0.9 * distances.size))

    return float(np.mean(np.sort(distances)[:keep]))


def _decision_directed_phase(
    symbols: np.ndarray,
    lattice: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:

    """Static decision-directed phase correction; returns (symbols, angle)."""

    grid = _normalized_lattice(lattice)

    current = symbols

    total = 0.0

    for _ in range(6):

        nearest = grid[
            np.argmin(
                np.abs(current[:, None] - grid[None, :]),
                axis=1,
            )
        ]

        product = np.mean(current * np.conj(nearest))

        if abs(product) < 1e-12:
            break

        angle = float(np.angle(product))

        current = current * np.exp(-1j * angle)
        total += angle

        if abs(angle) < 1e-3:
            break

    return current, total


def synchronize_qam_signal(
    signal: Signal,
    symbol_rate: float,
    lattice: np.ndarray | None = None,
    lattice_label: str = "16-QAM",
) -> FullSynchronizationResult:
    """
    Synchronize a 16-QAM (or other square-QAM) signal.

    ``lattice`` selects the reference constellation the timing and phase
    stages optimize against (default: the 16-QAM grid).  Passing the
    modulation's own ideal points makes the same matched-filter +
    lattice-fit treatment available to PSK: a QPSK capture whose coarse
    label was wrong otherwise keeps the ISI of an unmatched RRC filter and
    a sampling phase chosen for the wrong grid, which smears the
    recovered cloud and caps the BER at ~25%.

    The generic chain (M-th power carrier phase + magnitude-variation
    timing) is tuned for constant-envelope PSK and fails on
    amplitude-modulated constellations:

    - the M-th power phase estimate is data-dependent on QAM and
      mis-rotates the stream by an arbitrary angle;
    - the magnitude-variation timing metric has no maximum (the
      symbol magnitudes vary with the data anyway), so the timing
      origin is effectively random.

    This synchronizer instead uses two QAM-appropriate, blind
    criteria:

    1. Timing phase: mean distance to the nearest ideal 16-QAM
       grid point ("lattice fit"), searched over integer phases
       with a fractional-step refinement and fractional-interval
       symbol interpolation.
    2. Static carrier phase: decision-directed correction using
       the grid as reference (no known preamble required).

    A matched RRC filter is applied first: transmitters shape
    with a *root* raised cosine, whose zero-ISI property only
    holds after a second, identical RRC at the receiver. 16-QAM
    cannot tolerate the resulting ISI (its inner decision
    boundaries are narrower than the ISI term), unlike QPSK
    whose quadrant decisions survive it.

    Notes
    -----
    The recovered symbol stream has an arbitrary integer origin
    (which symbol is "first" is not observable blind) and an
    arbitrary 90-degree phase fold (invariant lattice). Fold and
    symbol-offset resolution against a reference bit stream is the
    BER stage's responsibility.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "synchronize_qam_signal expects a Signal object."
        )

    if symbol_rate <= 0:
        raise ValueError(
            "symbol_rate must be positive."
        )

    samples = np.asarray(
        signal.samples,
        dtype=np.complex128,
    )

    sample_rate = float(signal.sample_rate)

    # ---------------------------------------------------------
    # Stage 1: frequency handling.
    #
    # The isolation/DDC stage has normally already removed the
    # large carrier offset. The M-th power frequency estimate is
    # quantized to its FFT bin spacing (~fs/N Hz); applying it
    # when the true residual is near zero *introduces* a slow
    # rotation that corrupts the timing search and the decision-
    # directed phase stages (verified empirically). The residual,
    # if any, is instead measured decision-directed AFTER timing
    # recovery and removed there (stage 4).
    # ---------------------------------------------------------

    corrected = samples

    # ---------------------------------------------------------
    # Stage 1a: residual carrier-frequency correction.
    #
    # The isolation/DDC stage removes most of the carrier offset,
    # but its frequency grid is coarse and QAM constellation
    # geometry is far less tolerant of residual rotation than
    # PSK: even ~1 Hz over a multi-second capture accumulates
    # tens of radians and destroys the lattice. The 4th-power
    # spectrum still shows a usable tone (E[s^4] is real and
    # positive for square 16-QAM), so estimate the residual
    # there. A plain FFT bin (~0.2 Hz here) is NOT accurate
    # enough — the estimate is refined by zero-padded parabolic
    # interpolation to ~0.01 Hz, and only applied when it is
    # large enough to matter (below that, applying it would add
    # more error than it removes).
    # ---------------------------------------------------------

    n_fft = int(2 ** np.ceil(np.log2(max(1024, samples.size * 8))))

    fourth = np.asarray(samples, dtype=np.complex128) ** 4

    spectrum = np.abs(np.fft.fft(fourth, n=n_fft))

    freqs = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)

    # Search band: residual offsets are small by construction;
    # keep the window tight so noise/lattice sidelobes cannot win.

    search_hz = max(10.0, 0.02 * sample_rate)

    band = np.abs(freqs) <= search_hz

    if not np.any(band):

        freq_offset = 0.0

    else:

        band_idx = np.flatnonzero(band)

        peak = band_idx[np.argmax(spectrum[band_idx])]

        if 0 < peak < n_fft - 1:

            alpha, beta, gamma = (
                spectrum[peak - 1],
                spectrum[peak],
                spectrum[peak + 1],
            )

            denom = alpha - 2.0 * beta + gamma

            delta = (
                float(np.clip(0.5 * (alpha - gamma) / denom, -0.5, 0.5))
                if abs(denom) > 1e-30
                else 0.0
            )

        else:

            delta = 0.0

        fourth_power_offset = float(freqs[peak] + delta * (freqs[1] - freqs[0]))

        freq_offset = fourth_power_offset / 4.0

        # Deadband: below ~0.02 Hz the total rotation over the
        # capture is under ~0.6 rad and timing/DD stages absorb it.

        if abs(freq_offset) < 0.02:

            freq_offset = 0.0

        else:

            t = np.arange(samples.size) / sample_rate

            corrected = samples * np.exp(
                -1j * 2.0 * np.pi * freq_offset * t
            )

    # ---------------------------------------------------------
    # Stage 1b: matched RRC filter.
    #
    # TX pulse shaping uses a root raised cosine, whose zero-ISI
    # property only holds after convolution with a second,
    # identical RRC at the receiver. Without it, each symbol
    # carries ~8% ISI from its neighbors (h(T)/h(0) for beta=0.35).
    # QPSK tolerates that ISI (wide quadrant decisions), but
    # 16-QAM's inner decision boundaries (+/-0.316 normalized) are
    # narrower than the ISI term, which caps BER near 50% no
    # matter how accurate timing and carrier recovery are.
    # ---------------------------------------------------------

    reference_lattice = _normalized_lattice(lattice)

    sps_f = sample_rate / float(symbol_rate)

    if sps_f >= 8.0:

        taps = rrc_filter(
            int(round(sps_f)),
            rolloff=0.35,
            span_symbols=8,
        )

        # Convolve the (already frequency-corrected) signal, not the
        # raw samples: this stage must not discard stage 1a.
        corrected = np.convolve(corrected, taps, mode="same")

        # Group delay of a symmetric 'same'-mode convolution is
        # zero: the output is aligned with the input samples.

    # ---------------------------------------------------------
    # Stage 2: joint (phase, step) grid search on the lattice-fit
    # criterion.

    n_symbols = int(corrected.size / sps_f) - 2

    if n_symbols < 32:
        raise ValueError(
            "Capture too short for QAM synchronization."
        )

    def symbols_at(phase: float, step: float) -> np.ndarray:

        times = phase + np.arange(n_symbols) * step

        times = times[(times >= 0) & (times < corrected.size - 1)]

        idx = np.floor(times).astype(int)

        frac = times - idx

        return (
            corrected[idx] * (1.0 - frac)
            + corrected[idx + 1] * frac
        )

    # ---------------------------------------------------------
    # Stage 2: joint (phase, step) grid search on the lattice-fit
    # criterion.
    #
    # The cyclostationary rate estimate is not precise enough to
    # slice the whole capture with a single step: a 0.03% error
    # drifts several samples over 500 symbols and turns into ISI
    # at the capture tail. Correcting the step with per-segment
    # phase slopes was evaluated and rejected: the fit curve near
    # the true phase is shallow, so per-segment argmax picks are
    # noisy and fabricate slopes. A joint 2-D search over integer
    # phases x fractional steps is unambiguous: at the true
    # (phase, step) the whole capture aligns and the fit score is
    # sharply lower.
    # ---------------------------------------------------------

    sps_int = max(2, int(round(sps_f)))

    # Step candidates: sps_f plus fractional offsets covering the
    # realistic rate-estimate error band (+/-0.1%), plus the
    # rounded integer step.
    step_candidates = sorted(
        set(
            [float(sps_int)]
            + [
                round(sps_f * (1.0 + delta), 6)
                for delta in np.arange(-0.001, 0.00101, 0.0001)
            ]
        )
    )

    best_phase = 0.0
    best_step = sps_f
    best_score = float("inf")

    for step in step_candidates:

        for phase in range(sps_int):

            score = _search_score(
                symbols_at(float(phase), step), reference_lattice
            )

            if score < best_score:
                best_score = score
                best_phase = float(phase)
                best_step = step

    # Fractional phase refinement at the winning step.

    for d_phase in np.arange(-0.5, 0.51, 0.1):

        score = _search_score(
            symbols_at(best_phase + d_phase, best_step), reference_lattice
        )

        if score < best_score:
            best_score = score
            best_phase = best_phase + d_phase

    raw_symbols = symbols_at(best_phase, best_step)

    # ---------------------------------------------------------
    # Stage 3: decision-directed static phase correction
    # ---------------------------------------------------------

    rms = float(np.sqrt(np.mean(np.abs(raw_symbols) ** 2)))

    normalized = raw_symbols / rms

    phase_corrected, dd_angle = _decision_directed_phase(
        normalized, reference_lattice
    )

    # ---------------------------------------------------------
    # Stage 4: skipped. A decision-directed residual-frequency
    # tracker was evaluated here and removed: with the timing
    # search above already compensating rate error, the tracker
    # had no real drift to follow and only added noise.
    # ---------------------------------------------------------

    synchronized_signal = Signal(
        samples=phase_corrected * rms,
        sample_rate=sample_rate,
        metadata=signal.metadata.copy(),
    )

    # Timing confidence: how sharply the best phase beats the
    # median phase (0 = no discrimination, 100 = sharp peak).

    all_scores = [
        _search_score(symbols_at(float(p), best_step), reference_lattice)
        for p in range(int(np.ceil(sps_f)))
    ]

    median_score = float(np.median(all_scores))

    timing_confidence = (
        min(
            100.0,
            max(
                0.0,
                100.0
                * (median_score - best_score)
                / median_score,
            ),
        )
        if median_score > 1e-12
        else 0.0
    )

    synchronized_signal.add_metadata(
        synchronization="qam_lattice_fit",
        modulation_order=16,
        lattice=lattice_label,
        lattice_points=int(reference_lattice.size),
        frequency_offset=float(freq_offset),
        frequency_confidence=100.0 if freq_offset != 0.0 else 0.0,
        phase_offset=float(dd_angle),
        phase_confidence=100.0 if abs(dd_angle) > 1e-6 else 0.0,
        timing_offset=int(round(best_phase)),
        timing_confidence=float(timing_confidence),
        lattice_fit_score=float(best_score),
    )

    return FullSynchronizationResult(
        signal=synchronized_signal,
        symbol_rate=float(symbol_rate),
        samples_per_symbol=float(best_step),
        timing_offset=int(round(best_phase)),
        timing_confidence=float(timing_confidence),
        frequency_offset=float(freq_offset),
        frequency_confidence=100.0 if freq_offset != 0.0 else 0.0,
        phase_offset=float(dd_angle),
        phase_confidence=100.0 if abs(dd_angle) > 1e-6 else 0.0,
    )


def synchronize_signal(
    signal: Signal,
    symbol_rate: float,
    modulation_order: int = 4,
) -> FullSynchronizationResult:
    """
    Perform the complete Spectra synchronization pipeline.

    Carrier recovery is performed BEFORE timing recovery.

    This is important because timing recovery reduces the
    sampling rate to the symbol rate. Performing carrier
    recovery after that reduction can cause large frequency
    offsets to alias.

    Pipeline:

        Input Signal
             ↓
        Carrier/Frequency Recovery
             ↓
        Phase Recovery
             ↓
        Timing Recovery
             ↓
        One sample / symbol
             ↓
        Synchronized Signal
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "synchronize_signal expects a Signal object."
        )

    if symbol_rate <= 0:
        raise ValueError(
            "symbol_rate must be positive."
        )

    if modulation_order < 2:
        raise ValueError(
            "modulation_order must be at least 2."
        )

    # ---------------------------------------------------------
    # Stage 1: Carrier and phase synchronization
    # ---------------------------------------------------------

    carrier_result: CarrierRecoveryResult = (
        recover_carrier(
            signal,
            modulation_order=modulation_order,
        )
    )

    carrier_signal = carrier_result.signal

    # ---------------------------------------------------------
    # Stage 2: Timing synchronization
    # ---------------------------------------------------------

    timing_result: TimingSynchronizationResult = (
        synchronize_timing(
            carrier_signal,
            symbol_rate=symbol_rate,
        )
    )

    synchronized_signal = timing_result.signal

    # ---------------------------------------------------------
    # Preserve complete synchronization metadata
    # ---------------------------------------------------------

    synchronized_signal.add_metadata(
        synchronization="carrier_and_timing",

        symbol_rate=float(
            symbol_rate
        ),

        samples_per_symbol=float(
            timing_result.samples_per_symbol
        ),

        timing_offset=int(
            timing_result.timing_offset
        ),

        timing_confidence=float(
            timing_result.timing_confidence
        ),

        frequency_offset=float(
            carrier_result.frequency_offset
        ),

        frequency_confidence=float(
            carrier_result.frequency_confidence
        ),

        phase_offset=float(
            carrier_result.phase_offset
        ),

        phase_confidence=float(
            carrier_result.phase_confidence
        ),
    )

    return FullSynchronizationResult(
        signal=synchronized_signal,

        symbol_rate=float(
            symbol_rate
        ),

        samples_per_symbol=float(
            timing_result.samples_per_symbol
        ),

        timing_offset=int(
            timing_result.timing_offset
        ),

        timing_confidence=float(
            timing_result.timing_confidence
        ),

        frequency_offset=float(
            carrier_result.frequency_offset
        ),

        frequency_confidence=float(
            carrier_result.frequency_confidence
        ),

        phase_offset=float(
            carrier_result.phase_offset
        ),

        phase_confidence=float(
            carrier_result.phase_confidence
        ),
    )