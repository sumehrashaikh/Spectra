"""Baseline BPSK symbol decisions for recovered symbol-rate IQ samples."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BPSKDemodulation:
    """Hard-decision results for a BPSK symbol stream.

    ``bits`` uses the conventional local mapping ``+axis -> 0`` and
    ``-axis -> 1``.  A blind BPSK receiver has an unavoidable 180-degree
    polarity ambiguity, so a reference/preamble is required to establish an
    external bit polarity.
    """

    bits: np.ndarray
    aligned_symbols: np.ndarray
    carrier_phase_radians: float
    decision_values: np.ndarray
    decision_margin: float
    quadrature_ratio: float


def demodulate_bpsk(symbol_samples):
    """Phase-align recovered symbols and make one BPSK hard decision each.

    The phase estimate is derived from the second-order moment, making it
    independent of the BPSK data polarity.  It deliberately consumes only the
    caller-provided symbol samples: no samples-per-symbol or test-signal
    assumptions are introduced here.
    """
    symbols = np.asarray(symbol_samples, dtype=np.complex128).reshape(-1)
    if symbols.size == 0:
        raise ValueError("No symbol samples are available for BPSK demodulation.")

    symbols = symbols - np.mean(symbols)
    second_moment = np.mean(symbols ** 2)
    if abs(second_moment) < 1e-15:
        raise ValueError("Symbol samples do not have a recoverable BPSK axis.")

    carrier_phase = 0.5 * np.angle(second_moment)
    aligned = symbols * np.exp(-1j * carrier_phase)
    decisions = aligned.real
    bits = (decisions < 0.0).astype(np.uint8)

    axis_rms = np.sqrt(np.mean(decisions ** 2)) + 1e-12
    decision_margin = float(np.median(np.abs(decisions)) / axis_rms)
    quadrature_ratio = float(np.sqrt(np.mean(aligned.imag ** 2)) / axis_rms)

    return BPSKDemodulation(
        bits=bits,
        aligned_symbols=aligned,
        carrier_phase_radians=float(carrier_phase),
        decision_values=decisions,
        decision_margin=decision_margin,
        quadrature_ratio=quadrature_ratio,
    )
