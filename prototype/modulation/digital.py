"""
Additional digital demodulation kernels: 8-PSK and OOK/ASK.

These follow the same conventions as `modulation/demodulator.py`
(normalized symbols, hard decisions, Gray mapping) so the V2 dispatch
layer can treat them identically.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 8-PSK
# ============================================================

def psk8_decision(symbols):
    """
    Hard-decision Gray-coded 8-PSK demodulation.

    Constellation: unit-amplitude points at odd multiples of 45 degrees
    rotated so symbol k sits at angle (2k+1)*45deg - 180deg ... mapped
    with natural Gray coding around the circle:

        angle 22.5 + k*45 deg is NOT used; decision boundaries are the
        8 radial lines at k*45 degrees.

    Mapping (Gray code, counterclockwise from 0 degrees):
        symbol nearest angle 0 deg        -> 000
        nearest 45 deg                    -> 001
        nearest 90 deg                    -> 011
        nearest 135 deg                   -> 010
        nearest 180 deg                   -> 110
        nearest 225 deg                   -> 111
        nearest 270 deg                   -> 101
        nearest 315 deg                   -> 100
    """
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)

    if symbols.size == 0:
        raise ValueError("No symbols supplied.")

    # Ideal points at angles (2k+1)*pi/8, k=0..7 (between decision lines).
    ideal_angles = (2 * np.arange(8) + 1) * np.pi / 8.0
    ideal = np.exp(1j * ideal_angles)

    gray_bits = np.array(
        [
            [0, 0, 0],  # k=0, angle 22.5 deg
            [0, 0, 1],  # k=1, 67.5 deg
            [0, 1, 1],  # k=2, 112.5 deg
            [0, 1, 0],  # k=3, 157.5 deg
            [1, 1, 0],  # k=4, 202.5 deg
            [1, 1, 1],  # k=5, 247.5 deg
            [1, 0, 1],  # k=6, 292.5 deg
            [1, 0, 0],  # k=7, 337.5 deg
        ],
        dtype=np.uint8,
    )

    # Nearest ideal point by angle.
    angles = np.angle(symbols)
    diffs = np.exp(1j * (angles[:, None] - ideal_angles[None, :]))
    k_index = np.argmax(diffs.real, axis=1)

    bits = gray_bits[k_index].reshape(-1)
    decided = ideal[k_index]
    margins = np.abs(
        np.imag(symbols * np.conj(decided))
    )  # angular distance scaled by radius

    return (
        bits,
        decided,
        float(np.mean(margins)),
    )


def normalize_psk8_symbols(symbols):
    """Normalize received 8-PSK symbols to unit RMS magnitude."""
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if symbols.size == 0:
        raise ValueError("No symbols supplied.")
    rms = np.sqrt(np.mean(np.abs(symbols) ** 2))
    if rms < 1e-12:
        raise ValueError("Symbol signal has insufficient energy.")
    return symbols / rms


def demodulate_psk8(samples, samples_per_symbol, timing_offset):
    """
    Complete 8-PSK symbol extraction and demodulation.

    Phase is estimated from the eighth-power moment:
    mean(symbols**8) = -exp(j*8*phi), so psi = angle(-mean8)/8 is an
    unbiased estimate of the phase offset modulo pi/4. The residual
    pi/4 ambiguity is fundamental for blind reception of this
    constellation (it maps onto itself under 45-degree rotation);
    full resolution requires a preamble or differential encoding.
    """
    from .demodulator import extract_symbols

    symbols = extract_symbols(samples, samples_per_symbol, timing_offset)
    symbols = normalize_psk8_symbols(symbols)

    eighth_power = np.mean(symbols**8)
    if abs(eighth_power) > 1e-9:
        # The 8-PSK constellation sits at odd multiples of pi/8, so the
        # mean eighth power lands at angle 8*(pi/8) = pi for zero phase
        # offset. theta = (angle(eighth_power) - pi)/8 resolves the
        # offset only modulo pi/4 (the 8th-power ambiguity). Blind
        # receivers cannot distinguish these; we evaluate the tightest
        # cluster (residual distance to ideal points) across the 8
        # candidates and report the chosen rotation honestly.
        # Identity: every symbol sits at angle (2k+1)*pi/8 + phi, so
        #   mean(symbols**8) = -exp(j*8*phi)
        # and the unbiased folded estimate is
        #   psi = angle(-mean(symbols**8)) / 8.
        #
        # The old (angle - pi)/8 form was systematically biased: it could
        # never return a positive phase. A "tightest cluster" tie-break
        # across the 8 candidate folds is also provably useless: folding
        # to the nearest ideal point makes every cluster quality metric
        # invariant under k*pi/4 rotations (the constellation maps onto
        # itself), so all 8 candidates score identically.
        #
        # The residual pi/4 ambiguity is FUNDAMENTAL for blind 8-PSK:
        # resolving it requires a preamble or differential encoding.
        # Offsets with |phi| < pi/8 recover exactly; larger offsets
        # recover phase modulo pi/4 and the payload bits are shifted by
        # a whole Gray word. phase_estimate_rad reports the estimated
        # incoming offset (the applied correction is its negation).
        phase = float(np.angle(-eighth_power) / 8.0)
    else:
        phase = 0.0

    corrected = symbols * np.exp(-1j * phase)
    bits, decided, margin = psk8_decision(corrected)

    return {
        "symbols": symbols,
        "corrected_symbols": corrected,
        "decision_symbols": decided,
        "bits": bits,
        "phase_estimate_rad": float(phase),
        "decision_margin": margin,
        "num_symbols": int(corrected.size),
        "num_bits": int(bits.size),
    }


# ============================================================
# OOK / ASK
# ============================================================

def ook_decision(symbols):
    """
    Hard-decision OOK (2-level ASK) demodulation.

    Symbols are normalized so the 'on' level has RMS magnitude ~1;
    the decision threshold is the midpoint between the two dominant
    magnitude clusters (estimated robustly, not assumed to be 0.5).

    Returns (bits, decided_amplitudes, margin).
    """
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if symbols.size == 0:
        raise ValueError("No symbols supplied.")

    magnitudes = np.abs(symbols)

    # Two-cluster split on magnitudes (1-D k-means with sorted init).
    scale = np.std(magnitudes)
    if scale < 1e-12:
        # Constant envelope: not OOK; decide everything as 'on'.
        threshold = np.median(magnitudes) / 2.0
    else:
        centers = np.array(
            [np.percentile(magnitudes, 20), np.percentile(magnitudes, 80)]
        )
        for _ in range(50):
            labels = (
                np.abs(magnitudes[:, None] - centers[None, :]).argmin(axis=1)
            )
            for i in (0, 1):
                cluster = magnitudes[labels == i]
                if cluster.size:
                    centers[i] = cluster.mean()
            centers = np.sort(centers)
        threshold = float(centers.mean())

    bits = (magnitudes > threshold).astype(np.uint8)
    decided = np.where(bits == 1, 1.0, 0.0)

    on_levels = magnitudes[bits == 1]
    off_levels = magnitudes[bits == 0]
    if on_levels.size and off_levels.size:
        margin = float(
            (on_levels.mean() - off_levels.mean())
            / max(on_levels.std() + off_levels.std(), 1e-12)
        )
    else:
        margin = 0.0

    return bits, decided, margin


def demodulate_ook(samples, samples_per_symbol, timing_offset):
    """
    Complete OOK/ASK symbol extraction and demodulation.
    """
    from .demodulator import extract_symbols

    symbols = extract_symbols(samples, samples_per_symbol, timing_offset)

    # Align the residual phase to the dominant axis (OOK is real-valued
    # up to carrier phase): rotate by the principal component angle.
    second_moment = np.mean(symbols**2)
    if abs(second_moment) > 1e-15:
        phase = 0.5 * np.angle(second_moment)
        symbols = symbols * np.exp(-1j * phase)

    bits, decided, margin = ook_decision(symbols)

    return {
        "symbols": symbols,
        "decision_symbols": decided.astype(np.complex128),
        "bits": bits,
        "decision_margin": margin,
        "num_symbols": int(symbols.size),
        "num_bits": int(bits.size),
    }
