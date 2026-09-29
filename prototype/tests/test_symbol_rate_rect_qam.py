"""Regression: rectangular-pulse 16-QAM symbol-rate estimation.

The bundled ``data/generate_16qam_test.py`` fixture is a *sharp* (unshaped)
16-QAM capture.  Its delay-multiply spectrum peaks on a high harmonic of the
symbol rate (this capture peaks at 5x100 Hz), and the previous harmonic
search compared each subharmonic against a *shrinking* running threshold, so
it walked past the true 100 Hz down to a noise-floor bin at 62.5 Hz.  That
made the demodulated stream garbage (~0.5 BER).

These tests pin the fixed behaviour: the fundamental 100 Hz is recovered and
the pipeline recovers the payload with a clean BER.
"""

from __future__ import annotations

import numpy as np
import pytest

from prototype.parameters.symbol_rate import estimate_symbol_rate
from prototype.pipeline import analyze_samples

SAMPLE_RATE = 8000.0
SYMBOL_RATE = 100.0
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / SYMBOL_RATE)
NUM_SYMBOLS = 1000

GRAY_LEVELS = {-3, -1, 1, 3}
GRAY_MAP = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}


def _rect_qam16(seed: int = 123, snr_db: float = 18.0):
    """Sharp 16-QAM baseband capture + the transmitted bit reference."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, NUM_SYMBOLS * 4, dtype=np.uint8)

    groups = bits.reshape(NUM_SYMBOLS, 4)
    symbols = np.empty(NUM_SYMBOLS, dtype=np.complex128)
    for index, group in enumerate(groups):
        i_value = GRAY_MAP[(int(group[0]), int(group[1]))]
        q_value = GRAY_MAP[(int(group[2]), int(group[3]))]
        symbols[index] = i_value + 1j * q_value
    symbols /= np.sqrt(10.0)

    baseband = np.repeat(symbols, SAMPLES_PER_SYMBOL)

    signal_power = float(np.mean(np.abs(baseband) ** 2))
    noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10.0)) / 2.0)
    noise = noise_std * (
        rng.standard_normal(baseband.size)
        + 1j * rng.standard_normal(baseband.size)
    )
    return baseband + noise, bits


def test_rectangular_qam16_symbol_rate_is_the_fundamental():
    """Estimating on the *isolated* candidate (how the pipeline uses it)."""
    from prototype.core.isolator import isolate_signal
    from prototype.core.preprocessor import preprocess_signal
    from prototype.core.signal import Signal
    from prototype.detection.detector import detect_candidates

    samples, _ = _rect_qam16()
    signal = Signal(samples=samples, sample_rate=SAMPLE_RATE)
    processed = preprocess_signal(signal)
    candidate = detect_candidates(processed)[0]
    isolated = isolate_signal(
        processed,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
    ).signal

    estimate = estimate_symbol_rate(
        isolated.samples, SAMPLE_RATE, min_symbol_rate=20.0, max_symbol_rate=2000.0
    )
    assert estimate.symbol_rate == pytest.approx(SYMBOL_RATE, rel=0.01), (
        f"expected {SYMBOL_RATE} Hz, got {estimate.symbol_rate} Hz "
        f"({estimate.method})"
    )


def test_rectangular_qam16_pipeline_recovers_the_payload():
    samples, bits = _rect_qam16()
    result = analyze_samples(samples, SAMPLE_RATE, reference_bits=bits)
    data = result.to_dict()

    assert data["classification"]["modulation"] == "16-QAM"
    assert data["symbol_rate"]["symbol_rate_hz"] == pytest.approx(
        SYMBOL_RATE, rel=0.02
    )
    assert data["ber"] is not None
    assert data["ber"]["ber"] < 0.01, data["ber"]
    # 1000 symbols x 4 bits/symbol, minus the pulse-shaping boundary.
    assert data["demodulation"]["num_bits"] >= 3980
