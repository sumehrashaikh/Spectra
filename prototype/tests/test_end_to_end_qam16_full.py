import numpy as np
from scipy.signal import upfirdn

from prototype.core.signal import Signal
from prototype.pipeline import analyze_samples
from prototype.parameters.symbol_rate import rrc_filter


SAMPLE_RATE = 8000.0
SYMBOL_RATE = 100.0
SPS = int(SAMPLE_RATE / SYMBOL_RATE)
NUM_SYMBOLS = 512

# Gray mapping per axis used by modulation.demodulator.qam16_decision:
# index (b0, b1) -> level, i.e. 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(10.0)


def _make_capture(seed):
    """16-QAM burst: RRC-shaped, carrier-modulated, impaired, noisy."""

    rng = np.random.default_rng(seed)

    bits = rng.integers(0, 2, NUM_SYMBOLS * 4)

    symbols = LEVELS[bits[0::4] * 2 + bits[1::4]] + 1j * LEVELS[
        bits[2::4] * 2 + bits[3::4]
    ]

    taps = rrc_filter(SPS, rolloff=0.35, span_symbols=8)

    shaped = upfirdn(taps, symbols, up=SPS)[: NUM_SYMBOLS * SPS]

    t = np.arange(shaped.size) / SAMPLE_RATE

    iq = shaped * np.exp(1j * 2.0 * np.pi * 500.0 * t)

    # Timing offset (13 samples), residual carrier offset (37 Hz), and
    # a static phase rotation -- the standard impairment set.
    delayed = np.concatenate([np.zeros(13, dtype=np.complex128), iq])
    n = np.arange(delayed.size)
    delayed *= np.exp(1j * 2.0 * np.pi * 37.0 * n / SAMPLE_RATE)
    delayed *= np.exp(1j * np.deg2rad(30.0))
    iq = delayed

    rng = np.random.default_rng(123)
    noise = 0.004 * (
        rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)
    ) / np.sqrt(2.0)

    return iq + noise, bits


def test_end_to_end_qam16_full():
    """
    Full-pipeline 16-QAM regression.

    The pipeline must recover bit-exact data from an impaired 16-QAM
    capture using only blind synchronization. This exercises the three
    pieces the PSK paths do not need:

    - a matched RRC at the receiver (single-RRC ISI is fatal for the
      narrow 16-QAM decision boundaries),
    - sub-bin residual carrier-frequency correction (constellation
      rotation is unobservable to PSK-tuned stages but scrambles QAM),
    - a symbol-aligned fold x frame-origin search at the BER stage
      (the recovered origin is ambiguous by pulse-shaping group delay).
    """

    samples, reference_bits = _make_capture(seed=2)

    result = analyze_samples(
        samples=samples,
        sample_rate=SAMPLE_RATE,
        mode="balanced",
        reference_bits=reference_bits,
    )

    summary = result.to_dict()

    classification = summary["classification"]

    assert classification["modulation"] == "16-QAM", (
        f"expected 16-QAM classification, got "
        f"{classification['modulation']} "
        f"(confidence {classification.get('confidence')})"
    )

    ber_report = summary["ber"]

    assert ber_report is not None, "BER stage did not run"

    assert ber_report["ber"] == 0.0, (
        f"expected bit-exact recovery, got BER {ber_report['ber']} "
        f"({ber_report['bit_errors']} errors in "
        f"{ber_report['compared_bits']} bits) via "
        f"{ber_report['ambiguity_resolution']}"
    )
