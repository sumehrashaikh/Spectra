"""Labeled synthetic dataset generation for the modulation CNN.

Every frame is synthesized from first principles (symbol maps,
pulse shaping, carrier offsets) and passed through the project's own
channel simulator (``simulation.channel``), so training data and the
analysis pipeline share one impairment model. Labels are therefore
exact by construction — no manual annotation, no leakage ambiguity.

The 16-class set covers the modulations the repository can synthesize
honestly, plus noise as a first-class class:

    0 BPSK    1 QPSK    2 8PSK    3 16QAM
    4 64QAM   5 PAM4    6 BFSK    7 4FSK
    8 8FSK    9 MSK    10 GMSK   11 OOK
   12 ASK4   13 AM-DSB 14 AM-SSB 15 Noise
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prototype.parameters.symbol_rate import rrc_filter
from prototype.simulation.channel import ChannelConfig, apply_channel

SAMPLE_RATE = 8000.0
# Default frame length.  ``build_dataset*`` accept an override so a wider
# temporal context (ML v3 uses 1024) is a parameter, not a fork of the
# generator; every synthesizer reads the length from its ``FrameConfig``.
FRAME_LENGTH = 512

CLASS_NAMES: list[str] = [
    "BPSK", "QPSK", "8PSK", "16QAM",
    "64QAM", "PAM4", "BFSK", "4FSK",
    "8FSK", "MSK", "GMSK", "OOK",
    "ASK4", "AM-DSB", "AM-SSB", "Noise",
]

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------
# Per-frame random configuration
# --------------------------------------------------------------

@dataclass
class FrameConfig:
    snr_db: float = 12.0
    cfo_hz: float = 0.0
    phase_offset_rad: float = 0.0
    symbol_rate: float = 100.0
    rolloff: float = 0.35
    seed: int = 0
    # Frame length in samples.  512 reproduces the historical generator
    # bit-for-bit; 1024 gives the classifier more temporal context.
    frame_length: int = FRAME_LENGTH
    # Extended channel variation (opt-in; defaults reproduce the
    # original, timing/multipath-free generator exactly).
    timing_offset_samples: int = 0
    fractional_timing_offset: float = 0.0
    multipath_taps: tuple[tuple[int, float], ...] | None = None


def _random_frame_config(rng: np.random.Generator) -> FrameConfig:
    """Randomized but physical training conditions."""

    return FrameConfig(
        snr_db=float(rng.uniform(0.0, 18.0)),
        cfo_hz=float(rng.uniform(-25.0, 25.0)),
        phase_offset_rad=float(rng.uniform(-np.pi, np.pi)),
        symbol_rate=float(rng.uniform(80.0, 160.0)),
        rolloff=float(rng.uniform(0.2, 0.5)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def _random_multipath_taps(rng: np.random.Generator) -> tuple[tuple[int, float], ...]:
    """A small, physically plausible two/three-tap multipath profile.

    Delays are drawn in the first few baseband samples and gains decay
    geometrically, matching the controlled multipath the channel
    simulator already models.
    """

    count = int(rng.integers(2, 4))
    delays = sorted(int(d) for d in rng.integers(1, 6, count - 1))
    taps: list[tuple[int, float]] = []
    gain = 1.0
    for delay in delays:
        gain *= float(rng.uniform(0.25, 0.6))
        taps.append((delay, gain))
    return tuple(taps)


def _random_frame_config_v2(
    rng: np.random.Generator,
    snr_range: tuple[float, float] = (0.0, 18.0),
    channel_variation: bool = False,
    multipath_fraction: float = 0.0,
    timing_span_samples: int = 8,
    fractional_timing: bool = False,
    frame_length: int = FRAME_LENGTH,
) -> FrameConfig:
    """Randomized conditions, optionally with timing and multipath.

    With ``channel_variation=False``, ``multipath_fraction=0`` and the
    default SNR range this is exactly ``_random_frame_config``
    (identical RNG consumption), so the historical dataset is reproduced
    unchanged.  ``channel_variation=True`` additionally randomizes the
    sample-timing offset.  Multipath is off unless ``multipath_fraction``
    is raised, because the project's intended training distribution
    (``_random_frame_config``) never included it — enabling it is an
    explicit, measurable choice rather than a silent default.

    ``timing_span_samples`` widens the integer timing offset beyond the
    original 0-7 samples.  At the default 80-160 sym/s (50-100 samples
    per symbol) 0-7 samples is well under a tenth of a symbol, so it
    barely moves the sampling instant; spanning a whole symbol period is
    what makes the classifier invariant to *where* in the symbol it
    happened to start.  ``fractional_timing`` adds the channel
    simulator's linear-interpolation sub-sample delay, so the effective
    sampling phase is continuous rather than integer-quantized.
    """

    config = _random_frame_config(rng)
    config.frame_length = int(frame_length)
    if tuple(snr_range) != (0.0, 18.0):
        config.snr_db = float(rng.uniform(snr_range[0], snr_range[1]))

    span = max(1, int(timing_span_samples))
    if channel_variation or span != 8 or fractional_timing:
        config.timing_offset_samples = int(rng.integers(0, span))
    if fractional_timing:
        config.fractional_timing_offset = float(rng.uniform(-0.5, 0.5))
    if multipath_fraction > 0.0 and rng.random() < multipath_fraction:
        config.multipath_taps = _random_multipath_taps(rng)
    return config


# --------------------------------------------------------------
# Baseband synthesizers (one per class)
# --------------------------------------------------------------

def _symbols_psk(order: int, count: int, rng):
    k = rng.integers(0, order, count)
    return np.exp(2j * np.pi * k / order)


def _symbols_qam(order: int, count: int, rng):
    side = int(np.sqrt(order))
    levels = 2 * np.arange(side) - (side - 1)
    i = rng.integers(0, side, count)
    q = rng.integers(0, side, count)
    return levels[i] + 1j * levels[q]


def _symbols_pam4(count: int, rng):
    return np.array([-3.0, -1.0, 1.0, 3.0])[rng.integers(0, 4, count)]


def _pulse_shaped(symbols, sps: int, rolloff: float):
    taps = rrc_filter(sps, rolloff=rolloff, span_symbols=8)
    upsampled = np.zeros(symbols.size * sps, dtype=complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, taps, mode="same")


def _linear_mod(kind: str, cfg: FrameConfig, rng) -> np.ndarray:
    """PSK / QAM / PAM / OOK / ASK families (memoryless maps + RRC)."""

    sps = max(1, int(round(SAMPLE_RATE / cfg.symbol_rate)))
    count = cfg.frame_length // sps + 16

    if kind == "BPSK":
        syms = np.exp(1j * np.pi * rng.integers(0, 2, count))
    elif kind == "QPSK":
        syms = _symbols_psk(4, count, rng)
    elif kind == "8PSK":
        syms = _symbols_psk(8, count, rng)
    elif kind == "16QAM":
        syms = _symbols_qam(16, count, rng)
    elif kind == "64QAM":
        syms = _symbols_qam(64, count, rng)
    elif kind == "PAM4":
        syms = _symbols_pam4(count, rng)
    elif kind == "OOK":
        syms = rng.integers(0, 2, count) * 2.0 - 0.5
    elif kind == "ASK4":
        syms = np.array([0.25, 0.6, 1.0, 1.5])[rng.integers(0, 4, count)]
    else:
        raise ValueError(kind)

    syms = syms / np.sqrt(np.mean(np.abs(syms) ** 2))
    return _pulse_shaped(syms, sps, cfg.rolloff)


def _continuous_phase(kind: str, cfg: FrameConfig, rng) -> np.ndarray:
    """FSK family + MSK/GMSK (information lives in the frequency)."""

    sps = max(1, int(round(SAMPLE_RATE / cfg.symbol_rate)))
    count = cfg.frame_length // sps + 8
    bits = rng.integers(0, 2, count)

    if kind == "BFSK":
        dev = 0.5
    elif kind == "4FSK":
        dev = 0.25
    elif kind == "8FSK":
        dev = 0.125
    elif kind in ("MSK", "GMSK"):
        dev = 0.5
    else:
        raise ValueError(kind)

    freq_index = 2.0 * bits - 1.0
    if kind == "4FSK":
        freq_index = rng.integers(-1, 3, count) * 0.5
    elif kind == "8FSK":
        freq_index = (rng.integers(0, 8, count) - 3.5) * 0.25

    freq = freq_index * dev * cfg.symbol_rate

    if kind == "GMSK":
        # Gaussian pre-filter on the frequency trajectory (BT ~ 0.3).
        sigma = 0.3 * sps / np.sqrt(2.0 * np.log(2.0))
        radius = int(4 * sigma)
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        freq = np.convolve(np.repeat(freq, sps), kernel, mode="same")[: sps * count]
        phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        return np.exp(1j * phase)[: sps * count]

    freq = np.repeat(freq, sps)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return np.exp(1j * phase)


def _analog(kind: str, cfg: FrameConfig, rng) -> np.ndarray:
    """Amplitude-modulated carriers over a slow pseudo-audio message."""

    t = np.arange(cfg.frame_length) / SAMPLE_RATE
    message = np.sin(
        2.0 * np.pi * rng.uniform(40.0, 120.0) * t + rng.uniform(0, 2 * np.pi)
    )

    if kind == "AM-DSB":
        return (1.0 + 0.7 * message) * np.exp(1j * 2.0 * np.pi * 400.0 * t)
    if kind == "AM-SSB":
        analytic = message + 1j * np.imag(
            np.fft.ifft(
                np.fft.fft(message) * (np.arange(cfg.frame_length) > 0) * 2
            )
        )
        return 0.7 * analytic * np.exp(1j * 2.0 * np.pi * 400.0 * t)
    raise ValueError(kind)


def _synthesize_baseband(class_name: str, cfg: FrameConfig,
                         rng) -> np.ndarray:
    if class_name in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
                      "PAM4", "OOK", "ASK4"):
        return _linear_mod(class_name, cfg, rng)
    if class_name in ("BFSK", "4FSK", "8FSK", "MSK", "GMSK"):
        return _continuous_phase(class_name, cfg, rng)
    if class_name in ("AM-DSB", "AM-SSB"):
        return _analog(class_name, cfg, rng)
    if class_name == "Noise":
        return (rng.standard_normal(cfg.frame_length)
                + 1j * rng.standard_normal(cfg.frame_length)) / np.sqrt(2.0)
    raise ValueError(class_name)


# --------------------------------------------------------------
# Channel + framing
# --------------------------------------------------------------

def generate_frame(
    class_name: str,
    config: FrameConfig,
) -> np.ndarray:
    """One labeled ``(frame_length, 2)`` float32 frame."""

    frame_length = int(config.frame_length)
    rng = np.random.default_rng(config.seed)
    raw = _synthesize_baseband(class_name, config, rng)

    start = int(rng.integers(0, max(1, raw.size - frame_length)))
    segment = raw[start : start + frame_length]

    output = apply_channel(
        segment,
        SAMPLE_RATE,
        ChannelConfig(
            snr_db=config.snr_db,
            frequency_offset_hz=config.cfo_hz,
            phase_offset_rad=config.phase_offset_rad,
            timing_offset_samples=config.timing_offset_samples,
            fractional_timing_offset=config.fractional_timing_offset,
            multipath_taps=config.multipath_taps,
            seed=config.seed,
        ),
    )

    segment = np.asarray(output.samples, dtype=np.complex128)[:frame_length]
    frame = np.empty((frame_length, 2), dtype=np.float32)
    frame[:, 0] = segment.real
    frame[:, 1] = segment.imag
    return frame


def build_dataset(
    frames_per_class: int,
    seed: int = 0,
    classes: list[str] | None = None,
    frame_length: int = FRAME_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(N, frame_length, 2)`` frames + integer labels."""

    classes = classes or CLASS_NAMES
    rng = np.random.default_rng(seed)

    frames = np.empty((frames_per_class * len(classes), frame_length, 2),
                      dtype=np.float32)
    labels = np.empty(frames_per_class * len(classes), dtype=np.int64)

    row = 0
    for index, class_name in enumerate(classes):
        for _ in range(frames_per_class):
            config = _random_frame_config(rng)
            config.frame_length = int(frame_length)
            frames[row] = generate_frame(class_name, config)
            labels[row] = index
            row += 1

    order = rng.permutation(row)
    return frames[order], labels[order]


def build_dataset_v2(
    frames_per_class: int,
    seed: int = 0,
    classes: list[str] | None = None,
    snr_range: tuple[float, float] = (0.0, 18.0),
    channel_variation: bool = False,
    multipath_fraction: float = 0.0,
    frame_length: int = FRAME_LENGTH,
    timing_span_samples: int = 8,
    fractional_timing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate frames plus their labels, realization ids and true SNR.

    Every frame is produced by one ``generate_frame`` call, which draws
    its own symbol sequence *and* its own channel realization — so a
    frame is a realization, and a frame-level split is leakage-safe.  The
    realization ids are handed back anyway so the split can be proved
    disjoint rather than merely assumed.

    Returns ``(frames, labels, realization_ids, snr_db)``.
    """

    classes = classes or CLASS_NAMES
    rng = np.random.default_rng(seed)

    total = frames_per_class * len(classes)
    frames = np.empty((total, frame_length, 2), dtype=np.float32)
    labels = np.empty(total, dtype=np.int64)
    realization_ids = np.empty(total, dtype=np.int64)
    snr_db = np.empty(total, dtype=np.float64)

    row = 0
    for index, class_name in enumerate(classes):
        for _ in range(frames_per_class):
            config = _random_frame_config_v2(
                rng, snr_range, channel_variation, multipath_fraction,
                timing_span_samples=timing_span_samples,
                fractional_timing=fractional_timing,
                frame_length=frame_length,
            )
            frames[row] = generate_frame(class_name, config)
            labels[row] = index
            realization_ids[row] = row
            snr_db[row] = config.snr_db
            row += 1

    order = rng.permutation(row)
    return (
        frames[order], labels[order], realization_ids[order], snr_db[order],
    )
