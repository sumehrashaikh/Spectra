"""
Channel impairment simulator.

Applies a configurable chain of realistic channel impairments to
complex baseband samples and returns both the impaired samples and the
*true* parameters used, so benchmarks measure receiver errors against
known ground truth rather than assumptions.

Impairment order (documented and fixed):
    1. multipath (static taps)
    2. timing offset (integer + fractional)
    3. Doppler + carrier frequency offset (time-varying then static)
    4. phase noise (random walk)
    5. static phase offset
    6. IQ gain/phase imbalance
    7. DC offset
    8. narrowband interferer
    9. impulsive noise
    10. AWGN (per configured SNR)
    11. clipping
    12. quantization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration of the impairment chain. All values default to no-op."""

    snr_db: float | None = None
    frequency_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0
    phase_noise_deg_rms: float = 0.0
    timing_offset_samples: int = 0
    fractional_timing_offset: float = 0.0  # in (-0.5, 0.5), linear-interp delay
    multipath_taps: tuple[tuple[int, float], ...] | None = None  # (delay, gain)
    fading: str = "none"  # "none" | "rayleigh_block"
    fading_block_samples: int = 200
    doppler_rate_hz_per_s: float = 0.0
    iq_gain_imbalance_db: float = 0.0
    iq_phase_imbalance_deg: float = 0.0
    dc_offset: complex = 0j
    interferer_frequency_hz: float | None = None
    interferer_amplitude: float = 0.0
    impulsive_noise_probability: float = 0.0
    impulsive_noise_amplitude: float = 0.0
    clip_level: float | None = None
    quantization_bits: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.fading not in ("none", "rayleigh_block"):
            raise ValueError("fading must be 'none' or 'rayleigh_block'.")
        if not -0.5 <= self.fractional_timing_offset < 0.5:
            raise ValueError("fractional_timing_offset must lie in [-0.5, 0.5).")
        if self.quantization_bits is not None and self.quantization_bits < 1:
            raise ValueError("quantization_bits must be >= 1.")
        if self.fading_block_samples < 1:
            raise ValueError("fading_block_samples must be >= 1.")


@dataclass
class ChannelOutput:
    """Impaired samples plus the exact parameters that were applied."""

    samples: np.ndarray
    true_parameters: dict[str, Any] = field(default_factory=dict)


def _fractional_delay(
    samples: np.ndarray, fractional: float
) -> np.ndarray:
    """Linear-interpolation fractional delay (documented approximation)."""
    if fractional == 0.0:
        return samples
    n = np.arange(samples.size, dtype=np.float64) - fractional
    n0 = np.floor(n).astype(int)
    frac = n - n0
    valid0 = (n0 >= 0) & (n0 < samples.size)
    valid1 = (n0 + 1 >= 0) & (n0 + 1 < samples.size)
    out = np.zeros(samples.size, dtype=samples.dtype)
    out[valid0] += samples[np.clip(n0[valid0], 0, samples.size - 1)] * (1 - frac[valid0])
    out[valid1] += samples[np.clip(n0[valid1], 0, samples.size - 1)] * frac[valid1]
    return out


def _quantize(samples: np.ndarray, bits: int) -> np.ndarray:
    """Mid-rise uniform quantization of I and Q into ``bits`` levels."""
    levels = 2 ** (bits - 1)
    i = np.clip(np.round(np.real(samples) * levels), -levels, levels - 1) / levels
    q = np.clip(np.round(np.imag(samples) * levels), -levels, levels - 1) / levels
    return i + 1j * q


def apply_channel(
    samples: np.ndarray,
    sample_rate: float,
    config: ChannelConfig,
) -> ChannelOutput:
    """Apply the configured impairment chain to complex baseband samples."""
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.size == 0:
        raise ValueError("Cannot apply a channel to empty samples.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    rng = np.random.default_rng(config.seed)
    truth: dict[str, Any] = {}
    x = samples.copy()
    n_total = x.size

    # 1. Multipath ---------------------------------------------------
    if config.multipath_taps:
        max_delay = max(delay for delay, _ in config.multipath_taps)
        taps = np.zeros(max_delay + 1, dtype=np.complex128)
        for delay, gain in config.multipath_taps:
            taps[delay] += gain
        x = np.convolve(x, taps, mode="full")[: x.size]
        truth["multipath_taps"] = [tuple(t) for t in config.multipath_taps]

    # 2. Timing offset -----------------------------------------------
    if config.timing_offset_samples > 0:
        x = np.concatenate(
            [np.zeros(config.timing_offset_samples, dtype=np.complex128), x]
        )[: x.size]
        truth["timing_offset_samples"] = config.timing_offset_samples
    if config.fractional_timing_offset != 0.0:
        x = _fractional_delay(x, config.fractional_timing_offset)
        truth["fractional_timing_offset"] = config.fractional_timing_offset

    # 3. Doppler + CFO -------------------------------------------------
    t = np.arange(x.size, dtype=np.float64) / sample_rate
    if config.doppler_rate_hz_per_s != 0.0:
        phase_doppler = (
            2.0 * np.pi * config.doppler_rate_hz_per_s * t**2 / 2.0
        )
        x = x * np.exp(1j * phase_doppler)
        truth["doppler_rate_hz_per_s"] = config.doppler_rate_hz_per_s
    if config.frequency_offset_hz != 0.0:
        x = x * np.exp(
            1j * 2.0 * np.pi * config.frequency_offset_hz * t
        )
        truth["frequency_offset_hz"] = config.frequency_offset_hz

    # 4. Phase noise ---------------------------------------------------
    if config.phase_noise_deg_rms > 0.0:
        sigma = np.deg2rad(config.phase_noise_deg_rms)
        walk = np.cumsum(rng.standard_normal(x.size) * sigma / np.sqrt(sample_rate))
        x = x * np.exp(1j * walk)
        truth["phase_noise_deg_rms"] = config.phase_noise_deg_rms

    # 5. Static phase ---------------------------------------------------
    if config.phase_offset_rad != 0.0:
        x = x * np.exp(1j * config.phase_offset_rad)
        truth["phase_offset_rad"] = config.phase_offset_rad

    # 6. IQ imbalance ----------------------------------------------------
    if config.iq_gain_imbalance_db != 0.0 or config.iq_phase_imbalance_deg != 0.0:
        gain = 10.0 ** (config.iq_gain_imbalance_db / 20.0)
        theta = np.deg2rad(config.iq_phase_imbalance_deg)
        i = gain * np.real(x)
        q = np.real(x) * np.sin(theta) + np.imag(x) * np.cos(theta)
        x = i + 1j * q
        truth["iq_gain_imbalance_db"] = config.iq_gain_imbalance_db
        truth["iq_phase_imbalance_deg"] = config.iq_phase_imbalance_deg

    # 7. DC offset --------------------------------------------------------
    if config.dc_offset != 0j:
        x = x + config.dc_offset
        truth["dc_offset"] = complex(config.dc_offset)

    # 8. Interferer ---------------------------------------------------------
    if (
        config.interferer_frequency_hz is not None
        and config.interferer_amplitude > 0.0
    ):
        x = x + config.interferer_amplitude * np.exp(
            1j
            * 2.0
            * np.pi
            * config.interferer_frequency_hz
            * t
        )
        truth["interferer_frequency_hz"] = config.interferer_frequency_hz
        truth["interferer_amplitude"] = config.interferer_amplitude

    # 9. Impulsive noise ------------------------------------------------------
    if config.impulsive_noise_probability > 0.0:
        hits = rng.random(x.size) < config.impulsive_noise_probability
        bursts = (
            config.impulsive_noise_amplitude
            * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
        )
        x = x + hits * bursts
        truth["impulsive_hits"] = int(np.count_nonzero(hits))

    # 10. AWGN ------------------------------------------------------------------
    if config.snr_db is not None:
        signal_power = float(np.mean(np.abs(x) ** 2))
        noise_power = signal_power / (10.0 ** (config.snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
        )
        x = x + noise
        truth["target_snr_db"] = config.snr_db
        truth["measured_snr_db"] = float(
            10.0 * np.log10(signal_power / noise_power)
        )

    # 11. Clipping ----------------------------------------------------------------
    if config.clip_level is not None:
        magnitude = np.abs(x)
        scale = np.max(magnitude)
        if scale > config.clip_level > 0:
            x = x * (config.clip_level / scale)
            truth["clip_level"] = config.clip_level

    # 12. Quantization ---------------------------------------------------------------
    if config.quantization_bits is not None:
        x = _quantize(x, config.quantization_bits)
        truth["quantization_bits"] = config.quantization_bits

    truth["sample_rate"] = float(sample_rate)
    truth["seed"] = config.seed
    return ChannelOutput(samples=x, true_parameters=truth)
