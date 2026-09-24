"""
Configuration objects for the Spectra pipeline.

All tunable DSP parameters live here so behavior can be changed
without editing algorithm code. Configs are frozen dataclasses;
use ``dataclasses.replace`` to derive modified copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class DetectionConfig:
    """Spectral signal detection parameters."""

    threshold_db: float = 10.0
    min_bandwidth_hz: float = 1.0
    smoothing_window_bins: int = 41
    prominence_db: float = 8.0
    min_peak_distance_bins: int = 80
    merge_gap_bins: int = 20

    def __post_init__(self) -> None:
        if self.threshold_db <= 0:
            raise ConfigurationError("threshold_db must be positive")
        if self.min_bandwidth_hz <= 0:
            raise ConfigurationError("min_bandwidth_hz must be positive")
        if self.smoothing_window_bins < 3:
            raise ConfigurationError("smoothing_window_bins must be >= 3")


@dataclass(frozen=True)
class IsolationConfig:
    """Candidate isolation / DDC parameters."""

    filter_margin: float = 1.25
    filter_order: int = 6

    def __post_init__(self) -> None:
        if self.filter_margin <= 0:
            raise ConfigurationError("filter_margin must be positive")
        if self.filter_order < 1:
            raise ConfigurationError("filter_order must be >= 1")


@dataclass(frozen=True)
class SymbolRateConfig:
    """Symbol-rate estimation limits."""

    min_symbol_rate: float = 20.0
    max_symbol_rate: float | None = None  # None -> sample_rate / 4
    tolerance: float = 0.05  # relative tolerance for consistency checks

    def __post_init__(self) -> None:
        if self.min_symbol_rate <= 0:
            raise ConfigurationError("min_symbol_rate must be positive")


@dataclass(frozen=True)
class SynchronizationConfig:
    """Carrier and timing synchronization parameters."""

    modulation_order: int = 4
    estimate_symbol_rate: bool = True

    def __post_init__(self) -> None:
        if self.modulation_order < 2:
            raise ConfigurationError("modulation_order must be >= 2")


@dataclass(frozen=True)
class ClassificationConfig:
    """Classification strategy parameters."""

    confidence_threshold: float = 25.0  # below this -> Unknown
    use_constellation: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 100.0:
            raise ConfigurationError("confidence_threshold must be in [0, 100]")


@dataclass(frozen=True)
class DemodulationConfig:
    """Demodulation parameters."""

    samples_per_symbol: float | None = None
    timing_offset: float = 0.0
    freq_0: float | None = None  # BFSK tone frequencies
    freq_1: float | None = None
    synchronized: bool = False


@dataclass(frozen=True)
class FECConfig:
    """FEC configuration. FEC is never guessed; it must be configured."""

    scheme: str | None = None  # None -> no FEC
    crc: str | None = None  # "crc16", "crc32", or None
    interleave_depth: int = 1

    def __post_init__(self) -> None:
        from ..fec import list_schemes

        if self.scheme is not None and self.scheme not in list_schemes():
            raise ConfigurationError(
                f"Unknown FEC scheme '{self.scheme}'. "
                f"Available: {sorted(list_schemes())}"
            )
        if self.crc is not None and self.crc not in ("crc16", "crc32"):
            raise ConfigurationError("crc must be 'crc16', 'crc32', or None")
        if self.interleave_depth < 1:
            raise ConfigurationError("interleave_depth must be >= 1")


@dataclass(frozen=True)
class MLConfig:
    """Machine-learning classification stage (optional evidence).

    The CNN is a second opinion alongside the rule-based classifier;
    it never overrides or gates the DSP result.
    """

    enabled: bool = False
    max_frames: int = 8  # (512, 2) windows scored per capture
    artifact: str | None = None  # None -> packaged default artifact


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level pipeline configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    symbol_rate: SymbolRateConfig = field(default_factory=SymbolRateConfig)
    synchronization: SynchronizationConfig = field(
        default_factory=SynchronizationConfig
    )
    classification: ClassificationConfig = field(
        default_factory=ClassificationConfig
    )
    demodulation: DemodulationConfig = field(default_factory=DemodulationConfig)
    fec: FECConfig = field(default_factory=FECConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    candidate_index: int = 0  # which detected candidate to analyze (0 = strongest)
    max_samples: int | None = None  # truncate very long captures


def processing_mode_config(mode: str) -> AnalysisConfig:
    """
    Return an AnalysisConfig for a named processing mode.

    Modes
    -----
    quick :
        Lighter smoothing/thresholds, no advanced steps. Fastest.
    balanced :
        Default production settings.
    deep :
        Lower detection thresholds and tighter tolerances for weak
        signals. Slower.
    """
    modes = {
        "quick": AnalysisConfig(
            detection=replace(
                DetectionConfig(),
                smoothing_window_bins=21,
                prominence_db=10.0,
            ),
            classification=ClassificationConfig(confidence_threshold=35.0),
        ),
        "balanced": AnalysisConfig(),
        "deep": AnalysisConfig(
            detection=replace(
                DetectionConfig(),
                threshold_db=6.0,
                prominence_db=6.0,
            ),
            classification=ClassificationConfig(confidence_threshold=15.0),
        ),
        "realtime": AnalysisConfig(
            detection=replace(
                DetectionConfig(),
                smoothing_window_bins=15,
            ),
            classification=ClassificationConfig(confidence_threshold=40.0),
        ),
    }

    key = str(mode).lower()
    if key not in modes:
        raise ConfigurationError(
            f"Unknown processing mode '{mode}'. Available: {sorted(modes)}"
        )
    return modes[key]


def config_to_dict(config: AnalysisConfig) -> dict[str, Any]:
    """Serialize an AnalysisConfig for provenance manifests."""
    return {
        "detection": config.detection.__dict__.copy(),
        "isolation": config.isolation.__dict__.copy(),
        "symbol_rate": config.symbol_rate.__dict__.copy(),
        "synchronization": config.synchronization.__dict__.copy(),
        "classification": config.classification.__dict__.copy(),
        "demodulation": config.demodulation.__dict__.copy(),
        "fec": config.fec.__dict__.copy(),
        "ml": config.ml.__dict__.copy(),
        "candidate_index": config.candidate_index,
        "max_samples": config.max_samples,
    }
