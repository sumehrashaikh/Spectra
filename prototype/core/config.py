"""
Configuration objects for the Spectra pipeline.

All tunable DSP parameters live here so behavior can be changed
without editing algorithm code. Configs are frozen dataclasses;
use ``dataclasses.replace`` to derive modified copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
from prototype.fec import interleaving as _interleaving
from prototype.fec.interleaving import deinterleave_bits
from typing import Any
from pathlib import Path

from .exceptions import ConfigurationError
from prototype.protocol.config import FrameConfig


class FECMode:
    """Enum-like mode constants for processing and FEC direction.

    Values are plain strings so they round-trip cleanly through dicts,
    JSON provenance, and GUI dropdowns.
    """

    AUTO = "auto"
    MANUAL = "manual"
    NONE = "none"

    @classmethod
    def values(cls) -> tuple[str, str, str]:
        return (cls.AUTO, cls.MANUAL, cls.NONE)


# Supported de-interleaver families (manual mode).  Each family draws its
# parameter from ``FECConfig.interleave_depth``.
INTERLEAVE_FAMILIES: tuple[str, ...] = (
    "block",
    "convolutional",
    "diagonal",
    "pseudo_random",
)

from prototype.ml.fusion import FusionConfig


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
    interleave_family: str = "block"
    mode: str = FECMode.AUTO
    interleaving_mode: str = FECMode.AUTO

    def __post_init__(self) -> None:
        from ..fec import list_schemes

        if self.interleave_family not in INTERLEAVE_FAMILIES:
            raise ConfigurationError(
                f"Unknown interleave_family '{self.interleave_family}'. "
                f"Available: {list(INTERLEAVE_FAMILIES)}"
            )

        if self.scheme is not None and self.scheme not in list_schemes():
            raise ConfigurationError(
                f"Unknown FEC scheme '{self.scheme}'. "
                f"Available: {sorted(list_schemes())}"
            )
        if self.crc is not None and self.crc not in ("crc16", "crc32"):
            raise ConfigurationError("crc must be 'crc16', 'crc32', or None")
        if self.interleave_depth < 1:
            raise ConfigurationError("interleave_depth must be >= 1")
        if self.mode not in FECMode.values():
            raise ConfigurationError(
                f"Unknown FEC mode '{self.mode}'. "
                f"Available: {sorted(FECMode.values())}"
            )
        if self.interleaving_mode not in FECMode.values():
            raise ConfigurationError(
                f"Unknown interleaving mode '{self.interleaving_mode}'. "
                f"Available: {sorted(FECMode.values())}"
            )

    def identification_runs(self) -> bool:
        """True when the automatic FEC identification step executes."""
        return self.mode == FECMode.AUTO

    def interleaving_depth(self) -> int:
        """Block-interleaving depth effective for deinterleaving."""
        return self.interleave_depth

    def interleaving_runs(self) -> bool:
        """True when automatic block-interleaving identification runs."""
        return self.interleaving_mode == FECMode.AUTO

    def deinterleave_bits(self, bits: np.ndarray) -> np.ndarray:
        """Run the stored/manual deinterleaving for the configured family.

        MANUAL: apply the configured family/depth (or seed) authoritatively.
        NONE: as-is. AUTO: as-is (the pipeline deinterleaves inline once
        identification returns AUTO_DETECTED; this helper never forces a
        depth on its own).

        The family parameter reuses ``interleave_depth`` across families:
        depth for ``block``, number of streams ``k`` for ``convolutional``,
        square width for ``diagonal`` and the permutation seed for
        ``pseudo_random``.
        """
        arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
        applicable, _ = self.manual_deinterleave_plan(int(arr.size))
        if not applicable:
            return arr
        param = int(self.interleave_depth)
        family = self.interleave_family
        if family == "convolutional":
            return _interleaving.convolutional_deinterleave(arr, k=param)
        if family == "diagonal":
            return _interleaving.diagonal_deinterleave(arr, depth=param)
        if family == "pseudo_random":
            return _interleaving.pseudo_random_deinterleave(arr, seed=param)
        # block (row-column) is the default
        return deinterleave_bits(arr, depth=param, original_size=int(arr.size))

    def manual_deinterleave_plan(self, nbits: int) -> tuple[bool, str]:
        """Whether MANUAL deinterleaving applies to an ``nbits`` stream.

        Returns ``(applicable, reason)``; ``reason`` is empty when the plan
        applies and otherwise explains why no deinterleaving was performed
        (so the pipeline can warn instead of silently passing bits through).
        """
        if self.interleaving_mode != FECMode.MANUAL:
            return False, "interleaving mode is not manual"
        param = int(self.interleave_depth)
        family = self.interleave_family
        if family == "pseudo_random":
            return (nbits > 0), ("" if nbits > 0 else "empty bitstream")
        if family == "convolutional":
            if param < 2:
                return False, "convolutional k must be >= 2"
            if nbits % param != 0:
                return False, f"{nbits} bits not divisible by convolutional k={param}"
            return True, ""
        if family == "diagonal":
            if param < 2:
                return False, "diagonal depth must be >= 2"
            if nbits != param * param:
                return False, f"{nbits} bits != diagonal depth^2={param * param}"
            return True, ""
        if param < 2:
            return False, "block depth must be >= 2"
        return True, ""


@dataclass(frozen=True)
class MLConfig:
    """Machine-learning classification stage (optional evidence).

    The CNN is a second opinion alongside the rule-based classifier;
    it never overrides or gates the DSP result.
    """

    enabled: bool = False
    max_frames: int = 8  # (512, 2) windows scored per capture
    artifact: str | None = None  # None -> packaged default artifact
    fusion: FusionConfig = field(default_factory=FusionConfig)
    labels_json: str | Path | None = None  # optional human label map


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
    protocol: "FrameConfig" | None = None  # none -> no frame analysis
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
        "ml": {**config.ml.__dict__.copy(), "fusion": config.ml.fusion.__dict__.copy()},
        "protocol": (
            config.protocol.__dict__.copy() if config.protocol is not None else None
        ),
        "candidate_index": config.candidate_index,
        "max_samples": config.max_samples,
    }
