"""Configuration objects for the GNU Radio v1 acquisition layer.

These describe what a GNU Radio source *is* and what it *provides*.
Every GNU Radio node is mapped onto this metadata so the rest of
Spectra (detection, classification, protocol, reporting) treats it
uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prototype.core.exceptions import ConfigurationError

#: Acquisitions come from one of three source kinds.


@dataclass(frozen=True)
class SourceMetadata:
    """Where a chunk came from and what it represents.

    Distinguish ``known`` (the value was given by the source), ``unknown``
    (not reported), and ``not_applicable`` (the field does not apply to
    this source).  Every non-``not_applicable`` value is a number/string
    that the downstream pipeline may act upon.
    """

    source: SourceKind
    file_path: str | None = None  # only for source == "file"
    center_frequency_hz: float | None = None  # known / unknown / n/a
    sample_rate: float | None = None  # known
    iq_format: str = "complex"  # complex64 / complex128 / iq_pair
    gain_db: float | None = None  # known / unknown / n/a
    timestamp: float | None = None  # acquisition start (unix epoch)
    chunk_index: int = 0
    num_chunks: int = 0
    device_name: str | None = None  # hardware only
    device_address: str | None = None  # hardware only
    source_config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "file_path": self.file_path,
            "center_frequency_hz": self.center_frequency_hz,
            "sample_rate": self.sample_rate,
            "iq_format": self.iq_format,
            "gain_db": self.gain_db,
            "timestamp": self.timestamp,
            "chunk_index": self.chunk_index,
            "num_chunks": self.num_chunks,
            "device_name": self.device_name,
            "device_address": self.device_address,
            "source_config": dict(self.source_config),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GNURadioSourceConfig:
    """Shape of a GNU Radio *source node* (synthetic or hardware).

    Parameters are deliberately minimal.  Hardware backends (RTL-SDR,
    HackRF, USRP) subclass this and provide the missing fields.
    """

    center_frequency_hz: float | None = None
    sample_rate: float = 1_000_000.0
    iq_format: str = "complex"
    gain_db: float | None = None
    device_name: str | None = None
    device_address: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ConfigurationError(
                f"source sample_rate must be positive, got {self.sample_rate}"
            )
        if self.iq_format not in ("complex", "iq_pair"):
            raise ConfigurationError(
                f"iq_format must be 'complex' or 'iq_pair', got {self.iq_format!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_frequency_hz": self.center_frequency_hz,
            "sample_rate": self.sample_rate,
            "iq_format": self.iq_format,
            "gain_db": self.gain_db,
            "device_name": self.device_name,
            "device_address": self.device_address,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class GNURadioAcquisitionConfig:
    """Top-level GNU Radio v1 acquisition/streaming configuration.

    This is what a GUI source selector or a CLI ``spectra stream``
    command configures.  It describes the GNU Radio source to run and
    how to consume its output.
    """

    source: GNURadioSourceConfig
    chunk_size: int = 1 << 20  # 1 M complex samples per buffer
    max_chunks: int = 0  # 0 -> infinite; else cap the stream
    start_time: float | None = None  # unix epoch; None -> now
    source_file: str | None = None  # synthetic replay file path
    notes: str = ""

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ConfigurationError(
                f"chunk_size must be positive, got {self.chunk_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "chunk_size": self.chunk_size,
            "max_chunks": self.max_chunks,
            "start_time": self.start_time,
            "source_file": str(self.source_file) if self.source_file else None,
            "notes": self.notes,
        }
