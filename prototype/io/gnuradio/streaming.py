"""Streaming/chunk abstraction for the GNU Radio v1 layer.

GNU Radio v1 is an acquisition/streaming layer.  Rather than assuming a
capture already lives in memory, the adapter exposes a *stream* of fixed
size chunks.  Downstream consumers (and ultimately ``analyze_samples``)
replay the stream sample by sample.

The contract is deliberately small:

- a chunk is a ``(n, 2)`` float32 array of interleaved I/Q pairs
  (``(n,)`` complex128 is also accepted and converted), or a numpy array
  of complex samples,
- every chunk carries metadata: sample rate, center frequency, gain,
  timestamp, chunk index and source metadata,
- the streamer is responsible for total sample order (chunk 0, then
  chunk 1, ...), never interleaved reinterpret.

Nothing here couples to GNU Radio itself, so the chunking semantics can
be shared between a GNU Radio flowgraph, a synthetic offline generator,
and, in the future, real SDR hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .config import GNURadioAcquisitionConfig, SourceMetadata

logger = logging.getLogger("spectra.io.gnuradio")


@dataclass(frozen=True)
class ChunkMetadata:
    """Provenance metadata attached to every chunk.

    This is the GNU Radio layer's analog of a ``CaptureMetadata``: it
    records exactly what the source told us, so the downstream
    ``Signal`` built from these samples preserves it.
    """

    sample_rate: float
    center_frequency_hz: float | None
    gain_db: float | None
    timestamp: float
    chunk_index: int
    num_chunks: int
    source_kind: str
    source_config: dict[str, Any]
    device_name: str | None = None
    file_name: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "center_frequency_hz": self.center_frequency_hz,
            "gain_db": self.gain_db,
            "timestamp": self.timestamp,
            "chunk_index": self.chunk_index,
            "num_chunks": self.num_chunks,
            "source_kind": self.source_kind,
            "device_name": self.device_name,
            "source_config": dict(self.source_config),
            "file_name": self.file_name,
            "notes": self.notes,
        }


@dataclass
class Chunk:
    """One buffer of IQ samples plus its provenance metadata.

    Samples are ``(n, 2)`` ``float32`` interleaved I/Q, or ``(n,)``
    ``complex128``.  Both orientations are accepted on input; on
    conversion to a ``Signal`` we canonicalise to complex128.

    ``as_complex`` returns a ``(n,) complex128`` view/array; ``samples_2d``
    returns ``(n, 2)`` float32 pairs.  Avoids copying large arrays unless
    the dtype conversion is lossy.
    """

    samples: np.ndarray
    metadata: ChunkMetadata

    def as_complex(self) -> np.ndarray:
        arr = np.asarray(self.samples)
        if arr.dtype.kind == "c":
            return arr.astype(np.complex128, copy=False)
        if arr.ndim == 1:
            return arr.astype(np.complex128, copy=False)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return (arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64))
        raise ValueError(f"Unexpected chunk shape {arr.shape}; expected (n,2) or (n,).")

    def samples_2d(self) -> np.ndarray:
        arr = np.asarray(self.samples)
        if arr.dtype.kind == "c":
            return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float32)
        if arr.ndim == 1:
            return np.column_stack([arr, np.zeros_like(arr)]).astype(np.float32)
        raise ValueError(f"Unexpected chunk shape {arr.shape}")


class GNURadioStreamer:
    """Produce a deterministic *GNU Radio-like* stream of IQ chunks.

    This is the GNU Radio v1 foundation.  The *real* GNU Radio integration
    lives behind the optional ``gnuradio`` python package and is isolated
    in the adapter; here we implement the stream/buffer semantics that
    both the synthetic offline flow and a future hardware backend consume.

    A typical usage for offline validation::

        from prototype.io.gnuradio import GNURadioStreamer, GNURadioAcquisitionConfig
        from prototype.io.gnuradio.config import GNURadioSourceConfig

        source_config = GNURadioSourceConfig(
            center_frequency_hz=800.0e6,
            sample_rate=8_000.0,
            iq_format="complex",
            gain_db=12.0,
            device_name="synthetic-bpsk",
        )
        acquisition = GNURadioAcquisitionConfig(
            source=source_config,
            chunk_size=1024,
            max_chunks=0,
        )
        streamer = GNURadioStreamer(acquisition)
        for chunk, meta in streamer:
            ...
    """

    def __init__(
        self,
        acquisition: GNURadioAcquisitionConfig,
        *,
        chunk_factory: (
            Callable[["GNURadioStreamer", int, int, int], Iterator[Chunk]]
            | None
        ) = None,
    ) -> None:
        self.acquisition = acquisition
        self._chunk_size = int(acquisition.chunk_size)
        self._max_chunks = int(acquisition.max_chunks)
        self._factory = chunk_factory

        source = acquisition.source
        # ChunkMetadata.source_kind is the specific source identity
        # (device name such as "synthetic-bpsk"), NEVER the category.
        # The acquisition CATEGORY is carried at capture.source_kind by
        # the adapter, so chunking must not overload source_kind.
        chunk_source_kind = source.device_name or "synthetic"
        self._metadata = ChunkMetadata(
            sample_rate=source.sample_rate,
            center_frequency_hz=source.center_frequency_hz,
            gain_db=source.gain_db,
            timestamp=(acquisition.start_time or 0.0),
            chunk_index=0,
            num_chunks=0,
            source_kind=chunk_source_kind,
            device_name=source.device_name,
            source_config=source.to_dict(),
            file_name=(
                str(Path(acquisition.source_file).name)
                if acquisition.source_file
                else None
            ),
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def metadata(self) -> ChunkMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Public stream iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[np.ndarray, ChunkMetadata]]:
        """Yield ``(chunk_samples, chunk_metadata)`` tuples.

        ``chunk_samples`` is a ``(n,) complex128`` array.  This is the
        simplest contract and the one the pipeline adapter consumes.
        """
        for chunk in self._chunks():
            yield chunk.as_complex(), chunk.metadata

    def _chunks(self) -> Iterator[tuple[Chunk, ChunkMetadata]]:
        if self._factory is not None:
            yield from self._factory(self, 0, self._max_chunks, 0)
            return
        yield from self._synthetic_chunks()

    def _synthetic_chunks(self) -> Iterator[tuple[Chunk, ChunkMetadata]]:
        """Default synthetic/offline GNU Radio-like stream.

        Produces a low-noise multi-tone signal whose center frequency and
        sample rate are configurable.  This is the "GNU Radio-compatible
        synthetic/offline flow" used to validate that GNU Radio-sourced
        samples behave identically to direct WAV/IQ samples.
        """
        from prototype.simulation.channel import ChannelConfig, apply_channel

        rate = float(self._metadata.sample_rate)
        if rate <= 0:
            raise ValueError("sample_rate must be positive")

        center = self._metadata.center_frequency_hz
        if center is None:
            center = 800.0e6

        channel_cfg = ChannelConfig(
            snr_db=14.0,
            frequency_offset_hz=0.0,
            phase_offset_rad=0.0,
            seed=1234,
        )

        num_chunks = 0
        gen = np.random.default_rng(7)

        while self._max_chunks == 0 or num_chunks < self._max_chunks:
            n = self._chunk_size
            t = np.arange(n) / rate
            signal = (
                0.5 * np.exp(2j * np.pi * (center + 230.0) * t)
                + 0.3 * np.exp(2j * np.pi * (center - 230.0) * t)
                + 0.2 * np.exp(2j * np.pi * (center + 720.0) * t)
            )
            signal *= gen.standard_normal(n) + 1j * gen.standard_normal(n)
            signal /= np.sqrt(np.mean(np.abs(signal) ** 2))

            channel_in = ChannelConfig(
                snr_db=channel_cfg.snr_db,
                frequency_offset_hz=channel_cfg.frequency_offset_hz,
                phase_offset_rad=channel_cfg.phase_offset_rad,
                seed=int(gen.integers(0, 2**31 - 1)),
            )
            output = apply_channel(signal, rate, channel_in)
            samples = np.asarray(output.samples, dtype=np.complex128)

            if samples.size == 0:
                break

            yield Chunk(
                samples=samples,
                metadata=ChunkMetadata(
                    sample_rate=rate,
                    center_frequency_hz=self._metadata.center_frequency_hz,
                    gain_db=self._metadata.gain_db,
                    timestamp=(
                        self._metadata.timestamp
                        + num_chunks * self._chunk_size / rate
                    ),
                    chunk_index=num_chunks,
                    num_chunks=0,
                    source_kind=self._metadata.source_kind,
                    source_config=dict(self._metadata.source_config),
                    file_name=self._metadata.file_name,
                    notes=(
                        f"synthetic GNU Radio v1 offline flow; SNR "
                        f"{channel_cfg.snr_db} dB"
                    ),
                ),
            )
            num_chunks += 1
