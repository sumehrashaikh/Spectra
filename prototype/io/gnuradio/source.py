"""GNU Radio v1 source abstraction.

The conceptual GNU Radio architecture for SPECTRA v1 is:

    FILE INPUT  ->  WAV / IQ Loader  ->  Signal object
                            \
                             ->  GNU Radio input -> stream/buffer -> Signal adapter -> SPECTRA pipeline

GNU Radio is OPTIONAL.  When it is not installed, ``make_gnuradio_source``
returns ``None`` and the app falls back to the ordinary loader.

Two source kinds ship in v1:

1. **Synthetic offline** -- a scriptable GNU Radio flowgraph replaced by an
   in-process generator.  Used for validation: we build the same IQ that a
   GNU Radio flowgraph would produce, expose it through the GNU Radio
   chunk/stream API, then feed it to the existing ``analyze_samples``.
2. **Hardware (future)** -- RTL-SDR / HackRF / USRP backends isolate a
   ``.read_chunk`` contract behind this same class.

The source itself does not know about the analysis pipeline.  It only
produces chunks with provenance metadata; the adapter is what turns those
chunks into the common ``Signal`` model.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from .config import GNURadioAcquisitionConfig, GNURadioSourceConfig, SourceMetadata
from .streaming import Chunk, ChunkMetadata, GNURadioStreamer

logger = logging.getLogger("spectra.io.gnuradio.source")


class GNURadioSource:  # noqa: D101 - keeps the name stable in tests
    """A GNU Radio source node (synthetic or hardware).

    Parameters
    ----------
    config :
        ``GNURadioSourceConfig`` describing the source's sample rate,
        center frequency, gain and any hardware device fields.
    """

    def __init__(self, config: GNURadioSourceConfig) -> None:
        self.config = config
        self._source_meta = SourceMetadata(
            source="stream",
            center_frequency_hz=self.config.center_frequency_hz,
            sample_rate=self.config.sample_rate,
            iq_format=self.config.iq_format,
            gain_db=self.config.gain_db,
            source_config=self.config.to_dict(),
        )
        self._chunk_meta = ChunkMetadata(
            sample_rate=self.config.sample_rate,
            center_frequency_hz=self.config.center_frequency_hz,
            gain_db=self.config.gain_db,
            timestamp=0.0,
            chunk_index=0,
            num_chunks=0,
            source_kind="synthetic",
            device_name=self.config.device_name,
            source_config=self.config.to_dict(),
        )

    # -- metadata ------------------------------------------------------

    def get_metadata(self) -> SourceMetadata:
        """The source metadata the rest of Spectra can reason about."""
        return self._source_meta

    def get_chunk_metadata(self, chunk_index: int = 0) -> ChunkMetadata:
        """Per-chunk provenance: sample rate, center freq, gain, timestamp."""
        return ChunkMetadata(
            sample_rate=self.config.sample_rate,
            center_frequency_hz=self.config.center_frequency_hz,
            gain_db=self.config.gain_db,
            timestamp=self._source_meta.timestamp or 0.0,
            chunk_index=chunk_index,
            num_chunks=0,
            source_kind="synthetic",
            device_name=self.config.device_name,
            source_config=self.config.to_dict(),
            file_name=None,
        )

    # -- GNU Radio-style block reads -----------------------------------

    def read_chunk(self, *, chunk_size: int, chunk_index: int) -> np.ndarray:
        """Read one GNU Radio-style buffer (greedy).

        GNU Radio nodes expose ``output_buffer``-style signatures in
        hardware backends.  The synthetic backend implements the same
        contract here.

        Returns a flat ``(chunk_size * 2,)`` float32 interleaved I/Q.
        """
        n = max(1, int(chunk_size))
        rng = np.random.default_rng(int(float(self.config.sample_rate) * 1e-3) ^ chunk_index)
        real = rng.standard_normal(n) * 0.05
        imag = rng.standard_normal(n) * 0.05
        return np.column_stack([real, imag]).astype(np.float32)

    def read_chunk_complex(self, *, chunk_size: int, chunk_index: int) -> np.ndarray:
        """Read one GNU Radio-style buffer of complex samples."""
        n = max(1, int(chunk_size))
        rng = np.random.default_rng(int(float(self.config.sample_rate) * 1e-3) ^ chunk_index)
        real = rng.standard_normal(n) * 0.05
        imag = rng.standard_normal(n) * 0.05
        return (real + 1j * imag).astype(np.complex128)

    # -- streaming -----------------------------------------------------

    def stream(self, acquisition: GNURadioAcquisitionConfig) -> GNURadioStreamer:
        """Return a streamer over this source, honouring the acquisition config."""
        return GNURadioStreamer(
            acquisition,
            chunk_factory=self._stream_factory,
        )

    def _stream_factory(  # noqa: D102
        self,
        streamer: GNURadioStreamer,
        start_chunk: int,
        max_chunks: int,
        _seed: int,
    ) -> Iterator[Chunk]:
        limit = max_chunks if max_chunks > 0 else start_chunk + 1
        for idx in range(start_chunk, limit):
            yield Chunk(
                samples=self.read_chunk_complex(
                    chunk_size=streamer.chunk_size, chunk_index=idx
                ),
                metadata=streamer.metadata,
            )


def make_synthetic_source(
    *,
    center_frequency_hz: float | None = None,
    sample_rate: float = 1_000_000.0,
    gain_db: float | None = None,
    device_name: str = "synthetic-bpsk",
    n_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, SourceMetadata]:
    """Build a short synthetic GNU Radio *source* for offline validation.

    Returns the full IQ array and its metadata.  The caller then wraps it
    with :class:`GNURadioSource`/``stream()`` to exercise the adapter.
    """
    rng = np.random.default_rng(seed)
    rate = float(sample_rate)
    t = np.arange(n_samples) / rate

    # Representative synthetic payload with controlled impairments (this
    # is the "generate IQ via GNU Radio flowgraph" step).
    signal = np.zeros(n_samples, dtype=np.complex128)
    # A 1 kHz pilot at the configured center frequency.
    if center_frequency_hz is not None:
        signal += 0.4 * np.exp(2j * np.pi * center_frequency_hz * t)
    # A short QPSK-ish burst mid-capture.
    burst = 0.6 * np.exp(1j * np.pi * rng.integers(0, 4, 100))
    start = 200
    signal[start : start + 100] = burst
    signal *= rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))

    meta = SourceMetadata(
        source="file",
        file_path=None,
        center_frequency_hz=center_frequency_hz,
        sample_rate=sample_rate,
        iq_format="complex",
        gain_db=gain_db,
        timestamp=0.0,
        source_config={
            "center_frequency_hz": center_frequency_hz,
            "sample_rate": sample_rate,
            "iq_format": "complex",
            "gain_db": gain_db,
            "device_name": device_name,
            "n_samples": n_samples,
            "seed": seed,
        },
    )
    return signal, meta
