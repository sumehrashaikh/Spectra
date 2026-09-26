"""GNU Radio v1 adapter: bridge from GNU Radio chunks to the SPECTRA Signal model.

All GNU Radio input is normalized into the same :class:`prototype.core.signal.Signal`
object used by the WAV / raw-IQ loaders, so the analysis stages in
``pipeline.py``, ``classification/classifier.py`` and the reporters
never need to know the samples came from GNU Radio.

The GNU Radio backend is OPTIONAL.  If the ``gnuradio`` python package is
not installed, :func:`gnuradio_available` is False and the adapter falls
back to the ordinary loaders (or returns ``None`` to the caller, which is
expected to choose the WAV/IQ path).

Design
------
- ``signal_from_gnuradio_chunks`` consumes an iterable of chunks and builds
  one contiguous ``Signal``.
- ``signal_from_gnuradio_source`` wraps a :class:`GNURadioSource`
  (``.stream()``) into that call.
- Both keep the provenance metadata (``sample_rate``, ``center_frequency``,
  ``gain``, ``timestamp``, ``source``, ``chunk`` info) in ``Signal.metadata``
  under the ``capture.gnuradio`` key, mirroring how ``io/loaders.py``
  stores ``capture``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from prototype.core.signal import Signal

from .config import GNURadioAcquisitionConfig, SourceMetadata
from .streaming import Chunk, ChunkMetadata, GNURadioStreamer

logger = logging.getLogger("spectra.io.gnuradio.adapter")


def _normalize_iq(samples: np.ndarray) -> np.ndarray:
    """Canonicalise a GNU Radio chunk to ``(n,) complex128``.

    Accepts ``(n,) complex`` or ``(n,2)`` float32/64 interleaved I/Q.
    Nothing is copied more than once: real/complex dtypes use a view when
    the conversion is lossless.
    """
    arr = np.asarray(samples)
    if arr.dtype.kind == "c":
        if arr.ndim == 1:
            return arr.astype(np.complex128, copy=False)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return (arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64))
        raise ValueError(
            f"GNU Radio complex chunk has unexpected shape {arr.shape}; expected (n,) or (n,2)"
        )
    if arr.ndim == 2 and arr.shape[1] == 2:
        return (arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64))
    if arr.ndim == 1:
        return arr.astype(np.complex128, copy=False)
    raise ValueError(f"Unexpected GNU Radio chunk shape {arr.shape}")


def signal_from_gnuradio_chunks(
    chunks: Iterable[Chunk],
    *,
    source_metadata: SourceMetadata,
    provenance_override: dict | None = None,
) -> Signal:
    """Build one contiguous ``Signal`` from a GNU Radio-style chunk iterable.

    Chunks are concatenated in order.  Metadata from every chunk is merged;
    the first non-``None`` value wins for each metadata key, so a
    per-chunk timestamp is the earliest reported acquisition time.

    Parameters
    ----------
    chunks :
        Iterable of :class:`Chunk` (GNU Radio buffers).
    source_metadata :
        Top-level source metadata (sample rate, center frequency,
        gain, timestamp, source kind, source config).
    provenance_override :
        Optional dict merged into the Signal metadata under
        ``capture.gnuradio``.
    """
    collected: list[np.ndarray] = []
    meta = source_metadata.to_dict()

    for chunk in chunks:
        arr = chunk.as_complex()
        if chunk.metadata.center_frequency_hz is not None:
            meta.setdefault("center_frequency_hz", chunk.metadata.center_frequency_hz)
        if chunk.metadata.sample_rate:
            meta.setdefault("sample_rate", chunk.metadata.sample_rate)
        if chunk.metadata.gain_db is not None:
            meta.setdefault("gain_db", chunk.metadata.gain_db)
        if chunk.metadata.timestamp is not None:
            meta.setdefault("timestamp", chunk.metadata.timestamp)
        meta.setdefault("iq_format", chunk.metadata.source_kind)
        if chunk.metadata.device_name:
            meta.setdefault("device_name", chunk.metadata.device_name)
        if chunk.metadata.device_name:
            meta.setdefault("device_name", chunk.metadata.device_name)
        collected.append(arr)

    if not collected:
        raise ValueError("No GNU Radio chunks supplied; cannot build a Signal.")

    samples = np.concatenate(collected, axis=0)

    # GNU Radio-specific channel metadata.  Keep `source_kind` at the
    # high-level category (gnuradio), and push the specific source/device
    # identity into the nested source metadata (`source_config.device_name`),
    # consistent with the CLI/GUI convention.
    # GNU Radio-specific channel metadata.  Keep `source_kind` at the
    # high-level category (gnuradio); preserve the specific GNU Radio
    # source/device identity so the CLI/GUI convention below holds:
    #   capture.source_kind = "gnuradio"
    #   input_info.source     = device_name (synthetic-bpsk, RTL-SDR, ...)
    # GNU Radio-specific channel metadata.  Keep `source_kind` at the
    # high-level category (gnuradio); the specific GNU Radio source/device
    # identity is exposed under `source_config.device_name` in the same
    # capture dict, aligned with the CLI/GUI convention:
    #   capture.source_kind = "gnuradio"
    #   input_info.source     = device_name
    # so that metadata stays extensible for future hardware such as
    # RTL-SDR, HackRF, USRP, etc.
    # GNU Radio-specific channel metadata.  Keep `source_kind` at the
    # high-level category (gnuradio); preserve the specific GNU Radio
    # source/device identity in the more detailed `source_config`
    # metadata so the CLI/GUI convention below holds:
    #   capture.source_kind = "gnuradio"
    #   input_info.source     = device_name
    # This keeps the metadata extensible for future hardware such as
    # RTL-SDR, HackRF, USRP, etc., without overloading source_kind.
    gnuradio_meta = {
        "file_format": "gnuradio",
        "iq_format": meta.get("iq_format", "complex"),
        "sample_rate": float(meta.get("sample_rate", 0.0)),
        "center_frequency_hz": meta.get("center_frequency_hz"),
        "gain_db": meta.get("gain_db"),
        "timestamp": meta.get("timestamp"),
        "source_kind": "gnuradio",
        "num_chunks": 0,
        "chunk_index": 0,
        "source_config": meta.get("source_config", {}),
        "notes": "GNU Radio v1 adapter; optional acquisition layer",
    }
    if provenance_override:
        gnuradio_meta.update(provenance_override)

    capture_meta = gnuradio_meta

    return Signal(
        samples=samples,
        sample_rate=float(meta.get("sample_rate", 1_000_000.0)),
        metadata={"capture": capture_meta},
    )


def signal_from_gnuradio_source(
    source: "GNURadioSource",
    acquisition: GNURadioAcquisitionConfig,
    *,
    provenance_override: dict | None = None,
) -> Signal:
    """Convenience: a :class:`GNURadioSource` wrapped via its `stream()`."""
    stream = source.stream(acquisition)
    return signal_from_gnuradio_chunks(
        stream._chunks(),
        source_metadata=source.get_metadata(),
        provenance_override=provenance_override,
    )


def signal_from_gnuradio_file(
    filepath: str | Path,
    *,
    acquisition: GNURadioAcquisitionConfig,
) -> Signal:
    """Load a GNU Radio *replay* file (saved interleaved IQ) into a Signal.

    GNU Radio does not define a single file format, so this loader covers
    the common raw interleaved-IQ conventions (``complex64``, ``complex128``
    and scalar I/Q pairs).  It reuses the ordinary raw-IQ loader so the
    metadata and dtype/encoding rules stay identical.
    """
    from prototype.io.loaders import load_raw_iq  # local import avoids cycle

    signal = load_raw_iq(
        Path(filepath),
        sample_rate=acquisition.source.sample_rate,
        dtype="complex128",
        endianness="little",
        iq_order="iq",
        max_samples=None,
    )
    signal.metadata = {"capture": {
        "file_format": "gnuradio_replay",
        "iq_format": "complex",
        "sample_rate": float(signal.sample_rate),
        "center_frequency_hz": acquisition.source.center_frequency_hz,
        "gain_db": acquisition.source.gain_db,
        "timestamp": acquisition.source.timestamp,
        "source_kind": "file",
        "notes": str(filepath),
    }}
    return signal
