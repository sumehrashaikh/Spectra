"""
Acquisition layer: load WAV and raw IQ captures into Signal objects.

Supported inputs
----------------
- WAV (PCM 8/16/32-bit and IEEE float32), mono (real, Hilbert-analytic)
  or stereo (interleaved IQ).
- Raw interleaved IQ files: complex64, complex128, int8, uint8, int16,
  int32, float32, float64, either endianness, I-first or Q-first order.
- Separate I and Q files.
- JSON metadata sidecars (``<name>.meta.json``) supplying sample rate,
  center frequency, dtype, endianness, and ordering for raw captures.
- Chunked streaming for files larger than memory.

All loaders return :class:`prototype.core.signal.Signal` objects with
provenance metadata. Loading never silently modifies sample data except
for documented, reported normalizations (integer scaling to [-1, 1)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..core.exceptions import LoaderError, SidecarError, UnsupportedFormatError
from ..core.signal import Signal

logger = logging.getLogger("spectra.io")

# -------------------------------------------------------------

_RAW_DTYPES: dict[str, np.dtype] = {
    "complex64": np.dtype(np.complex64),
    "complex128": np.dtype(np.complex128),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
}


@dataclass
class CaptureMetadata:
    """Acquisition metadata attached to every loaded Signal."""

    source_path: str | None = None
    file_format: str = "raw_iq"
    dtype: str = "complex64"
    endianness: str = "little"
    iq_order: str = "iq"
    center_frequency_hz: float | None = None
    gain_db: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "file_format": self.file_format,
            "dtype": self.dtype,
            "endianness": self.endianness,
            "iq_order": self.iq_order,
            "center_frequency_hz": self.center_frequency_hz,
            "gain_db": self.gain_db,
            "notes": self.notes,
            "extra": self.extra,
        }


def _resolve_dtype(
    dtype: str | np.dtype,
    endianness: str,
) -> np.dtype:
    """Build a byte-order-resolved dtype descriptor."""
    name = (
        dtype.name if isinstance(dtype, np.dtype) else str(dtype).strip().lower()
    )
    if name not in _RAW_DTYPES:
        raise UnsupportedFormatError(
            f"Unsupported raw dtype '{dtype}'. "
            f"Supported: {sorted(_RAW_DTYPES)}"
        )

    base = _RAW_DTYPES[name]
    if base.kind == "c":
        # Complex types: resolve byte order of the component floats.
        component = np.dtype(base.itemsize // 2 == 8 and np.float64 or np.float32)
        byte_order = "<" if endianness == "little" else ">"
        return np.dtype(f"{byte_order}c{base.itemsize}") if byte_order != "=" else base

    if endianness not in ("little", "big", "native"):
        raise UnsupportedFormatError(
            f"endianness must be 'little', 'big', or 'native', got '{endianness}'"
        )
    order = {"little": "<", "big": ">", "native": "="}[endianness]
    return base.newbyteorder(order) if order != "=" else base


def _scale_integer(samples: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Scale integer sample formats into the normalized [-1, 1) range."""
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        return samples.astype(np.complex128) / float(abs(info.min))
    if dtype.kind == "u":
        info = np.iinfo(dtype)
        midpoint = (float(info.max) + 1.0) / 2.0
        return (samples.astype(np.float64) - midpoint) / midpoint
    return samples


def _interleave_to_complex(
    flat: np.ndarray,
    dtype: np.dtype,
    iq_order: str,
) -> np.ndarray:
    """Convert a flat interleaved I/Q array into complex samples."""
    if flat.size % 2 != 0:
        raise LoaderError(
            "Raw IQ file does not contain an even number of scalar "
            "samples; it cannot be a complete IQ stream."
        )

    if iq_order == "iq":
        i_component = flat[0::2]
        q_component = flat[1::2]
    elif iq_order == "qi":
        i_component = flat[1::2]
        q_component = flat[0::2]
    else:
        raise UnsupportedFormatError(
            f"iq_order must be 'iq' or 'qi', got '{iq_order}'"
        )

    i_scaled = _scale_integer(i_component, dtype)
    q_scaled = _scale_integer(q_component, dtype)

    return i_scaled + 1j * q_scaled


def load_raw_iq(
    path: str | Path,
    sample_rate: float,
    dtype: str = "complex64",
    endianness: str = "little",
    iq_order: str = "iq",
    offset_bytes: int = 0,
    max_samples: int | None = None,
    center_frequency_hz: float | None = None,
    metadata: CaptureMetadata | None = None,
) -> Signal:
    """
    Load a raw interleaved IQ file.

    Parameters
    ----------
    path :
        File path.
    sample_rate :
        Sample rate in Hz (raw files carry none).
    dtype :
        Sample format; see module docstring for supported values.
        ``complex64``/``complex128`` files are read directly; all other
        formats are treated as interleaved scalar I/Q components.
    endianness :
        ``little``, ``big``, or ``native``.
    iq_order :
        ``iq`` (I first) or ``qi`` (Q first).
    offset_bytes :
        Skip this many leading bytes (headers).
    max_samples :
        Optionally truncate to this many complex samples.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise LoaderError(f"File not found: {file_path}")

    resolved = _resolve_dtype(dtype, endianness)

    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        raise LoaderError(f"Cannot stat file {file_path}: {exc}") from exc

    data_size = file_size - offset_bytes
    if data_size <= 0:
        raise LoaderError(
            f"File {file_path} has no data after a {offset_bytes}-byte offset."
        )

    item_bytes = max(resolved.itemsize, 1)

    if resolved.kind == "c":
        if data_size % resolved.itemsize != 0:
            raise LoaderError(
                f"File size ({data_size} bytes after offset) is not a "
                f"multiple of the {dtype} sample size "
                f"({resolved.itemsize} bytes). The file may be truncated "
                "or the dtype/offset is wrong."
            )
        count = data_size // resolved.itemsize
        if max_samples is not None:
            count = min(count, max_samples)
        try:
            samples = np.fromfile(
                file_path, dtype=resolved, count=count, offset=offset_bytes
            )
        except (OSError, ValueError) as exc:
            raise LoaderError(f"Could not read IQ file {file_path}: {exc}") from exc
        samples = samples.astype(np.complex128)
        used_dtype = str(np.dtype(dtype if not isinstance(dtype, np.dtype) else dtype))
    else:
        if data_size % (2 * item_bytes) != 0:
            raise LoaderError(
                f"File size ({data_size} bytes after offset) is not a "
                f"multiple of two {dtype} components. The file may be "
                "truncated or the dtype/offset is wrong."
            )
        count = data_size // (2 * item_bytes)
        if max_samples is not None:
            count = min(count, max_samples)
        try:
            flat = np.fromfile(
                file_path, dtype=resolved, count=2 * count, offset=offset_bytes
            )
        except (OSError, ValueError) as exc:
            raise LoaderError(f"Could not read IQ file {file_path}: {exc}") from exc
        samples = _interleave_to_complex(flat, resolved, iq_order)
        used_dtype = str(resolved)

    if samples.size == 0:
        raise LoaderError(f"IQ file {file_path} contains no samples.")

    if not np.all(np.isfinite(samples)):
        raise LoaderError(
            f"IQ file {file_path} contains NaN or infinite values; "
            "the capture or dtype interpretation is invalid."
        )

    capture = metadata or CaptureMetadata()
    capture.source_path = str(file_path)
    capture.file_format = "raw_iq"
    capture.dtype = used_dtype
    capture.endianness = endianness
    capture.iq_order = iq_order
    capture.center_frequency_hz = center_frequency_hz

    signal = Signal(
        samples=samples,
        sample_rate=float(sample_rate),
        metadata={"capture": capture.to_dict()},
    )
    logger.info(
        "Loaded %d complex samples (%s) from %s",
        signal.num_samples,
        used_dtype,
        file_path.name,
    )
    return signal


def load_iq_pair(
    i_path: str | Path,
    q_path: str | Path,
    sample_rate: float,
    dtype: str = "float32",
    endianness: str = "little",
    center_frequency_hz: float | None = None,
) -> Signal:
    """Load separate I and Q component files into one complex Signal."""
    i_file = Path(i_path)
    q_file = Path(q_path)

    for file_path in (i_file, q_file):
        if not file_path.is_file():
            raise LoaderError(f"File not found: {file_path}")

    if i_file.stat().st_size != q_file.stat().st_size:
        raise LoaderError(
            "I and Q files differ in size "
            f"({i_file.stat().st_size} vs {q_file.stat().st_size} bytes); "
            "they cannot form aligned IQ pairs."
        )

    resolved = _resolve_dtype(dtype, endianness)
    if resolved.kind == "c":
        raise UnsupportedFormatError(
            "Separate I/Q files require real component dtypes."
        )

    i_data = np.fromfile(i_file, dtype=resolved)
    q_data = np.fromfile(q_file, dtype=resolved)

    if i_data.size == 0:
        raise LoaderError("I/Q files contain no samples.")

    samples = _scale_integer(i_data, resolved) + 1j * _scale_integer(
        q_data, resolved
    )

    capture = CaptureMetadata(
        source_path=f"{i_file} + {q_file}",
        file_format="iq_pair",
        dtype=str(resolved),
        endianness=endianness,
        iq_order="separate",
        center_frequency_hz=center_frequency_hz,
    )
    return Signal(samples=samples, sample_rate=float(sample_rate),
                  metadata={"capture": capture.to_dict()})


def sidecar_path_for(path: str | Path) -> Path:
    """Return the conventional sidecar path ``<stem>.meta.json``."""
    file_path = Path(path)
    return file_path.with_name(f"{file_path.stem}.meta.json")


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """Load a JSON metadata sidecar, raising SidecarError when invalid."""
    sidecar = sidecar_path_for(path)
    if not sidecar.is_file():
        raise SidecarError(f"No sidecar metadata file at {sidecar}")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SidecarError(f"Invalid sidecar {sidecar}: {exc}") from exc
    if not isinstance(data, dict):
        raise SidecarError(f"Sidecar {sidecar} must contain a JSON object.")
    return data


def load_signal(
    path: str | Path,
    sample_rate: float | None = None,
    dtype: str | None = None,
    endianness: str | None = None,
    iq_order: str | None = None,
    center_frequency_hz: float | None = None,
    max_samples: int | None = None,
) -> Signal:
    """
    Load any supported capture, dispatching on file extension.

    Explicit keyword arguments override sidecar metadata. For raw IQ
    files a sample rate is required either explicitly or from the
    sidecar.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise LoaderError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    sidecar: dict[str, Any] = {}
    try:
        sidecar = load_sidecar(file_path)
    except SidecarError:
        pass

    def _get(key: str, override):
        if override is not None:
            return override
        return sidecar.get(key)

    resolved_rate = _get("sample_rate", sample_rate)
    resolved_dtype = _get("dtype", dtype) or "complex64"
    resolved_endian = _get("endianness", endianness) or "little"
    resolved_order = _get("iq_order", iq_order) or "iq"
    resolved_center = _get("center_frequency_hz", center_frequency_hz)

    if suffix == ".wav":
        return load_wav_signal(file_path, max_samples=max_samples)

    if resolved_rate is None:
        raise LoaderError(
            f"Raw IQ file {file_path.name} requires a sample rate: "
            "pass sample_rate= or provide a sidecar "
            f"({sidecar_path_for(file_path).name}) with 'sample_rate'."
        )

    return load_raw_iq(
        file_path,
        sample_rate=float(resolved_rate),
        dtype=str(resolved_dtype),
        endianness=str(resolved_endian),
        iq_order=str(resolved_order),
        center_frequency_hz=(
            float(resolved_center) if resolved_center is not None else None
        ),
        max_samples=max_samples,
    )


def load_wav_signal(path: str | Path, max_samples: int | None = None) -> Signal:
    """
    Load a WAV file (16/32-bit PCM or float32) into a Signal.

    Mono files are treated as real passband waveforms and converted to
    analytic (complex baseband-equivalent) signals via the Hilbert
    transform. Stereo files are interpreted as interleaved IQ.
    """
    try:
        from scipy.io import wavfile
        from scipy.signal import hilbert
    except ImportError as exc:  # pragma: no cover
        raise LoaderError("scipy is required for WAV loading.") from exc

    file_path = Path(path)
    try:
        rate, data = wavfile.read(file_path)
    except (OSError, ValueError) as exc:
        raise LoaderError(f"Invalid WAV file {file_path}: {exc}") from exc

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[0] == 0:
        raise LoaderError(f"WAV file {file_path} contains no samples.")

    kind = data.dtype.kind
    if kind == "i":
        info = np.iinfo(data.dtype)
        scaled = data.astype(np.float64) / float(abs(info.min))
    elif kind == "u":
        info = np.iinfo(data.dtype)
        midpoint = (float(info.max) + 1.0) / 2.0
        scaled = (data.astype(np.float64) - midpoint) / midpoint
    elif kind == "f":
        scaled = data.astype(np.float64)
        if not np.all(np.isfinite(scaled)):
            raise LoaderError(
                f"WAV file {file_path} contains NaN or infinite samples."
            )
    else:
        raise UnsupportedFormatError(
            f"Unsupported WAV sample format: {data.dtype}"
        )

    channels = scaled.shape[1]
    if channels == 1:
        real = scaled[:, 0]
        if max_samples is not None:
            real = real[:max_samples]
        samples = hilbert(real).astype(np.complex128)
        mode = "mono_analytic"
    elif channels == 2:
        samples = scaled[:, 0] + 1j * scaled[:, 1]
        if max_samples is not None:
            samples = samples[:max_samples]
        mode = "stereo_iq"
    else:
        raise UnsupportedFormatError(
            f"WAV file has {channels} channels; only mono and stereo are supported."
        )

    capture = CaptureMetadata(
        source_path=str(file_path),
        file_format="wav",
        dtype=str(data.dtype),
        notes=f"wav_mode={mode}",
    )
    signal = Signal(
        samples=samples,
        sample_rate=float(rate),
        metadata={"capture": capture.to_dict()},
    )
    logger.info(
        "Loaded WAV %s: %d samples @ %d Hz (%s)",
        file_path.name,
        signal.num_samples,
        rate,
        mode,
    )
    return signal


def stream_raw_iq(
    path: str | Path,
    chunk_samples: int = 1 << 20,
    dtype: str = "complex64",
    endianness: str = "little",
    iq_order: str = "iq",
    offset_bytes: int = 0,
) -> Iterator[np.ndarray]:
    """
    Stream a raw IQ file in fixed-size complex chunks.

    Enables chunked processing of captures larger than memory. The
    final chunk may be shorter. Raises the same validation errors as
    :func:`load_raw_iq`.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise LoaderError(f"File not found: {file_path}")

    resolved = _resolve_dtype(dtype, endianness)
    item_bytes = max(resolved.itemsize, 1)

    if resolved.kind == "c":
        item_bytes = resolved.itemsize
        if chunk_samples % 1:
            raise LoaderError("chunk_samples must be an integer.")
        bytes_per_complex = item_bytes
    else:
        bytes_per_complex = 2 * item_bytes

    if chunk_samples <= 0:
        raise LoaderError("chunk_samples must be positive.")

    with open(file_path, "rb") as handle:
        if offset_bytes:
            handle.seek(offset_bytes)
        while True:
            raw = handle.read(chunk_samples * bytes_per_complex)
            if not raw:
                break
            if resolved.kind == "c":
                chunk = np.frombuffer(raw, dtype=resolved).astype(np.complex128)
            else:
                if len(raw) % (2 * item_bytes) != 0:
                    # Truncated tail: drop the incomplete final pair.
                    usable = (len(raw) // (2 * item_bytes)) * 2 * item_bytes
                    raw = raw[:usable]
                    if not raw:
                        break
                flat = np.frombuffer(raw, dtype=resolved)
                chunk = _interleave_to_complex(flat, resolved, iq_order)
            yield chunk


def save_iq(
    path: str | Path,
    samples: np.ndarray,
    dtype: str = "complex64",
    endianness: str = "little",
) -> Path:
    """
    Save complex samples as a raw interleaved IQ file (round-trip support).
    """
    file_path = Path(path)
    resolved = _resolve_dtype(dtype, endianness)

    data = np.asarray(samples, dtype=np.complex128)
    if not np.all(np.isfinite(data)):
        raise LoaderError("Refusing to save samples containing NaN or Inf.")

    if resolved.kind == "c":
        out = data.astype(resolved)
        out.tofile(file_path)
    else:
        if resolved.kind == "i":
            info = np.iinfo(resolved)
            i_part = np.clip(np.real(data) * abs(info.min), info.min, info.max)
            q_part = np.clip(np.imag(data) * abs(info.min), info.min, info.max)
            interleaved = np.empty(2 * data.size, dtype=resolved)
            interleaved[0::2] = i_part.astype(resolved)
            interleaved[1::2] = q_part.astype(resolved)
        elif resolved.kind == "u":
            info = np.iinfo(resolved)
            midpoint = (float(info.max) + 1.0) / 2.0
            i_part = np.clip(np.real(data) * midpoint + midpoint, 0, info.max)
            q_part = np.clip(np.imag(data) * midpoint + midpoint, 0, info.max)
            interleaved = np.empty(2 * data.size, dtype=resolved)
            interleaved[0::2] = i_part.astype(resolved)
            interleaved[1::2] = q_part.astype(resolved)
        else:
            interleaved = np.empty(2 * data.size, dtype=resolved)
            interleaved[0::2] = np.real(data).astype(resolved)
            interleaved[1::2] = np.imag(data).astype(resolved)
        if endianness in ("little", "big"):
            interleaved = interleaved.astype(
                interleaved.dtype.newbyteorder(
                    "<" if endianness == "little" else ">"
                )
            )
        interleaved.tofile(file_path)

    logger.info("Saved %d complex samples to %s", data.size, file_path.name)
    return file_path


def write_sidecar(path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write a JSON metadata sidecar next to a capture file."""
    file_path = Path(path)
    sidecar = sidecar_path_for(file_path)
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sidecar
