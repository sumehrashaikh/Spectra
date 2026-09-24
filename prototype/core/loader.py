from pathlib import Path
import struct
import wave

import numpy as np

from prototype.core.logging_config import logger

try:
    from scipy.signal import hilbert
except ImportError:
    hilbert = None


def _load_wav_float_via_scipy(file_path: Path):
    """
    Read an IEEE-float WAV via scipy and apply the same channel
    convention as the PCM path (stereo → IQ, mono → Hilbert).
    """

    try:
        from scipy.io import wavfile as _wavfile
    except ImportError as exc:  # pragma: no cover
        raise ValueError(
            "scipy is required to read IEEE-float WAV files."
        ) from exc

    try:
        sample_rate, data = _wavfile.read(str(file_path))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Invalid WAV file: {file_path}"
        ) from exc

    data = np.asarray(data)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    if data.shape[0] == 0:
        raise ValueError("WAV file contains no samples.")

    if data.dtype.kind == "f":
        scaled = data.astype(np.float64)

        if not np.all(np.isfinite(scaled)):
            scaled = np.nan_to_num(
            scaled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            )
    else:
        # scipy decoded an integer PCM format; rescale like the PCM path
        if data.dtype.kind == "i":
            info = np.iinfo(data.dtype)
            scaled = data.astype(np.float64) / float(abs(info.min))
        elif data.dtype.kind == "u":
            info = np.iinfo(data.dtype)
            midpoint = (float(info.max) + 1.0) / 2.0
            scaled = (data.astype(np.float64) - midpoint) / midpoint
        else:
            raise ValueError(
                f"Unsupported WAV sample format: {data.dtype}"
            )

    channels = scaled.shape[1]

    if channels == 1:

        logger.info("WAV mode: MONO real signal (float)")

        if hilbert is None:
            raise ImportError(
                "scipy is required for mono RF/WAV "
                "analytic-signal conversion."
            )

        samples = hilbert(
            scaled[:, 0]
        ).astype(np.complex128)

    elif channels == 2:

        logger.info("WAV mode: STEREO IQ (float)")

        samples = (
            scaled[:, 0]
            + 1j * scaled[:, 1]
        )

    else:
        raise ValueError(
            "Unsupported WAV channel count. "
            "Only mono and stereo WAV files "
            "are currently supported."
        )

    if samples.size == 0:
        raise ValueError("WAV file contains no samples.")

    samples = samples - np.mean(samples)

    return samples, float(sample_rate)


def load_wav(path: str):
    """
    Load a WAV file and return complex-valued samples.

    Stereo WAV:
        Channel 0 -> I
        Channel 1 -> Q

    Mono WAV:
        Real waveform -> analytic complex signal
        using the Hilbert transform.

    Returns
    -------
    samples : np.ndarray
        Complex-valued signal.

    sample_rate : int
        Sampling frequency in Hz.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    # --------------------------------------------------------
    # Open WAV
    # --------------------------------------------------------

    try:
        wav_file = wave.open(
            str(file_path),
            "rb"
        )
    except wave.Error as exc:
        # Python's wave module rejects IEEE-float WAVs (format tag 3).
        # Those are common for IQ captures, so fall back to scipy's
        # reader which supports float32/float64 WAV natively.
        if "unknown format: 3" in str(exc):
            return _load_wav_float_via_scipy(file_path)
        raise ValueError(
            f"Invalid WAV file: {file_path}"
        ) from exc

    # --------------------------------------------------------
    # Read WAV information
    # --------------------------------------------------------

    with wav_file as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()

        raw = wav.readframes(
            frame_count
        )

    # --------------------------------------------------------
    # IEEE-float WAV (format tag 3) is not handled by the PCM
    # branches below; detect it by re-reading the fmt chunk.
    # --------------------------------------------------------

    is_float_format = False

    try:
        with open(file_path, "rb") as fh:
            header = fh.read(4096)

        fmt_index = header.find(b"fmt ")

        if fmt_index >= 0 and fmt_index + 16 <= len(header):
            (
            audio_format,
            _nchannels,
            _rate,
            _byterate,
            _align,
            bits,
            ) = struct.unpack_from(
            "<HHIIHH",
            header,
            fmt_index + 8,
            )

            is_float_format = (
            audio_format == 3 and bits == 64
            ) or (
            audio_format == 3 and bits == 32
            )

    except OSError:
        # header unreadable: assume PCM; the main read already succeeded
        is_float_format = False

    # --------------------------------------------------------
    # Convert PCM → floating point
    # --------------------------------------------------------

    if is_float_format:

        dtype = np.float32 if sample_width == 4 else np.float64

        data = np.frombuffer(
        raw,
        dtype=dtype,
        ).astype(np.float64)

        if not np.all(np.isfinite(data)):
            # sanitize so the complex math stays finite; the NaN check
            # below still guards the final samples
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    elif sample_width == 1:

        data = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float64)

        data = (
            data - 128.0
        ) / 128.0

    elif sample_width == 2:

        data = np.frombuffer(
            raw,
            dtype=np.int16
        ).astype(np.float64)

        data /= 32768.0

    elif sample_width == 4:

        data = np.frombuffer(
            raw,
            dtype=np.int32
        ).astype(np.float64)

        data /= 2147483648.0

    else:

        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width} bytes"
        )

    # --------------------------------------------------------
    # Reshape channels
    # --------------------------------------------------------

    data = data.reshape(
        -1,
        channels
    )

    # --------------------------------------------------------
    # Convert to complex representation
    # --------------------------------------------------------

    if channels == 1:

        logger.info("WAV mode: MONO real signal")

        real_signal = data[:, 0]

        # ----------------------------------------------------
        # Convert real RF waveform into analytic signal
        # ----------------------------------------------------

        if hilbert is None:

            raise ImportError(
                "scipy is required for mono RF/WAV "
                "analytic-signal conversion.\n"
                "Install it with:\n"
                "pip install scipy"
            )

        samples = hilbert(
            real_signal
        ).astype(
            np.complex128
        )

    elif channels == 2:

        logger.info("WAV mode: STEREO IQ (channel 0 = I, channel 1 = Q)")

        samples = (
            data[:, 0]
            + 1j * data[:, 1]
        )

    else:

        raise ValueError(
            "Unsupported WAV channel count. "
            "Only mono and stereo WAV files "
            "are currently supported."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if samples.size == 0:

        raise ValueError(
            "WAV file contains no samples."
        )

    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    samples = (
        samples
        - np.mean(samples)
    )

    return samples, sample_rate


def load_iq(
    path: str,
    dtype=np.complex64
):
    """
    Load a raw complex-IQ file.

    Assumes the file already contains
    complex-valued samples.
    """

    file_path = Path(path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    try:

        samples = np.fromfile(
            file_path,
            dtype=dtype
        )

    except (OSError, ValueError) as exc:

        raise ValueError(
            f"Could not read IQ file: {file_path}"
        ) from exc

    if samples.size == 0:

        raise ValueError(
            "IQ file contains no samples."
        )

    return samples