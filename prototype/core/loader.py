from pathlib import Path
import wave

import numpy as np

try:
    from scipy.signal import hilbert
except ImportError:
    hilbert = None


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
    # Convert PCM → floating point
    # --------------------------------------------------------

    if sample_width == 1:

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

        print(
            "WAV mode: MONO real signal"
        )

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

        print(
            "WAV mode: STEREO IQ"
        )

        print(
            "Channel 0 = I"
        )

        print(
            "Channel 1 = Q"
        )

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