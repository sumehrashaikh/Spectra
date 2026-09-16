import numpy as np
from scipy import signal

from matplotlib.figure import Figure


def compute_spectrum(
    samples: np.ndarray,
    sample_rate: float
):
    """Compute a centered FFT magnitude spectrum."""

    if samples.size == 0:
        raise ValueError("Signal is empty.")

    n = len(samples)

    window = np.hanning(n)

    spectrum = np.fft.fftshift(
        np.fft.fft(samples * window)
    )

    frequency = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1 / sample_rate
        )
    )

    magnitude = np.abs(spectrum)

    return frequency, magnitude


def create_time_figure(
    samples,
    sample_rate,
    max_points=5000
):
    """Create time-domain Matplotlib figure."""

    count = min(
        len(samples),
        max_points
    )

    t = np.arange(count) / sample_rate

    figure = Figure(
        figsize=(8, 3),
        tight_layout=True
    )

    ax = figure.add_subplot(111)

    ax.plot(
        t,
        samples[:count].real
    )

    ax.set_title("Time-Domain Signal")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    return figure


def create_spectrum_figure(
    samples,
    sample_rate
):
    """Create FFT spectrum figure."""

    frequency, magnitude = compute_spectrum(
        samples,
        sample_rate
    )

    figure = Figure(
        figsize=(8, 3),
        tight_layout=True
    )

    ax = figure.add_subplot(111)

    ax.plot(
        frequency,
        magnitude
    )

    ax.set_title("Signal Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)

    return figure


def create_waterfall_figure(
    samples,
    sample_rate,
    nperseg=1024,
    noverlap=768
):
    """Create a time-frequency waterfall/spectrogram."""

    if len(samples) < 16:
        raise ValueError(
            "Signal is too short for a waterfall."
        )

    # Prevent an unnecessarily large FFT
    nperseg = min(
        nperseg,
        len(samples)
    )

    noverlap = min(
        noverlap,
        nperseg - 1
    )

    frequencies, times, Zxx = signal.stft(
        samples,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )

    frequencies = np.fft.fftshift(
        frequencies
    )

    Zxx = np.fft.fftshift(
        Zxx,
        axes=0
    )

    magnitude_db = 20 * np.log10(
        np.abs(Zxx) + 1e-12
    )

    figure = Figure(
        figsize=(8, 3),
        tight_layout=True
    )

    ax = figure.add_subplot(111)

    mesh = ax.pcolormesh(
        times,
        frequencies,
        magnitude_db,
        shading="auto"
    )

    ax.set_title("Waterfall / Spectrogram")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    figure.colorbar(
        mesh,
        ax=ax,
        label="Magnitude (dB)"
    )

    return figure


def create_constellation_figure(
    samples,
    max_points=5000,
    title="IQ Constellation"
):
    """Create IQ constellation figure."""

    if len(samples) == 0:
        raise ValueError(
            "Signal is empty."
        )

    # Mono real signal has no Q component
    if np.allclose(
        samples.imag,
        0,
        atol=1e-12
    ):
        raise ValueError(
            "Constellation unavailable: "
            "this signal has no Q component."
        )

    count = min(
        len(samples),
        max_points
    )

    iq = samples[:count]

    figure = Figure(
        figsize=(5, 4),
        tight_layout=True
    )

    ax = figure.add_subplot(111)

    ax.scatter(
        iq.real,
        iq.imag,
        s=5
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.axvline(
        0,
        linestyle="--"
    )

    ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)
    ax.axis("equal")

    return figure
