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


def create_spectrum_figure_gnuradio(fft_dict):
    """Create FFT spectrum figure from GNU Radio output."""
    freqs = np.asarray(fft_dict.get("freqs", []))
    mag = np.asarray(fft_dict.get("magnitude", []))
    source = fft_dict.get("source", "GNU Radio")

    figure = Figure(
        figsize=(8, 3),
        tight_layout=True
    )
    ax = figure.add_subplot(111)
    ax.plot(freqs, mag, color="#1f77b4")
    ax.set_title(f"Signal Spectrum [Source: {source}]")
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


def create_waterfall_figure_gnuradio(waterfall_dict):
    """Create waterfall figure from GNU Radio output."""
    freqs = np.asarray(waterfall_dict.get("freqs", []))
    times = np.asarray(waterfall_dict.get("times", []))
    power_db = np.asarray(waterfall_dict.get("power_db", []))
    source = waterfall_dict.get("source", "GNU Radio")

    figure = Figure(
        figsize=(8, 3),
        tight_layout=True
    )
    ax = figure.add_subplot(111)

    if power_db.size > 0 and len(times) > 0 and len(freqs) > 0:
        # Mesh expects X: times, Y: freqs, C: power_db shape (len(freqs), len(times))
        if power_db.shape == (len(freqs), len(times)):
            mesh = ax.pcolormesh(times, freqs, power_db, shading="auto", cmap="viridis")
        elif power_db.shape == (len(times), len(freqs)):
            mesh = ax.pcolormesh(times, freqs, power_db.T, shading="auto", cmap="viridis")
        else:
            mesh = ax.imshow(
                power_db,
                aspect="auto",
                origin="lower",
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                cmap="viridis"
            )
        figure.colorbar(mesh, ax=ax, label="Power (dB)")

    ax.set_title(f"Waterfall / Spectrogram [Source: {source}]")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    return figure


def create_error_figure(title: str, message: str) -> Figure:
    """Create a figure displaying an error or unavailable status message."""
    figure = Figure(figsize=(8, 3), tight_layout=True)
    ax = figure.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        f"{title}\n\n{message}",
        ha="center",
        va="center",
        wrap=True,
        fontsize=11,
        color="#c0392b",
        fontweight="bold"
    )
    ax.set_axis_off()
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


def create_spectrum_figure_from_psd(
    frequencies,
    magnitude,
    title="Signal Spectrum (GNU Radio)",
):
    """Spectrum figure from precomputed (frequencies, magnitude) arrays.

    Used when the FFT was produced by an external backend (the GNU Radio
    headless flowgraph) instead of :func:`compute_spectrum`.
    """

    frequencies = np.asarray(frequencies, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)

    if frequencies.size == 0 or frequencies.size != magnitude.size:
        raise ValueError("Spectrum arrays are empty or mismatched.")

    figure = Figure(figsize=(8, 3), tight_layout=True)

    ax = figure.add_subplot(111)

    ax.plot(frequencies, magnitude)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)

    return figure


def create_waterfall_figure_from_matrix(
    frequencies,
    times,
    power_db,
    title="Waterfall / Spectrogram (GNU Radio)",
):
    """Waterfall figure from a precomputed ``(bins, slices)`` power matrix.

    ``power_db`` is laid out as rows = frequency bins, columns = time
    slices (the same orientation the GNU Radio flowgraph emits).
    """

    frequencies = np.asarray(frequencies, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    power_db = np.asarray(power_db, dtype=np.float64)

    if power_db.ndim != 2 or power_db.size == 0:
        raise ValueError("Waterfall matrix is empty.")
    if power_db.shape[0] != frequencies.size:
        raise ValueError("Waterfall matrix rows do not match the frequency axis.")
    if power_db.shape[1] != times.size:
        raise ValueError("Waterfall matrix columns do not match the time axis.")

    figure = Figure(figsize=(8, 3), tight_layout=True)

    ax = figure.add_subplot(111)

    mesh = ax.pcolormesh(times, frequencies, power_db, shading="auto")

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    figure.colorbar(mesh, ax=ax, label="Magnitude (dB)")

    return figure


def create_symbol_constellation_figure(
    symbols,
    max_points=4096,
    title="Recovered Constellation",
):
    """Constellation of synchronized hard-decision symbols.

    Unlike the raw-IQ scatter, this shows the recovered symbol lattice
    after synchronization: for a correctly demodulated signal the
    points cluster on the ideal grid, which is the visual quality
    signal for the whole receive chain.
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.size == 0:
        raise ValueError("No symbols to plot.")

    count = min(symbols.size, max_points)

    iq = symbols[:count]

    figure = Figure(
        figsize=(5, 4),
        tight_layout=True,
    )

    ax = figure.add_subplot(111)

    ax.scatter(
        iq.real,
        iq.imag,
        s=8,
        alpha=0.6,
    )

    # Equal aspect keeps the lattice geometry readable regardless of
    # modulation order; no ideal-grid overlay is drawn here because
    # the point set depends on the classified constellation, not on a
    # fixed assumption.
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))

    if rms > 1e-12:

        limit = 1.6 * rms

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

    ax.axhline(0, linestyle="--", color="0.6", linewidth=0.7)
    ax.axvline(0, linestyle="--", color="0.6", linewidth=0.7)

    ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    return figure
