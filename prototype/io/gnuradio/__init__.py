"""GNU Radio integration for Spectra v1.

An OPTIONAL acquisition/streaming/SDR layer. GNU Radio is NOT a
dependency of the core WAV/IQ analysis path; it is installed and used
only when ``pip install spectra[gnuradio]`` is chosen.

The following invariants hold:

- Spectra works perfectly well with WAV / raw IQ and needs no GNU Radio.
- GNU Radio is an INPUT / STREAMING / SDR acquisition layer, not a
  replacement for the SPECTRA DSP engine.
- The GNU Radio backend is import-guarded. If GNU Radio is not installed
  the adapter returns ``None`` rather than raising ``ImportError``.
- The same ``Signal`` produced by a GNU Radio source is fed into the
  existing ``analyze_samples`` / ``analyze_capture`` pipeline, so the
  analysis code never knows where the samples came from.
"""

from __future__ import annotations

import logging

from .config import GNURadioAcquisitionConfig, GNURadioSourceConfig
from .streaming import GNURadioStreamer
from .source import GNURadioSource
from .adapter import signal_from_gnuradio_source, signal_from_gnuradio_chunks


__all__ = [
    "GNURadioAcquisitionConfig",
    "GNURadioSourceConfig",
    "GNURadioSource",
    "GNURadioStreamer",
    "make_gnuradio_source",
    "gnuradio_available",
    "signal_from_gnuradio_source",
    "signal_from_gnuradio_chunks",
]
from .source import GNURadioSource
from .adapter import signal_from_gnuradio_source, signal_from_gnuradio_chunks

__all__ = [
    "GNURadioAcquisitionConfig",
    "GNURadioSourceConfig",
    "GNURadioStreamer",
    "make_gnuradio_source",
    "gnuradio_available",
]

logger = logging.getLogger("spectra.io.gnuradio")


def gnuradio_available() -> bool:
    """True when the GNU Radio package or subprocess runtime is available."""
    if _import_gnuradio():
        return True
    try:
        from dsp_gnuradio.bridge import check_gnuradio_available
        avail, _ = check_gnuradio_available()
        return bool(avail)
    except Exception:
        try:
            from gnuradio_integration.dsp_gnuradio.bridge import check_gnuradio_available
            avail, _ = check_gnuradio_available()
            return bool(avail)
        except Exception:
            return False


def make_gnuradio_source(
    config: GNURadioAcquisitionConfig | GNURadioSourceConfig,
) -> "GNURadioSource | None":
    """Build a GNU Radio source if the runtime supports it, else None."""
    if not gnuradio_available():
        return None
    try:
        from .source import GNURadioSource
    except Exception:  # noqa: BLE001
        logger.warning("GNU Radio source backend unavailable", exc_info=True)
        return None
    source_cfg = config.source if hasattr(config, "source") else config
    return GNURadioSource(source_cfg)


def _import_gnuradio():
    """Dynamically import GNU Radio, returning True on success."""
    try:
        import gnuradio  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_gnuradio_importable():
    """Raise a helpful error if GNU Radio is missing but required."""
    if not gnuradio_available():
        raise ImportError(
            "GNU Radio is not installed. Install it with "
            "`pip install spectra[gnuradio]` or `pip install gnuradio` "
            "and re-run."
        )


# Expose the GNU Radio package version when available (informational
# only; never a hard dependency).
try:
    import gnuradio

    __version__ = getattr(gnuradio, "__version__", "unknown")
except Exception:  # noqa: BLE001
    __version__ = "unavailable"
