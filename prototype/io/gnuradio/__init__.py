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
from .viz import (
    compute_spectrum_waterfall,
    gnuradio_python,
    viz_script_path,
)


__all__ = [
    "GNURadioAcquisitionConfig",
    "GNURadioSourceConfig",
    "GNURadioSource",
    "GNURadioStreamer",
    "make_gnuradio_source",
    "gnuradio_available",
    "gnuradio_runtime_status",
    "signal_from_gnuradio_source",
    "signal_from_gnuradio_chunks",
    "compute_spectrum_waterfall",
    "gnuradio_python",
    "viz_script_path",
]

logger = logging.getLogger("spectra.io.gnuradio")


def gnuradio_available() -> bool:
    """True when the GNU Radio Python package is importable."""
    return _import_gnuradio()


def make_gnuradio_source(
    config: GNURadioAcquisitionConfig,
) -> "GNURadioSource | None":
    """Build a GNU Radio source if the runtime supports it, else None."""
    if not gnuradio_available():
        return None
    try:
        from .source import GNURadioSource
    except Exception:  # noqa: BLE001
        logger.warning("GNU Radio source backend unavailable", exc_info=True)
        return None
    return GNURadioSource(config)


def _import_gnuradio():
    """Dynamically import GNU Radio, returning True on success."""
    try:
        import gnuradio  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def gnuradio_runtime_status() -> tuple[bool, str]:
    """Whether a *usable* GNU Radio runtime exists: ``(available, detail)``.

    ``gnuradio_available()`` answers a narrower question -- is ``gnuradio``
    importable into this very Python process?  GNU Radio normally lives in
    its own environment (radioconda), so that check is False while a
    perfectly good GNU Radio is installed and runnable, which made the GUI
    claim "GNU Radio is not installed" permanently.

    This function answers the question the acquisition and visualization
    paths actually care about: can a GNU Radio runtime be executed?  It
    uses the headless flowgraph probe (``viz.gnuradio_available``, a
    subprocess in the GNU Radio interpreter) and only falls back to the
    in-process import check.

    ``detail`` is ``"GNU Radio <version>"`` on success, otherwise a
    human-readable reason for the failure.
    """
    reason = "GNU Radio runtime not detected"
    try:
        from .viz import gnuradio_available as _runtime_check

        available, detail = _runtime_check()
        if available:
            return True, str(detail)
        reason = str(detail)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        reason = f"GNU Radio runtime check failed: {exc}"

    # Fallback: this interpreter itself can import GNU Radio.
    try:
        import gnuradio

        return True, f"GNU Radio {getattr(gnuradio, '__version__', 'unknown')}"
    except Exception:  # noqa: BLE001
        return False, reason


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
