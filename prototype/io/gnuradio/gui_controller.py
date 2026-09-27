"""
GUI acquisition controller for GNU Radio v1.

This module isolates the GNU Radio GUI acquisition flow from the Qt main
window.  It is the only place that knows how to start/stop a GNU Radio
acquisition and how to hand the resulting Signal across to the existing
SPECTRA analysis pipeline.

Flow:
    GUI
    -> gui_controller.GNURadioAcquisitionWorker
    -> GNURadioSource
    -> existing Chunk -> Signal adapter (prototype/io/gnuradio/adapter)
    -> existing SPECTRA pipeline (prototype/pipeline)

GNU Radio is OPTIONAL.  When the optional ``gnuradio`` python package is
not installed, the whole layer degrades to the synthetic offline source so
the GUI and tests keep working headlessly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from PySide6.QtCore import QThread, Signal as QtSignal

from prototype.core.logging_config import logger
from prototype.core.signal import Signal

from .config import (
    GNURadioAcquisitionConfig,
    GNURadioSourceConfig,
)
from .source import GNURadioSource
from .adapter import (
    signal_from_gnuradio_chunks,
    signal_from_gnuradio_source,
)

logger = logging.getLogger("spectra.io.gnuradio.gui_controller")


class GNURadioAcquisitionWorker(QThread):
    """
    Run a GNU Radio (synthetic) acquisition in a background thread.

    Signals
    -------
    finished_with_result
        Emitted with ``(signal: Signal, source_metadata: dict)`` or with
        a plain ``dict`` when the result was built from a saved replay
        file.
    failed
        Emitted with a human-readable error message.
    """

    finished_with_result = QtSignal(object)
    failed = QtSignal(str)

    # ------------------------------------------------------------
    # factory / config hooks (injected by the GUI or tests)
    # ------------------------------------------------------------

    #: Factory used to build a source when the real GNU Radio module is
    #: missing.  Defaults to the synthetic GNU Radio streamer.
    source_factory: Callable[..., "GNURadioSource | None"] = None

    def __init__(
        self,
        device_name: str = "synthetic-bpsk",
        center_frequency_hz: float = 800.0e6,
        sample_rate: float = 1_000_000.0,
        gain_db: float | None = 12.0,
        n_samples: int = 4096,
        chunk_size: int = 1 << 16,
        max_chunks: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.device_name = device_name
        self.center_frequency_hz = center_frequency_hz
        self.sample_rate = float(sample_rate)
        self.gain_db = gain_db
        self.n_samples = int(n_samples)
        self.chunk_size = int(chunk_size)
        self.max_chunks = int(max_chunks)

        self._acquisition = None
        self._source = None
        self._signal = None
        self._source_meta = None

        # Defaults to the synthetic GNU Radio source so headless tests
        # work without the optional gnuradio python package.
        self.source_factory = lambda **kw: GNURadioSource(
            GNURadioSourceConfig(
                center_frequency_hz=kw.get("center_frequency_hz", 800.0e6),
                sample_rate=kw.get("sample_rate", 1_000_000.0),
                iq_format="complex",
                gain_db=kw.get("gain_db"),
                device_name=kw.get("device_name", "synthetic-bpsk"),
            )
        )

    def run(self) -> None:  # noqa: D102 - Qt override
        try:
            self._build_and_convert()
        except Exception as exc:  # noqa: BLE001
            logger.exception("GNU Radio GUI acquisition failed")
            self.failed.emit(f"GNU Radio acquisition failed: {exc}")
            return

        self.finished_with_result.emit(
            {"signal": self._signal, "source_metadata": self._source_meta}
        )

    # ------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------

    def _build_and_convert(self) -> None:
        """Create the source, stream it, and convert to a Signal."""
        acquisition = GNURadioAcquisitionConfig(
            source=GNURadioSourceConfig(
                center_frequency_hz=self.center_frequency_hz,
                sample_rate=self.sample_rate,
                iq_format="complex",
                gain_db=self.gain_db,
                device_name=self.device_name,
            ),
            chunk_size=self.chunk_size,
            max_chunks=self.max_chunks,
        )

        # Resolve the source.  The factory either returns a real
        # GNURadioSource (when the optional gnuradio package is
        # importable) or a synthetic GNU Radio streamer implementation.
        source = self._resolve_source(acquisition)

        if source is None:
            # Synthetic fallback: the GNU Radio python package is not
            # installed; reuse the synthetic GNU Radio streamer adapter.
            signal, meta = self._synthetic_source(acquisition)
        else:
            signal = signal_from_gnuradio_source(
                source,
                acquisition,
            )
            meta = source.get_metadata()

        self._signal = signal
        self._source_meta = meta.to_dict()

    def _resolve_source(self, acquisition: GNURadioAcquisitionConfig):
        """Return a GNURadioSource, or None if the backend is unavailable."""
        try:
            from prototype.io.gnuradio import (
                GNURadioSource,
                make_gnuradio_source,
                gnuradio_available,
            )

            if not gnuradio_available():
                return None

            return make_gnuradio_source(acquisition)
        except Exception:  # noqa: BLE001
            logger.warning(
                "GNU Radio backend unavailable; using synthetic GNU Radio source",
                exc_info=True,
            )
            return None

    def _synthetic_source(self, acquisition: GNURadioAcquisitionConfig):
        """Build a short synthetic GNU Radio-style source for validation."""
        source_config = GNURadioSourceConfig(
            center_frequency_hz=self.center_frequency_hz,
            sample_rate=self.sample_rate,
            iq_format="complex",
            gain_db=self.gain_db,
            device_name=self.device_name,
        )

        source = GNURadioSource(source_config)

        signal = signal_from_gnuradio_source(
            source,
            acquisition,
        )

        return signal, source.get_metadata()

    # ------------------------------------------------------------
    # public accessors
    # ------------------------------------------------------------

    @property
    def signal(self) -> "Signal | None":
        return self._signal

    @property
    def source_metadata(self) -> dict[str, Any] | None:
        return self._source_meta

    def get_metadata(self) -> dict[str, Any]:
        """Complete capture metadata for the GUI display."""
        if self._source_meta is None:
            return {}

        meta = dict(self._source_meta)
        meta.setdefault("capture", {})
        meta["capture"]["file_format"] = "gnuradio"
        meta["capture"]["source_kind"] = "gnuradio"
        meta["capture"]["device_name"] = self.device_name
        return meta
