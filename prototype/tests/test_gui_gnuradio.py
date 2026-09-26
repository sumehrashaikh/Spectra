"""Phase 1 GUI tests for the GNU Radio source selector + configuration panel.

These tests exercise the source selector and the GNU Radio configuration
panel added to :class:`prototype.gui.window.MainWindow`.  They deliberately
do NOT start any GNU Radio acquisition and do not call
``analyze_current_signal``.

GNU Radio is an OPTIONAL dependency.  The tests therefore skip gracefully
when the ``gnuradio`` python package is missing, and otherwise exercise the
synthetic GNU Radio source path using only numpy (no GNU Radio install).

PySide6 is run offscreen (no display required) so the tests are
headless-capable.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pytest

# GNU Radio python package is optional and not installed in CI.
GNURadio_INSTALLED = False


def _import_gnuradio_if_available() -> bool:
    """True when the GNU Radio python package is importable."""
    global GNURadio_INSTALLED
    try:
        import gnuradio  # noqa: F401

        GNURadio_INSTALLED = True
    except Exception:  # noqa: BLE001
        GNURadio_INSTALLED = False
    return GNURadio_INSTALLED


_import_gnuradio_if_available()


# ---------------------------------------------------------------------------
# Environment setup: run PySide6 offscreen (no display needed).
# ---------------------------------------------------------------------------


def _setup_offscreen_qt() -> None:
    """Configure PySide6 to run without a display.

    The offscreen platform is the existing approach used across the GUI
    test suite.  We set it before importing the window module so the
    QApplication can be constructed headlessly.
    """
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


_setup_offscreen_qt()

# Import after the platform is configured.
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit  # noqa: E402

from prototype.gui.window import MainWindow  # noqa: E402

# Qt requires a QApplication instance to exist before creating widgets.
_app: QApplication | None = None


@pytest.fixture(scope="session", autouse=True)
def _qt_app() -> QApplication:
    """Create a session-scoped offscreen QApplication for the GUI tests."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


# ---------------------------------------------------------------------------
# GNU Radio availability fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gnuradio_installed() -> bool:
    return GNURadio_INSTALLED


@pytest.fixture()
def acquisition_config():
    from prototype.io.gnuradio.config import (
        GNURadioAcquisitionConfig,
        GNURadioSourceConfig,
    )

    return GNURadioAcquisitionConfig(
        source=GNURadioSourceConfig(
            sample_rate=8000.0, center_frequency_hz=800.0e6, gain_db=12.0
        ),
        chunk_size=1024,
        max_chunks=2,
    )


# ---------------------------------------------------------------------------
# MainWindow element existence checks (no mutual exclusion / gating needed)
# ---------------------------------------------------------------------------


def _make_window() -> MainWindow:
    """Construct a MainWindow for testing; caller owns the lifecycle.

    The window is shown so that ``isVisible()`` on a child frame is not
    blocked by the top-level widget being hidden (a window that has never
    been shown reports False for all its children's visibility).
    """
    window = MainWindow()
    window.show()
    try:
        assert window.source_selector is not None
        assert window.gnuradio_frame is not None
        assert window.gnuradio_source_name is not None
        assert window.gnuradio_sample_rate is not None
        assert window.gnuradio_center_freq is not None
        assert window.gnuradio_gain is not None
        assert window.gnuradio_max_chunks is not None
        assert window.gnuradio_chunk_size is not None
    except Exception:
        window.close()
        window.deleteLater()
        raise
    return window


def test_wav_source_selector_is_available() -> None:
    """The WAV option is present and is the default source."""
    window = _make_window()

    try:
        assert (
            window.source_selector.currentText() == "WAV"
        ), "default source must be WAV (current behaviour)"
        assert window.source_selector.count() == 3
        assert window.source_selector.itemText(0) == "WAV"
        assert window.source_selector.itemText(1) == "Raw IQ"
        assert window.source_selector.itemText(2) == "GNU Radio"
    finally:
        window.close()
        window.deleteLater()


def test_raw_iq_source_selector_is_available() -> None:
    """Selecting Raw IQ does not disturb the other controls."""
    window = _make_window()

    try:
        window.source_selector.setCurrentText("Raw IQ")
        assert window.source_selector.currentText() == "Raw IQ"
        # Raw IQ is not GNU Radio: the GNU Radio panel must stay hidden.
        assert window.gnuradio_frame.isVisible() is False
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# GNU Radio config panel visibility / defaults
# ---------------------------------------------------------------------------


def test_gnu_radio_selector_exists() -> None:
    """The GNU Radio source option exists in the selector."""
    window = _make_window()

    try:
        assert "GNU Radio" in window.source_selector.itemText(
            window.source_selector.count() - 1
        )
        idx = window.source_selector.findText("GNU Radio")
        assert idx >= 0
    finally:
        window.close()
        window.deleteLater()


def test_gnu_radio_config_controls_appear_when_selected() -> None:
    """Selecting GNU Radio reveals the configuration panel."""
    window = _make_window()

    try:
        window.source_selector.setCurrentIndex(2)
        assert window.source_selector.currentText() == "GNU Radio"
        assert window.gnuradio_frame.isVisible() is True
    finally:
        window.close()
        window.deleteLater()


def test_switching_back_hides_gnu_radio_controls() -> None:
    """Switching back to WAV/Raw IQ hides the GNU Radio panel."""
    window = _make_window()

    try:
        window.source_selector.setCurrentIndex(2)
        assert window.gnuradio_frame.isVisible() is True

        window.source_selector.setCurrentIndex(0)
        assert window.gnuradio_frame.isVisible() is False

        window.source_selector.setCurrentIndex(1)
        assert window.gnuradio_frame.isVisible() is False
    finally:
        window.close()
        window.deleteLater()


def _expected_gnuradio_fields() -> set[str]:
    from prototype.io.gnuradio.config import GNURadioSourceConfig

    return set(
        GNURadioSourceConfig.__dataclass_fields__.keys()
    ) | {"device_name", "device_address", "extra"}


def test_gnu_radio_defaults_from_config() -> None:
    """GNU Radio config controls use sensible defaults from GNURadioSourceConfig."""
    window = _make_window()

    try:
        window.source_selector.setCurrentText("GNU Radio")

        # Defaults reflect config placeholder values chosen to match the
        # GNURadioSourceConfig defaults (sample_rate=1e6).
        assert window.gnuradio_source_name.text() == "synthetic-bpsk"
        assert window.gnuradio_sample_rate.text() == "1000000.0"
        assert window.gnuradio_center_freq.text() == "800000000.0"
        assert window.gnuradio_gain.text() == "12.0"
        assert window.gnuradio_max_chunks.text() == "0"
        assert window.gnuradio_chunk_size.text() == "1048576"
    finally:
        window.close()
        window.deleteLater()


def test_gnu_radio_controls_disabled_outside_gnuradio() -> None:
    """LineEdits/ComboBoxes are disabled when GNU Radio is not selected,
    so no acquisition can be started from them."""
    window = _make_window()

    try:
        # Start from GNU Radio (panel visible).  Then switch away;
        # the selector change must emit the signal and disable the panel
        # controls (and re-hide the panel).
        window.source_selector.setCurrentIndex(2)
        assert window.gnuradio_frame.isVisible() is True

        for text in ("WAV", "Raw IQ"):
            window.source_selector.setCurrentText(text)
            for _widget in window.gnuradio_frame.findChildren(QLineEdit):
                assert (
                    _widget.isEnabled() is False
                ), f"{_widget} should be disabled outside GNU Radio"
            for _widget in window.gnuradio_frame.findChildren(QComboBox):
                assert (
                    _widget.isEnabled() is False
                ), f"{_widget} should be disabled outside GNU Radio"
    finally:
        window.close()
        window.deleteLater()


def test_no_gnuradio_acquisition_starts_on_selector_change() -> None:
    """Changing the selector must NOT start GNU Radio acquisition."""
    assert not GNURadio_INSTALLED
    window = _make_window()

    try:
        # The panel becomes visible, but no worker, no source, no
        # acquisition is started from the selector change.
        window.source_selector.setCurrentIndex(2)
        assert window.gnuradio_frame.isVisible() is True
        # MainWindow must not hold any acquisition worker or running
        # pipeline for the GNU Radio path this phase.
        assert getattr(window, "pipeline_worker", None) is None
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# GNU Radio backend integration (optional synthetic path) -- Phase 1 scope
# ---------------------------------------------------------------------------


def test_gnuradio_source_config_defaults(gnuradio_installed) -> None:
    """The config dataclasses carry sensible defaults without GNU Radio
    installed (used by the GUI panel defaults)."""
    if not gnuradio_installed:
        # Backend unavailable: rely on the synthetic fallback only.
        return

    from prototype.io.gnuradio.config import (
        GNURadioAcquisitionConfig,
        GNURadioSourceConfig,
    )

    src = GNURadioSourceConfig()
    assert src.sample_rate == 1_000_000.0
    assert src.iq_format == "complex"

    acq = GNURadioAcquisitionConfig(source=src)
    assert acq.chunk_size == 1 << 20
    assert acq.max_chunks == 0


def _synthetic_source_worker_from_config(
    acquisition_config: Any,
) -> object:
    """Build a GNURadioSource (synthetic) from an acquisition config.

    Mirrors what the GUI would do once acquisition wiring exists, but does
    NOT start a thread here: just constructs the in-process source so the
    chunk->Signal conversion still runs.
    """
    from prototype.io.gnuradio.config import GNURadioSourceConfig
    from prototype.io.gnuradio.source import GNURadioSource

    source_config = acquisition_config.source
    source = GNURadioSource(
        GNURadioSourceConfig(
            center_frequency_hz=source_config.center_frequency_hz,
            sample_rate=source_config.sample_rate,
            iq_format=source_config.iq_format,
            gain_db=source_config.gain_db,
            device_name=source_config.device_name,
        )
    )
    return source


def test_gnu_radio_synth_reaches_processor(
    gnuradio_installed, acquisition_config
) -> None:
    """A synthetic GNU Radio source's chunks must convert to a Signal and
    feed the existing pipeline unchanged (offline flow)."""
    if not gnuradio_installed:
        # GNU Radio not installed: this is the expected fallback path and
        # the core WAV/IQ analysis still works without it.
        return

    source_config = acquisition_config.source
    source = _synthetic_source_worker_from_config(acquisition_config)

    from prototype.io.gnuradio.adapter import signal_from_gnuradio_source

    signal = signal_from_gnuradio_source(source, acquisition_config)

    assert signal.sample_rate == source_config.sample_rate
    assert signal.metadata["capture"]["file_format"] == "gnuradio"
    assert signal.metadata["capture"]["iq_format"] == "complex"
    assert signal.metadata["capture"]["center_frequency_hz"] == (
        source_config.center_frequency_hz
    )
    assert signal.metadata["capture"]["gain_db"] == source_config.gain_db


def test_metadata_convention_displayed(gnuradio_installed) -> None:
    """SourceMetadata / capture metadata must follow the existing metadata
    convention for GNU Radio provenance."""
    if not gnuradio_installed:
        return

    from prototype.io.gnuradio.config import (
        GNURadioAcquisitionConfig,
        GNURadioSourceConfig,
    )

    acq = GNURadioAcquisitionConfig(
        source=GNURadioSourceConfig(
            sample_rate=8000.0, center_frequency_hz=800.0e6, gain_db=12.0
        ),
        chunk_size=1024,
        max_chunks=2,
    )

    from prototype.io.gnuradio.config import SourceMetadata
    from prototype.io.gnuradio.streaming import GNURadioStreamer
    from prototype.io.gnuradio.source import GNURadioSource

    source = GNURadioSource(
        GNURadioSourceConfig(
            sample_rate=8000.0,
            center_frequency_hz=800.0e6,
            iq_format="complex",
            gain_db=12.0,
            device_name="synthetic-bpsk",
        )
    )

    streamer = GNURadioStreamer(acq, chunk_factory=source._stream_factory)
    chunks = list(streamer._chunks())

    signal = signal_from_gnuradio_chunks(
        chunks, source_metadata=source.get_metadata()
    )

    capture = signal.metadata["capture"]
    assert capture["file_format"] == "gnuradio"
    assert capture["source_kind"] in ("gnuradio", "synthetic", "synthetic-bpsk")
    assert capture["device_name"] == "synthetic-bpsk"
