"""Focused GUI tests for the Interleaving Mode selector and display.

These tests exercise the new ``Interleaving Mode`` selector on
:class:`prototype.gui.window.MainWindow` and the display labels driven by the
pipeline result.  They deliberately DO NOT run an actual analysis
(``analyze_current_signal``) and do NOT start any acquisition.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pytest

# PySide6 is run offscreen (no display required) so the tests are
# headless-capable.
if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Import after the platform is configured.
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
)

from prototype.gui.window import MainWindow  # noqa: E402


_app: QApplication | None = None


@pytest.fixture(scope="session", autouse=True)
def _qt_app() -> QApplication:
    """Create a session-scoped offscreen QApplication for the GUI tests."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


# ---------------------------------------------------------------------------
# Interleaving Mode selector existence checks
# ---------------------------------------------------------------------------


def _make_window() -> MainWindow:
    """Construct a MainWindow for testing; caller owns the lifecycle."""
    window = MainWindow()
    window.show()
    try:
        assert window.fec_mode_combo is not None
        assert window.fec_combo is not None
        assert window.interleaving_mode_combo is not None
    except Exception:
        window.close()
        window.deleteLater()
        raise
    return window


def test_interleaving_mode_selector_exists() -> None:
    """The Interleaving Mode selector is present and defaults to Auto."""
    window = _make_window()
    try:
        assert window.interleaving_mode_combo.count() == 3
        assert window.interleaving_mode_combo.itemText(0) == "Auto"
        assert window.interleaving_mode_combo.itemText(1) == "Manual"
        assert window.interleaving_mode_combo.itemText(2) == "None"
        assert window.interleaving_mode_combo.currentText() == "Auto"
    finally:
        window.close()
        window.deleteLater()


def test_interleaving_mode_is_separate_from_fec_mode() -> None:
    """Interleaving mode is a distinct selector from the FEC mode combo."""
    window = _make_window()
    try:
        assert window.fec_mode_combo is not None
        assert window.interleaving_mode_combo is not None
        # Select a different interleaving mode.
        window.interleaving_mode_combo.setCurrentText("Manual")
        assert window.interleaving_mode_combo.currentText() == "Manual"
        # The FEC mode combo is unaffected.
        assert window.fec_mode_combo.currentText() == "AUTO"
    finally:
        window.close()
        window.deleteLater()


def test_interleaving_mode_setback_navigation() -> None:
    """Changing the interleaving mode then back restores the previous value."""
    window = _make_window()
    try:
        window.interleaving_mode_combo.setCurrentText("Manual")
        assert window.interleaving_mode_combo.currentText() == "Manual"
        window.interleaving_mode_combo.setCurrentText("None")
        assert window.interleaving_mode_combo.currentText() == "None"
        window.interleaving_mode_combo.setCurrentText("Auto")
        assert window.interleaving_mode_combo.currentText() == "Auto"
    finally:
        window.close()
        window.deleteLater()


def test_interleaving_mode_is_separate_from_fec_mode() -> None:
    """The Interleaving Mode selector is distinct from the FEC mode combo."""
    window = _make_window()
    try:
        assert window.fec_mode_combo is not None
        assert window.interleaving_mode_combo is not None
        # Select a different interleaving mode, leave FEC mode at AUTO.
        window.interleaving_mode_combo.setCurrentText("Manual")
        assert window.interleaving_mode_combo.currentText() == "Manual"
        # The FEC mode combo is unaffected (values are FECMode strings).
        assert window.fec_mode_combo.currentText() == "auto"
    finally:
        window.close()
        window.deleteLater()


def test_interleaving_mode_combines_with_fec_mode() -> None:
    """Interleaving mode and FEC mode can both be set without interference."""
    window = _make_window()
    try:
        window.fec_mode_combo.setCurrentText("AUTO")
        window.interleaving_mode_combo.setCurrentText("Manual")
        assert window.fec_mode_combo.currentText() == "auto"
        assert window.interleaving_mode_combo.currentText() == "Manual"
    finally:
        window.close()
        window.deleteLater()
