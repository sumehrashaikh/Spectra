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
        # The FEC mode combo is unaffected. Its label is title-case for
        # readability and carries the canonical FECMode value as data.
        assert window.fec_mode_combo.currentText() == "Auto"
        assert window.fec_mode_combo.currentData() == "auto"
    finally:
        window.close()
        window.deleteLater()


def test_interleaving_mode_combines_with_fec_mode() -> None:
    """Interleaving mode and FEC mode can both be set without interference."""
    window = _make_window()
    try:
        window.fec_mode_combo.setCurrentIndex(
            window.fec_mode_combo.findData("auto")
        )
        window.interleaving_mode_combo.setCurrentText("Manual")
        assert window.fec_mode_combo.currentData() == "auto"
        assert window.interleaving_mode_combo.currentText() == "Manual"
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# Extended de-interleaving family selector
# ---------------------------------------------------------------------------


def test_interleave_family_selector_exists() -> None:
    """The family selector offers all four families and defaults to block."""
    window = _make_window()
    try:
        combo = window.interleave_family_combo
        assert combo is not None
        assert combo.count() == 4
        assert combo.currentData() == "block"
        values = {combo.itemData(i) for i in range(combo.count())}
        assert values == {"block", "convolutional", "diagonal", "pseudo_random"}
        # disabled until Manual interleaving is selected
        assert combo.isEnabled() is False
    finally:
        window.close()
        window.deleteLater()


def test_interleave_family_enabled_only_for_manual() -> None:
    """The family selector is enabled only when Interleaving = Manual."""
    window = _make_window()
    try:
        window.interleaving_mode_combo.setCurrentText("Manual")
        assert window.interleave_family_combo.isEnabled() is True
        window.interleaving_mode_combo.setCurrentText("Auto")
        assert window.interleave_family_combo.isEnabled() is False
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# Recovered-information and sync-word read-out
# ---------------------------------------------------------------------------


def test_recovered_bits_and_sync_word_labels() -> None:
    """Decoded bit count and sync-word/frame info reach the read-out."""
    window = _make_window()
    try:
        window.analysis = {"signal_detected": True}
        window._pipeline_demod_summary = {
            "num_symbols": 10,
            "num_bits": 20,
            "decision_margin": 0.5,
        }
        window._pipeline_fec_summary = {
            "scheme": "reedsolomon",
            "corrected_errors": 1,
            "uncorrectable_blocks": 0,
            "decoded_bit_count": 256,
            "source": "explicit_config",
        }
        window._pipeline_protocol_summary = {
            "sync_found": True,
            "sync_confidence": 0.93,
            "payload_bytes": b"\x01\x02\x03",
        }
        window.update_analysis_parameters()

        recovered = window.parameter_recovered_bits.text()
        assert "256" in recovered and "decoded" in recovered
        sync = window.parameter_sync_word.text()
        assert "found" in sync and "3 payload bytes" in sync
    finally:
        window.close()
        window.deleteLater()


def test_recovered_bits_without_fec_and_sync_not_found() -> None:
    """Honest fallbacks: received-only bits and a negative sync result."""
    window = _make_window()
    try:
        window.analysis = {"signal_detected": True}
        window._pipeline_demod_summary = {"num_symbols": 8, "num_bits": 16}
        window._pipeline_fec_summary = None
        window._pipeline_protocol_summary = {"sync_found": False}
        window.update_analysis_parameters()

        assert "16" in window.parameter_recovered_bits.text()
        assert "received" in window.parameter_recovered_bits.text()
        assert window.parameter_sync_word.text() == "Sync word: not found"
    finally:
        window.close()
        window.deleteLater()
