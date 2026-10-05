"""GUI: light/dark theme toggle + GNU Radio spectrum/waterfall toggle.

Both features are additive: the theme toggle lives in the title row (a
corner button, no layout change to the existing controls) and the GNU
Radio visualization toggle lives inside the existing GNU Radio tab.
GNU Radio is optional, so the visualization assertions hold whether or
not it is installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from prototype.gui.window import MainWindow  # noqa: E402

_app: QApplication | None = None


@pytest.fixture(scope="session", autouse=True)
def _qt_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


@pytest.fixture()
def window() -> MainWindow:
    win = MainWindow()
    win.show()
    try:
        yield win
    finally:
        win.close()
        win.deleteLater()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def test_theme_toggle_lives_in_the_title_row(window):
    assert window.theme_button is not None
    assert window.theme_button.isEnabled()
    assert window.theme_mode in ("light", "dark")


def test_theme_starts_light_and_toggles(window):
    assert window.theme_mode == "light"

    window.toggle_theme()
    assert window.theme_mode == "dark"
    assert "Light" in window.theme_button.text()

    window.toggle_theme()
    assert window.theme_mode == "light"
    assert "Dark" in window.theme_button.text()


def test_theme_is_applied_to_every_plot_canvas(window):
    rng = np.random.default_rng(0)
    window.samples = (
        rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    ).astype(np.complex64)
    window.sample_rate = 8000.0

    window.vis_tabs.setCurrentIndex(1)
    window._draw_spectrum()

    window.set_theme("dark")
    plots = window._plot_widgets()
    assert plots
    for plot in plots:
        assert plot.theme_mode == "dark"

    figure = window.spectrum_plot.canvas.figure
    dark_background = figure.get_facecolor()
    assert dark_background[0] < 0.5, "canvas should be dark after set_theme('dark')"

    window.set_theme("light")
    assert window.spectrum_plot.canvas.figure.get_facecolor()[0] > 0.5


def test_ml_line_can_render_without_an_analysis(window):
    """Regression: the summary asked for a local ``modulation`` that does not
    exist in ``_pipeline_summary_text``, so any analysis with the ML toggle
    on failed with ``NameError: name 'modulation' is not defined``."""

    window.modulation_result = "16-QAM"
    window._pipeline_ml_summary = {
        "predicted_class": "ASK4",
        "predicted_class_canonical": "ASK4",
        "confidence": 0.36,
        "frame_agreement": 0.5,
        "validated": False,
        "validation_accuracy": 0.34375,
        "presented_as": "evidence",
        "top3": [
            {"class": "ASK4", "class_canonical": "ASK4", "score": 0.36},
            {"class": "OOK", "class_canonical": "OOK", "score": 0.22},
        ],
    }

    summary = window._pipeline_summary_text({})

    assert "Modulation: 16-QAM" in summary
    assert "ML Prediction:" in summary
    assert "from DSP analysis" in summary
    assert "ML top-3:" in summary
    assert "ASK4" in summary


def test_ml_row_shows_the_mirrored_dsp_class(window):
    """An unvalidated CNN must not put its own argmax in the ML row: the
    deterministic classification is displayed instead, and the network's
    answer is named as the raw class."""

    window.modulation_result = "16-QAM"
    window._pipeline_ml_summary = {
        "predicted_class": "ASK4",
        "ml_raw_class": "ASK4",
        "display_class": "16-QAM",
        "display_confidence": 0.9312,
        "display_source": "dsp_mirror",
        "mirrored_from_dsp": True,
        "validated": False,
        "presented_as": "evidence",
        "top3": [],
    }

    summary = window._pipeline_summary_text({})

    assert "ML Prediction: 16-QAM (93%)" in summary
    assert "CNN said ASK4" in summary

    window.analysis = {
        "signal_detected": True,
        "sample_rate": 8000.0,
        "detected_signals": [],
    }
    window.update_analysis_parameters()
    assert "16-QAM" in window.parameter_ml.text()


def test_ml_line_flags_disagreement_with_the_dsp_result(window):
    window.modulation_result = "16-QAM"
    window._pipeline_ml_summary = {
        "predicted_class": "QPSK",
        "predicted_class_canonical": "QPSK",
        "confidence": 0.9,
        "frame_agreement": 1.0,
        "validated": True,
        "validation_accuracy": 0.9,
        "presented_as": "prediction",
        "top3": [],
    }

    summary = window._pipeline_summary_text({})

    assert "ML Prediction: QPSK (90%)" in summary
    assert "disagrees with DSP (16-QAM)" in summary


def test_ml_parameter_row_renders_with_an_ml_summary(window):
    window.analysis = {
        "signal_detected": True,
        "sample_rate": 8000.0,
        "detected_signals": [],
    }
    window.modulation_result = "QPSK"
    window._pipeline_ml_summary = {
        "predicted_class": "QPSK",
        "predicted_class_canonical": "QPSK",
        "confidence": 0.8,
        "frame_agreement": 1.0,
        "validated": True,
        "validation_accuracy": 0.9,
        "presented_as": "prediction",
        "top3": [],
    }

    window.update_analysis_parameters()

    assert window.parameter_ml.text() == "ML Prediction: QPSK (80%)"


def test_light_theme_keeps_the_original_stylesheet(window):
    window.set_theme("light")
    app = QApplication.instance()
    assert app.styleSheet() in ("", None) or "QWidget" not in app.styleSheet()


# ---------------------------------------------------------------------------
# Symbols / bits inspectors
# ---------------------------------------------------------------------------


def _populate_analysis(window):
    """Install a synthetic analysis payload for the inspector dialogs."""

    window.modulation_result = "16-QAM"
    window._pipeline_demod_summary = {
        "num_symbols": 4,
        "num_bits": 16,
        "decision_margin": 0.42,
        "constellation": {
            "symbols": [
                [0.316, 0.316],
                [-0.316, 0.316],
                [-0.316, -0.316],
                [0.316, -0.316],
            ]
        },
        "received_bits": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        "aligned_bits": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        "quality_estimate": {
            "method": "evm_estimate",
            "evm_percent": 6.22,
            "snr_db": 24.13,
            "ber_estimate": 1e-09,
        },
    }
    window._pipeline_fec_summary = {
        "scheme": "reedsolomon",
        "decoded_bits": [1, 0, 1, 1],
        "decoded_bit_count": 4,
        "uncorrectable_blocks": 0,
        "post_fec_ber": 0.0,
    }
    window._interleaving_result = {"status": "NONE", "family": None, "best_depth": None}
    window._identification_result = {"status": "AUTO_DETECTED", "best_scheme": "reedsolomon"}
    window._pipeline_ber_summary = None


def test_symbols_dialog_lists_the_recovered_symbols(window, monkeypatch):
    _populate_analysis(window)

    captured = {}

    def _fake_dialog(title, text, save_name="spectra.txt"):
        captured["title"] = title
        captured["text"] = text
        captured["save_name"] = save_name

    monkeypatch.setattr(window, "_show_text_dialog", _fake_dialog)

    window.show_symbols_dialog()

    assert captured["title"] == "Recovered Symbols"
    assert "Recovered symbols: 4" in captured["text"]
    assert "+0.31600000" in captured["text"]
    assert captured["save_name"] == "symbols.txt"


def test_symbols_dialog_explains_when_nothing_is_recovered(window, monkeypatch):
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: messages.append(a[2] if len(a) > 2 else a)),
    )

    window.show_symbols_dialog()

    assert messages and "No recovered symbols" in str(messages[0])


def test_bits_dialog_lists_every_bitstream_and_the_ber_readout(
    window, monkeypatch
):
    _populate_analysis(window)

    captured = {}

    monkeypatch.setattr(
        window,
        "_show_text_dialog",
        lambda title, text, save_name="spectra.txt": captured.update(
            {"title": title, "text": text}
        ),
    )

    window.show_bits_dialog()

    text = captured["text"]
    assert captured["title"] == "Bit Stream / BER"
    assert "Demodulated (received) (16 bits)" in text
    assert "Codeword-aligned" in text
    assert "FEC-decoded payload (4 bits)" in text
    assert "1011" in text
    assert "EVM 6.22%" in text
    assert "BER est" in text
    assert "Interleaving: NONE" in text
    assert "Auto FEC: AUTO_DETECTED" in text


def test_ber_row_uses_the_evm_estimate_without_a_reference(window):
    _populate_analysis(window)

    text = window._ber_display_text()

    assert "not measured (no reference)" in text
    assert "EVM 6.22%" in text
    assert "SNR est 24.1 dB" in text


def test_ber_row_uses_the_measured_ber_when_available(window):
    _populate_analysis(window)
    window._pipeline_ber_summary = {
        "ber": 0.00304878,
        "bit_errors": 4,
        "compared_bits": 1312,
    }

    text = window._ber_display_text()

    assert text.startswith("BER: 0.00304878")
    assert "4/1312 bits" in text
    assert "post-FEC 0" in text


def test_interleaving_rows_explain_none_and_unresolved():
    """No bare status codes: each outcome says what happened."""
    from prototype.gui.window import MainWindow

    none_required = {
        "status": "NONE",
        "family": None,
        "best_depth": None,
        "evidence": {"mode": "reference_agreement"},
    }
    blind = {"status": "UNRESOLVED", "family": None, "best_depth": None}
    detected = {"status": "AUTO_DETECTED", "best_type": "block", "best_depth": 8}

    assert "none required" in MainWindow._interleaving_family_label(none_required)
    assert "no deinterleaving applied" in MainWindow._interleaving_depth_text(
        none_required
    )
    assert "not identifiable" in MainWindow._interleaving_family_label(blind)
    assert "none applied" in MainWindow._interleaving_depth_text(blind)
    assert MainWindow._interleaving_family_label(detected) == "block"
    assert MainWindow._interleaving_depth_text(detected) == "8"


def test_ber_reset_rows_are_not_dead_ends(window):
    """A fresh capture says the BER is pending, never "no reference loaded"."""

    assert "not measured yet" in window.results_il_ber_label.text()
    assert "No reference loaded" not in window.parameter_ber.text()


def test_inspector_dialogs_are_reachable(window):
    assert window.symbols_button.text() == "View Symbols (I/Q)"
    assert window.bits_button.text() == "View Bits / BER"
    assert window.results_symbols_button.text() == "View Symbols (I/Q)"
    assert window.results_bits_button.text() == "View Bits / BER"


def test_text_dialog_renders_and_closes(window, monkeypatch):
    """The dialog itself must construct and close (no modal hang)."""

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog, QPlainTextEdit

    monkeypatch.setattr(QDialog, "exec", lambda self: 0)

    window._show_text_dialog("Smoke", "line one\nline two")

    editors = [
        child
        for child in window.findChildren(QPlainTextEdit)
    ]
    # The dialog is a child of the window; the editor carries the text.
    assert any(editor.toPlainText().startswith("line one") for editor in editors)


# ---------------------------------------------------------------------------
# GNU Radio spectrum / waterfall (visualization only)
# ---------------------------------------------------------------------------


def test_gnuradio_viz_toggle_exists_and_defaults_off(window):
    assert window.gnuradio_viz_checkbox is not None
    assert window.gnuradio_viz_checkbox.isChecked() is False
    assert window.gnuradio_viz_status.text() != ""


def test_toggle_off_never_runs_the_flowgraph(window, monkeypatch):
    from prototype.io.gnuradio import viz

    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("the flowgraph must not run while disabled")

    monkeypatch.setattr(viz, "compute_spectrum_waterfall", _boom)

    rng = np.random.default_rng(0)
    window.samples = (rng.standard_normal(4096) + 0j).astype(np.complex64)
    window.sample_rate = 8000.0

    assert window._gnuradio_spectrum_waterfall() is None


def test_toggle_on_falls_back_honestly_without_gnuradio(window, monkeypatch):
    from prototype.io.gnuradio import viz

    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (False, "not installed (test)")
    )

    rng = np.random.default_rng(1)
    window.samples = (
        rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    ).astype(np.complex64)
    window.sample_rate = 8000.0

    window.gnuradio_viz_checkbox.setChecked(True)

    # The NumPy plot path still works and the status row says why.
    window.vis_tabs.setCurrentIndex(1)
    window._draw_spectrum()
    window.vis_tabs.setCurrentIndex(2)
    window._draw_waterfall()

    assert "NumPy" in window.gnuradio_viz_status.text()
    assert window._gnuradio_spectrum_waterfall() is None


def test_toggle_on_uses_gnuradio_matrices_when_available(window, monkeypatch):
    """A successful flowgraph result is plotted on the Spectrum tab."""
    from prototype.io.gnuradio import viz

    freqs = np.linspace(-4000.0, 4000.0, 64)
    result = {
        "status": "ok",
        "backend_used": "gnuradio",
        "source": "GNU Radio",
        "fft_size": 64,
        "n_frames": 4,
        "spectrum": {
            "freqs": freqs.tolist(),
            "magnitude": (np.abs(freqs) + 1.0).tolist(),
            "power": (np.abs(freqs) + 1.0).tolist(),
            "power_db": (10 * np.log10(np.abs(freqs) + 1.0)).tolist(),
        },
        "waterfall": {
            "freqs": freqs.tolist(),
            "times": np.linspace(0, 0.1, 4).tolist(),
            "power_db": np.zeros((64, 4)).tolist(),
        },
        "provenance": {"duration_s": 0.01},
    }

    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (True, "GNU Radio test")
    )
    monkeypatch.setattr(
        viz, "compute_spectrum_waterfall", lambda *a, **k: result
    )

    rng = np.random.default_rng(2)
    window.samples = (
        rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    ).astype(np.complex64)
    window.sample_rate = 8000.0

    window.gnuradio_viz_checkbox.setChecked(True)
    window._draw_spectrum()
    window._draw_waterfall()

    assert "GNU Radio" in window.gnuradio_viz_status.text()
    title = window.spectrum_plot.canvas.figure.axes[0].get_title()
    assert "GNU Radio" in title
    waterfall_title = window.waterfall_plot.canvas.figure.axes[0].get_title()
    assert "GNU Radio" in waterfall_title
