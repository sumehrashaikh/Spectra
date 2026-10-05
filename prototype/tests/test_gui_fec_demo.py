"""GUI coverage tests for the SIH feature surface (FEC / sync / recovered).

Covers the parts that were reachable from the backend but not from the
window:

* the FEC-mode selector reaches the worker as a canonical FECMode value;
* the frame/sync-word control is the only GUI path to the protocol stage,
  and it builds the same ``FrameConfig`` shape the CLI builds;
* AUTO identification results stored under ``demodulation`` reach the
  "Auto FEC" row;
* a BATCH payload populates the detail rows from its candidate instead of
  overwriting them with the batch envelope (the reported "wrong
  constellation / Unknown modulation after Analyze" regression);
* the results rows are not dead: a real analysis fills FEC, recovered
  bits, sync word and interleaving rows.

Run offscreen; no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from prototype.gui.window import MainWindow  # noqa: E402
from prototype.gui.worker import AnalysisWorker  # noqa: E402

_app: QApplication | None = None


@pytest.fixture(scope="session", autouse=True)
def _qt_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Analysis end dialogs must not block a headless run."""
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
        )


def _make_window() -> MainWindow:
    window = MainWindow()
    window.show()
    return window


# ---------------------------------------------------------------------------
# Controls reach the backend configuration
# ---------------------------------------------------------------------------


def test_fec_mode_selector_carries_canonical_values():
    window = _make_window()
    try:
        combo = window.fec_mode_combo
        assert [combo.itemData(i) for i in range(combo.count())] == [
            "auto",
            "manual",
            "none",
        ]
        combo.setCurrentIndex(combo.findData("manual"))
        assert combo.currentData() == "manual"
    finally:
        window.close()
        window.deleteLater()


def test_worker_accepts_canonical_fec_and_interleave_configuration():
    worker = AnalysisWorker(
        samples=np.zeros(64, dtype=np.complex128),
        sample_rate=1000.0,
        fec_mode="manual",
        fec_scheme="reedsolomon",
        interleaving_mode="manual",
        interleave_depth=8,
        interleave_family="pseudo_random",
    )
    assert worker._interleave_family == "pseudo_random"
    assert worker._interleave_depth == 8


def test_frame_search_control_builds_frame_config():
    window = _make_window()
    try:
        assert window.frame_checkbox.isChecked() is False
        assert window._frame_search_config() is None
        assert window.sync_word_edit.isEnabled() is False

        window.frame_checkbox.setChecked(True)
        assert window.sync_word_edit.isEnabled() is True
        window.sync_word_edit.setText("0xAA55AA55")
        window.data_bytes_spin.setValue(16)

        config = window._frame_search_config()
        assert config is not None
        assert config.sync_word == 0xAA55AA55
        assert config.data_bytes == 16
        assert config.sync_word_capacity == 32
    finally:
        window.close()
        window.deleteLater()


def test_frame_search_rejects_a_bad_sync_word():
    window = _make_window()
    try:
        window.frame_checkbox.setChecked(True)
        window.sync_word_edit.setText("not-a-number")
        with pytest.raises(ValueError):
            window._frame_search_config()
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------


def test_auto_identification_is_read_from_demodulation():
    """The pipeline stores the AUTO verdict under ``demodulation``."""
    window = _make_window()
    try:
        window.analysis = {"signal_detected": True, "detected_signals": []}
        window._apply_candidate_detail(
            {
                "classification": {"modulation": "16-QAM"},
                "demodulation": {
                    "fec_identification": {
                        "status": "AUTO_DETECTED",
                        "best_scheme": "reedsolomon",
                        "confidence": 70.0,
                    }
                },
            }
        )
        assert window._identification_result is not None
        assert window._identification_result["best_scheme"] == "reedsolomon"
        window.update_analysis_parameters()
        row = window.parameter_fec_auto.text()
        assert "reedsolomon" in row
        assert "70%" in row
    finally:
        window.close()
        window.deleteLater()


def test_auto_identification_row_formats_a_fractional_confidence():
    """Confidence is a heuristic score, not a probability: a 0..1 value
    must render as a percentage, never as "confidence 0"."""
    window = _make_window()
    try:
        window.analysis = {"signal_detected": True, "detected_signals": []}
        window._apply_candidate_detail(
            {
                "classification": {"modulation": "16-QAM"},
                "demodulation": {
                    "fec_identification": {
                        "status": "AUTO_DETECTED",
                        "best_scheme": "concatenated",
                        "confidence": 0.83,
                        "confirmed_by": "reference_payload_agreement",
                    }
                },
            }
        )
        window.update_analysis_parameters()
        row = window.parameter_fec_auto.text()
        assert "concatenated" in row
        assert "83%" in row
        assert "confidence 0" not in row
    finally:
        window.close()
        window.deleteLater()


def _batch_payload() -> dict:
    candidate = {
        "candidate_index": 0,
        "selected_candidate": {"center_frequency": 500.0},
        "parameters": {"snr_db": 12.5, "noise_power": 1e-6},
        "classification": {"modulation": "16-QAM", "confidence": 88.0},
        "symbol_rate": {"symbol_rate_hz": 100.0, "samples_per_symbol": 80.0},
        "synchronization": {"frequency_offset_hz": 0.1, "phase_offset_rad": 0.02},
        "demodulation": {
            "modulation": "16-QAM",
            "num_symbols": 256,
            "num_bits": 1024,
            "constellation": {"symbols": [[0.1, 0.2], [0.3, 0.4]]},
            "received_bits": [0, 1, 0, 1],
            "fec": {
                "scheme": "conv12",
                "corrected_errors": 1,
                "uncorrectable_blocks": 0,
                "decoded_bits": [1, 0, 1],
                "decoded_bit_count": 3,
                "source": "explicit_config",
            },
        },
        "ber": None,
        "protocol": {"sync_found": True, "sync_confidence": 0.9, "payload_bytes": b""},
        "ml": None,
        "fusion": None,
        "warnings": [],
    }
    return {
        "input": {"sample_rate": 8000.0, "num_samples": 4096},
        "detections": [{"center_frequency": 500.0, "bandwidth": 200.0}],
        "analyzed_candidates": [candidate],
        "warnings": [],
    }


def test_batch_payload_populates_candidate_detail():
    """Batch mode must show the candidate, not the batch envelope."""
    window = _make_window()
    try:
        window._apply_pipeline_result(_batch_payload())
        assert window.modulation_result == "16-QAM"
        assert window._pipeline_parameters.get("snr_db") == 12.5
        assert window._pipeline_candidates is not None
        assert len(window._pipeline_candidates) == 1
        demod = window._pipeline_demod_summary or {}
        assert demod.get("num_bits") == 1024
        assert (demod.get("fec") or {}).get("scheme") == "conv12"
        window.update_analysis_parameters()
        assert "conv12" in window.parameter_fec.text()
        assert "3" in window.parameter_recovered_bits.text()
    finally:
        window.close()
        window.deleteLater()


def test_empty_batch_clears_the_detail_state():
    window = _make_window()
    try:
        window._apply_pipeline_result(
            {
                "input": {"sample_rate": 8000.0, "num_samples": 10},
                "detections": [],
                "analyzed_candidates": [],
                "warnings": [],
            }
        )
        assert window.modulation_result == "Unknown"
        assert window._pipeline_parameters == {}
        assert window._pipeline_demod_summary is None
        assert window._pipeline_fec_summary is None
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# Real analysis through the window (deterministic demo capture)
# ---------------------------------------------------------------------------


def _run_window_analysis(window: MainWindow, samples, sample_rate, source) -> None:
    from prototype.core.loader import load_wav

    loaded, rate = load_wav(str(source))
    window.samples = loaded
    window.sample_rate = rate
    window.current_file = Path(source)
    assert np.asarray(loaded).size > 0
    window.fec_mode_combo.setCurrentIndex(window.fec_mode_combo.findData("manual"))
    window.fec_combo.setCurrentText("reedsolomon")
    window.interleaving_mode_combo.setCurrentText("Manual")
    window.interleave_family_combo.setCurrentIndex(
        window.interleave_family_combo.findData("block")
    )
    window.interleave_depth_spin.setValue(8)
    window.frame_checkbox.setChecked(True)

    window.analyze_current_signal()
    loop = QEventLoop()
    window.pipeline_worker.finished_with_result.connect(lambda *_: loop.quit())
    window.pipeline_worker.failed.connect(lambda m: loop.quit())
    QTimer.singleShot(180000, loop.quit)
    loop.exec()
    QApplication.processEvents()
    assert window.pipeline_worker is not None


def test_analysis_fills_the_fec_and_recovered_rows(tmp_path):
    from tests import demo_captures as dc

    capture = dc.build_capture(
        modulation="16-QAM",
        fec_scheme="reedsolomon",
        interleave_family="block",
        interleave_param=8,
        nbits=1024,
    )
    wav = tmp_path / "demo.wav"
    dc.write_wav(str(wav), capture.samples)

    window = _make_window()
    try:
        _run_window_analysis(window, capture.samples, dc.SAMPLE_RATE, wav)
        assert window.pipeline_result is not None, "analysis did not complete"
        assert window.modulation_result == "16-QAM"

        window.update_analysis_parameters()
        fec_text = window.parameter_fec.text()
        assert "reedsolomon" in fec_text
        assert "Recovered bits:" in window.parameter_recovered_bits.text()
        assert window.parameter_sync_word.text() != "Sync word: not run"
        assert "MANUALLY_CONFIGURED" in window.parameter_interleaving_mode.text()

        # Every visualisation tab must render after a single analysis.
        for index in range(window.vis_tabs.count()):
            window.vis_tabs.setCurrentIndex(index)
            window._refresh_active_plot()
            QApplication.processEvents()
        assert window.constellation_plot.canvas is not None
        title = window.constellation_plot.canvas.figure.axes[0].get_title()
        assert "Recovered" in title
    finally:
        window.close()
        window.deleteLater()
