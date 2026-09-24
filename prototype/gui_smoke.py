import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QEventLoop, QTimer

app = QApplication.instance() or QApplication(sys.argv)

# Auto-dismiss modal dialogs: without this, _apply_pipeline_result's summary
# dialog blocks the event loop and the smoke test times out spuriously.
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)

from prototype.gui.window import MainWindow
print("import ok", flush=True)
win = MainWindow()
print("window built", flush=True)

from scipy.signal import upfirdn
from prototype.parameters.symbol_rate import rrc_filter

fs, sps = 8000.0, 80
rng = np.random.default_rng(5)
bits = rng.integers(0, 2, 4000)
syms = (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)
taps = rrc_filter(sps, rolloff=0.35)
shaped = upfirdn(taps, syms, up=sps)[:4000 * sps]
t = np.arange(shaped.size) / fs
iq = shaped * np.exp(1j * 2 * np.pi * 500 * t)
iq += 0.02 * (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size))
stereo = np.stack([iq.real, iq.imag], axis=1).astype(np.float32)
from scipy.io import wavfile

wavfile.write("gui_smoke.wav", int(fs), stereo)

from prototype.core.loader import load_wav

samples, sample_rate = load_wav("gui_smoke.wav")
win.samples = samples
win.sample_rate = sample_rate
from pathlib import Path

win.current_file = Path("gui_smoke.wav")
print("loaded:", samples.size, flush=True)

win.analyze_current_signal()
print("worker started:", win.pipeline_worker.isRunning(), flush=True)
loop = QEventLoop()
win.pipeline_worker.finished_with_result.connect(lambda *_: loop.quit())
win.pipeline_worker.failed.connect(
    lambda m: (print("WORKER FAILED:", m, flush=True), loop.quit())
)
QTimer.singleShot(180000, lambda: (print("TIMEOUT", flush=True), loop.quit()))
loop.exec()

r = getattr(win, "pipeline_result", None)
print("result present:", r is not None, flush=True)
if isinstance(r, dict):
    cls = r.get("classification") or {}
    rate = r.get("symbol_rate") or {}
    print("modulation:", cls.get("modulation"), flush=True)
    print("confidence:", cls.get("confidence"), flush=True)
    print("symbol rate:", rate.get("symbol_rate_hz") if isinstance(rate, dict) else rate, flush=True)
    print("ber:", r.get("ber"), flush=True)
elif r is not None:
    print("modulation:", getattr(r, "modulation", None), flush=True)
    print("symbol rate:", getattr(r, "symbol_rate", None), flush=True)
    print("ber:", getattr(r, "ber", None), flush=True)
win.close()
print("SMOKE DONE", flush=True)
