"""Render window + capture the three plots after analysis."""
import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QEventLoop, QTimer

app = QApplication.instance() or QApplication(sys.argv)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)

from prototype.gui.window import MainWindow
win = MainWindow()
win.resize(1500, 2200)
win.show()
app.processEvents()

from prototype.core.loader import load_wav
from pathlib import Path
samples, fs = load_wav("gui_qam16.wav")
win.samples = samples
win.sample_rate = fs
win.current_file = Path("gui_qam16.wav")

# Visualizations before analysis (waveform/spectrum/waterfall load on open)
from prototype.visualization.plots import create_time_figure, create_spectrum_figure, create_waterfall_figure
try:
    win.time_plot.set_figure(create_time_figure(samples, fs))
    win.spectrum_plot.set_figure(create_spectrum_figure(samples, fs))
    win.waterfall_plot.set_figure(create_waterfall_figure(samples, fs))
except Exception as e:
    print("prefill viz err:", e, flush=True)
app.processEvents()
win.grab().save("gui_full_loaded.png")
print("captured loaded", flush=True)

win.analyze_current_signal()
loop = QEventLoop()
win.pipeline_worker.finished_with_result.connect(lambda *_: loop.quit())
win.pipeline_worker.failed.connect(lambda m: print("failed:", m))
QTimer.singleShot(60000, loop.quit)
loop.exec()
app.processEvents()
win.grab().save("gui_full_analyzed.png")
print("captured analyzed", flush=True)
win.close()
print("SCREENSHOTS DONE", flush=True)
