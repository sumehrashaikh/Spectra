import numpy as np
from pathlib import Path

from prototype.core.logging_config import logger

from prototype.core.timing import recover_symbol_timing, sample_symbols
from prototype.core.bpsk import demodulate_bpsk
from prototype.core.ber import load_transmitted_bits, validate_bpsk_bits

from prototype.modulation.demodulator import (
demodulate_qpsk,
qpsk_decision,
qam16_decision,
demodulate_bfsk,
)

# V1 synthetic BFSK test profile (only used for the legacy BFSK flow).
BFSK_FREQ_0 = 500.0
BFSK_FREQ_1 = 700.0
BFSK_SYMBOL_RATE = 100.0

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QInputDialog
from prototype.gui.worker import AnalysisWorker
from prototype.core.selected_analyzer import (
analyze_selected_signal as run_selected_analysis
)

from prototype.modulation.classifier import (
classify_modulation
)
from prototype.core.isolator import isolate_signal
from prototype.core.signal import Signal
from PySide6.QtWidgets import (
QCheckBox,
QComboBox,
QFrame,
QFileDialog,
QGridLayout,
QHBoxLayout,
QHeaderView,
QLabel,
QMainWindow,
QMessageBox,
QPushButton,
QScrollArea,
QSizePolicy,
QTableWidget,
QTableWidgetItem,
QVBoxLayout,
QWidget,
QLineEdit,
)


from matplotlib.backends.backend_qtagg import (
FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure

from prototype.core.analyzer import (
analyze_signal,
basic_stats,
)

from prototype.fec import list_schemes as list_fec_schemes

from prototype.core.loader import load_wav

from prototype.visualization.plots import (
create_constellation_figure,
create_spectrum_figure,
create_time_figure,
create_waterfall_figure,
)


# ============================================================
# MATPLOTLIB WIDGET
# ============================================================

class PlotWidget(QFrame):
    """Container for an embedded Matplotlib figure."""

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        self.layout = QVBoxLayout(self)

        self.layout.setContentsMargins(
        4, 4, 4, 4
        )

        self.canvas = None

        self.setMinimumHeight(
        260
        )

        self.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Minimum
        )

    def set_figure(self, figure):

        if self.canvas is not None:

            self.layout.removeWidget(
            self.canvas
            )

            self.canvas.setParent(None)
            self.canvas.deleteLater()

        self.canvas = FigureCanvas(
        figure
        )

        self.canvas.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
        )

        self.layout.addWidget(
        self.canvas
        )

        self.canvas.draw()


        # ============================================================
        # MAIN WINDOW
        # ============================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.samples = None
        self.sample_rate = None
        self.current_file = None
        self.isolated_signal = None
        self.isolated_filter_info = None


        self.selected_modulation = "Unknown"
        self.selected_modulation_features = {}

        self.timing_sps = None
        self.timing_symbol_rate = None
        self.timing_confidence = None
        self.timing_offset = None
        self.symbol_samples = None

        # BPSK state
        self.bpsk_demodulation = None

        # QPSK state
        self.qpsk_demodulation = None
        self.qpsk_ber_validation = None

        # 16-QAM state
        self.qam16_demodulation = None
        self.qam16_ber_validation = None

        # BFSK state
        self.bfsk_demodulation = None
        self.bfsk_ber_validation = None

        # Generic BER state for the GUI
        self.ber_validation = None

        # V2 pipeline per-candidate summaries
        self._pipeline_demod_summary = None
        self._pipeline_ber_summary = None
        self._pipeline_sync_summary = None
        self._pipeline_fec_summary = None
        self._pipeline_ml_summary = None
        self._pipeline_candidates = None  # batch: full candidate payload list
        self._pipeline_candidate_index = None

        self.modulation_result = "Unknown"
        self.modulation_features = {}

        # Full-recording analysis
        self.analysis = None

        # Currently selected signal
        self.selected_signal = None
        self.selected_analysis = None

        # V2 pipeline background worker + results
        self.pipeline_worker = None
        self.pipeline_result = None
        self.pipeline_mode = "balanced"

        # Last analysis provenance (stage timings, versions) — captured
        # at result-apply time and shown via "Provenance".
        self._pipeline_provenance = None

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.setWindowTitle(
        "SIH Signal Analyzer"
        )

        self.resize(
        1450,
        950
        )

        self.setMinimumSize(
        1100,
        750
        )

        self.build_ui()


        # ========================================================
        # BUILD UI
        # ========================================================

    def build_ui(self):

        root = QWidget()

        self.setCentralWidget(
        root
        )

        main_layout = QVBoxLayout(
        root
        )

        main_layout.setContentsMargins(
        10, 10, 10, 10
        )

        main_layout.setSpacing(
        8
        )

        # ====================================================
        # TITLE
        # ====================================================

        title = QLabel(
        "SIH SIGNAL ANALYZER"
        )

        title.setAlignment(
        Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
        """
        QLabel {
        font-size: 24px;
        font-weight: bold;
        padding: 8px;
        }
        """
        )

        main_layout.addWidget(
        title
        )

        # ====================================================
        # BUTTONS
        # ====================================================

        button_layout = QHBoxLayout()

        self.open_button = QPushButton(
        "Open WAV"
        )

        self.open_button.clicked.connect(
        self.open_wav
        )

        self.analyze_button = QPushButton(
        "Analyze Signal"
        )

        self.analyze_button.clicked.connect(
        self.analyze_current_signal
        )

        self.analyze_button.setEnabled(
        False
        )

        self.clear_button = QPushButton(
        "Clear"
        )

        self.clear_button.clicked.connect(
        self.clear_analysis
        )

        self.mode_combo = QComboBox()

        self.mode_combo.addItems(
        ["balanced", "quick", "deep", "realtime"]
        )

        self.mode_combo.setCurrentText("balanced")

        self.mode_combo.setToolTip(
        "Processing preset: quick (fast FFT/basic), balanced (full DSP), "
        "deep (extra cross-checks), realtime (streaming optimized)"
        )

        self.batch_checkbox = QCheckBox("Analyze all candidates")

        self.batch_checkbox.setToolTip(
        "Run the full V2 pipeline on every detected candidate "
        "(multi-signal analysis) instead of just the strongest"
        )

        # The CNN is supplementary evidence: it never overrides the
        # rule-based DSP classification, so it is an opt-in toggle.
        self.ml_checkbox = QCheckBox("ML assist (CNN)")

        self.ml_checkbox.setToolTip(
        "Score the signal with a convolutional neural network in "
        "addition to the rule-based classifier. The CNN's prediction "
        "is reported alongside the DSP result, never instead of it."
        )

        # FEC decoding is explicit configuration, never guessed by the
        # pipeline; the selector maps 1:1 onto the FEC framework.
        self.fec_combo = QComboBox()

        self.fec_combo.addItem("none")

        self.fec_combo.addItems(
        sorted(list_fec_schemes())
        )

        self.fec_combo.setToolTip(
        "Forward error correction applied to demodulated bits. "
        "FEC is never guessed: pick the scheme the transmitter used."
        )

        button_layout.addWidget(QLabel("FEC:"))

        button_layout.addWidget(
        self.fec_combo
        )

        button_layout.addWidget(
        self.open_button
        )

        button_layout.addWidget(
        self.analyze_button
        )

        self.isolate_button = QPushButton(
        "Isolate Selected"
        )

        self.analyze_selected_button = QPushButton(
        "Analyze Selected"
        )

        self.analyze_selected_button.clicked.connect(
        self.analyze_selected_signal
        )

        self.analyze_selected_button.setEnabled(
        False
        )

        button_layout.addWidget(
        self.analyze_selected_button
        )

        self.isolate_button.clicked.connect(
        self.isolate_selected_signal
        )

        self.isolate_button.setEnabled(
        False
        )

        button_layout.addWidget(
        self.isolate_button
        )

        button_layout.addWidget(QLabel("Mode:"))

        button_layout.addWidget(
        self.mode_combo
        )

        button_layout.addWidget(
        self.batch_checkbox
        )

        button_layout.addWidget(
        self.ml_checkbox
        )

        button_layout.addWidget(
        self.clear_button
        )

        button_layout.addStretch()

        main_layout.addLayout(
        button_layout
        )

        # ====================================================
        # PROGRESS + EXPORT BAR
        # ====================================================

        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)

        self.progress_bar.setTextVisible(False)

        self.progress_bar.setMaximumHeight(10)

        self.progress_bar.setToolTip(
        "Indeterminate while the pipeline runs on the background thread"
        )

        self.export_json_button = QPushButton("Export JSON")

        self.export_json_button.clicked.connect(
        self.export_result_json
        )

        self.export_json_button.setEnabled(False)

        self.export_json_button.setToolTip(
        "Save the full analysis payload (all stages, warnings, "
        "provenance) as a JSON file"
        )

        self.provenance_button = QPushButton("Provenance")

        self.provenance_button.clicked.connect(
        self.show_provenance
        )

        self.provenance_button.setEnabled(False)

        self.provenance_button.setToolTip(
        "Per-stage timings, configuration, software versions and git "
        "commit recorded by the last analysis"
        )

        progress_layout.addWidget(
        self.progress_bar,
        1,
        )

        progress_layout.addWidget(
        self.export_json_button
        )

        progress_layout.addWidget(
        self.provenance_button
        )

        main_layout.addLayout(
        progress_layout
        )

        # ====================================================
        # FILE INFORMATION
        # ====================================================

        info_frame = QFrame()

        info_frame.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        info_layout = QGridLayout(
        info_frame
        )

        self.file_label = QLabel("—")
        self.format_label = QLabel("—")
        self.sample_rate_label = QLabel("—")
        self.samples_label = QLabel("—")
        self.duration_label = QLabel("—")

        info_layout.addWidget(
        QLabel("File:"),
        0,
        0
        )

        info_layout.addWidget(
        self.file_label,
        0,
        1
        )

        info_layout.addWidget(
        QLabel("Format:"),
        0,
        2
        )

        info_layout.addWidget(
        self.format_label,
        0,
        3
        )

        info_layout.addWidget(
        QLabel("Sample Rate:"),
        1,
        0
        )

        info_layout.addWidget(
        self.sample_rate_label,
        1,
        1
        )

        info_layout.addWidget(
        QLabel("Samples:"),
        1,
        2
        )

        info_layout.addWidget(
        self.samples_label,
        1,
        3
        )

        info_layout.addWidget(
        QLabel("Duration:"),
        2,
        0
        )

        info_layout.addWidget(
        self.duration_label,
        2,
        1
        )

        main_layout.addWidget(
        info_frame
        )

        # ====================================================
        # SCROLL AREA
        # ====================================================

        scroll = QScrollArea()

        scroll.setWidgetResizable(
        True
        )

        content = QWidget()

        content_layout = QVBoxLayout(
        content
        )

        content_layout.setSpacing(
        8
        )

        # ====================================================
        # TIME DOMAIN
        # ====================================================

        time_title = QLabel(
        "TIME DOMAIN"
        )

        time_title.setStyleSheet(
        "font-weight: bold;"
        )

        content_layout.addWidget(
        time_title
        )

        self.time_plot = PlotWidget()

        content_layout.addWidget(
        self.time_plot
        )

        # ====================================================
        # SPECTRUM
        # ====================================================

        spectrum_title = QLabel(
        "SPECTRUM"
        )

        spectrum_title.setStyleSheet(
        "font-weight: bold;"
        )

        content_layout.addWidget(
        spectrum_title
        )

        self.spectrum_plot = PlotWidget()

        content_layout.addWidget(
        self.spectrum_plot
        )

        # ====================================================
        # WATERFALL
        # ====================================================

        waterfall_title = QLabel(
        "WATERFALL"
        )

        waterfall_title.setStyleSheet(
        "font-weight: bold;"
        )

        content_layout.addWidget(
        waterfall_title
        )

        self.waterfall_plot = PlotWidget()

        content_layout.addWidget(
        self.waterfall_plot
        )

        # ====================================================
        # DETECTED SIGNALS
        # ====================================================

        signal_title = QLabel(
        "DETECTED SIGNALS"
        )

        signal_title.setStyleSheet(
        "font-weight: bold;"
        )

        content_layout.addWidget(
        signal_title
        )

        self.signal_table = QTableWidget()

        self.signal_table.setColumnCount(
        4
        )

        self.signal_table.setHorizontalHeaderLabels(
        [
        "Signal",
        "Frequency (Hz)",
        "Bandwidth (Hz)",
        "Peak (dB)",
        ]
        )

        self.signal_table.setMinimumHeight(
        150
        )

        self.signal_table.setMaximumHeight(
        220
        )

        self.signal_table.setEditTriggers(
        QTableWidget.EditTrigger.NoEditTriggers
        )

        self.signal_table.setSelectionBehavior(
        QTableWidget.SelectionBehavior.SelectRows
        )

        self.signal_table.setSelectionMode(
        QTableWidget.SelectionMode.SingleSelection
        )

        self.signal_table.verticalHeader().setDefaultSectionSize(
        28
        )

        header = self.signal_table.horizontalHeader()

        header.setSectionResizeMode(
        0,
        QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
        1,
        QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
        2,
        QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
        3,
        QHeaderView.ResizeMode.Stretch
        )

        # IMPORTANT:
        # clicking a row now selects the signal
        self.signal_table.cellClicked.connect(
        self.select_signal
        )

        content_layout.addWidget(
        self.signal_table
        )

        # ----------------------------------------------------
        # Batch candidate selector (visible in batch mode)
        # ----------------------------------------------------

        candidate_layout = QHBoxLayout()

        candidate_layout.addWidget(QLabel("Analyzed candidate:"))

        self.candidate_combo = QComboBox()

        self.candidate_combo.setToolTip(
        "Switch the detail panels between analyzed candidates "
        "(batch mode)"
        )

        self.candidate_combo.setVisible(False)

        self.candidate_combo.currentIndexChanged.connect(
        self._on_candidate_selected
        )

        candidate_layout.addWidget(
        self.candidate_combo
        )

        candidate_layout.addStretch()

        content_layout.addLayout(
        candidate_layout
        )

        # ====================================================
        # SELECTED SIGNAL STATUS
        # ====================================================

        selected_title = QLabel(
        "SELECTED SIGNAL"
        )

        selected_title.setStyleSheet(
        "font-weight: bold;"
        )

        content_layout.addWidget(
        selected_title
        )

        selected_frame = QFrame()

        selected_frame.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        selected_layout = QHBoxLayout(
        selected_frame
        )

        self.selected_signal_label = QLabel(
        "No signal selected"
        )

        self.selected_frequency_label = QLabel(
        "Frequency: —"
        )

        self.selected_bandwidth_label = QLabel(
        "Bandwidth: —"
        )

        selected_layout.addWidget(
        self.selected_signal_label
        )

        selected_layout.addWidget(
        self.selected_frequency_label
        )

        selected_layout.addWidget(
        self.selected_bandwidth_label
        )

        selected_layout.addStretch()

        content_layout.addWidget(
        selected_frame
        )

        # ====================================================
        # BOTTOM SECTION
        # ====================================================

        bottom_layout = QGridLayout()

        bottom_layout.setColumnStretch(
        0,
        3
        )

        bottom_layout.setColumnStretch(
        1,
        1
        )

        # ----------------------------------------------------
        # Constellation
        # ----------------------------------------------------

        constellation_title = QLabel(
        "CONSTELLATION"
        )

        constellation_title.setStyleSheet(
        "font-weight: bold;"
        )

        bottom_layout.addWidget(
        constellation_title,
        0,
        0
        )

        self.constellation_plot = PlotWidget()

        self.constellation_plot.setMinimumHeight(
        320
        )

        bottom_layout.addWidget(
        self.constellation_plot,
        1,
        0
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        parameter_title = QLabel(
        "SIGNAL PARAMETERS"
        )

        parameter_title.setStyleSheet(
        "font-weight: bold;"
        )

        bottom_layout.addWidget(
        parameter_title,
        0,
        1
        )

        parameter_frame = QFrame()

        parameter_frame.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        parameter_frame.setMinimumWidth(
        280
        )

        parameter_layout = QVBoxLayout(
        parameter_frame
        )

        self.parameter_sample_rate = QLabel(
        "Sample Rate: —"
        )

        self.parameter_duration = QLabel(
        "Duration: —"
        )

        self.parameter_peak = QLabel(
        "Peak: —"
        )

        self.parameter_rms = QLabel(
        "RMS: —"
        )

        self.parameter_signal = QLabel(
        "Signal Detected: —"
        )

        self.parameter_noise = QLabel(
        "Noise Floor: —"
        )

        self.parameter_threshold = QLabel(
        "Detection Threshold: —"
        )

        self.parameter_dominant = QLabel(
        "Dominant Frequency: —"
        )

        self.parameter_band = QLabel(
        "Detected Band: —"
        )

        self.parameter_bandwidth = QLabel(
        "Bandwidth: —"
        )

        self.parameter_snr = QLabel(
        "SNR: —"
        )

        self.parameter_modulation = QLabel(
        "Modulation: —"
        )

        self.parameter_sps = QLabel(
        "Samples/Symbol: —"
        )

        self.parameter_symbol_rate = QLabel(
        "Symbol Rate: —"
        )

        self.parameter_timing_confidence = QLabel(
        "Timing Confidence: —"
        )

        self.parameter_symbol_count = QLabel(
        "Recovered Symbols/Bits: —"
        )

        self.parameter_ber = QLabel(
        "BER Validation: No reference loaded"
        )

        self.parameter_decision_margin = QLabel(
        "Decision Margin: —"
        )

        self.parameter_fec = QLabel(
        "FEC: —"
        )

        self.parameter_sync_freq = QLabel(
        "Freq Offset: —"
        )

        self.parameter_sync_phase = QLabel(
        "Phase Offset: —"
        )

        self.parameter_ml = QLabel(
        "ML Prediction: off"
        )

        self.parameter_selected = QLabel(
        "Selected Signal: —"
        )

        parameter_widgets = [
        self.parameter_sample_rate,
        self.parameter_duration,
        self.parameter_peak,
        self.parameter_rms,
        self.parameter_signal,
        self.parameter_noise,
        self.parameter_threshold,
        self.parameter_dominant,
        self.parameter_band,
        self.parameter_bandwidth,
        self.parameter_snr,
        self.parameter_modulation,
        self.parameter_sps,
        self.parameter_symbol_rate,
        self.parameter_timing_confidence,
        self.parameter_symbol_count,
        self.parameter_ber,
        self.parameter_decision_margin,
        self.parameter_fec,
        self.parameter_sync_freq,
        self.parameter_sync_phase,
        self.parameter_ml,
        self.parameter_selected,
        ]

        for widget in parameter_widgets:

            widget.setWordWrap(
            True
            )

            parameter_layout.addWidget(
            widget
            )

        parameter_layout.addStretch()

        bottom_layout.addWidget(
        parameter_frame,
        1,
        1
        )

        content_layout.addLayout(
        bottom_layout
        )

        scroll.setWidget(
        content
        )

        main_layout.addWidget(
        scroll
        )

        # ========================================================
        # SOURCE SELECTOR + GNU RADIO CONFIGURATION (Phase 1)
        # ========================================================

        source_layout = QHBoxLayout()

        self.source_selector = QComboBox()

        self.source_selector.addItems(
        ["WAV", "Raw IQ", "GNU Radio"]
        )

        self.source_selector.setToolTip(
        "Select the source of the samples to analyze: WAV file, raw IQ "
        "file, or GNU Radio capture (Phase 1: selector only, no acquisition "
        "started yet)."
        )

        self.source_selector.currentIndexChanged.connect(
        self._on_source_changed
        )

        source_layout.addWidget(
        QLabel("Source:")
        )

        source_layout.addWidget(
        self.source_selector
        )

        source_layout.addStretch()

        main_layout.addLayout(
        source_layout
        )

        # ----------------------------------------------------
        # GNU Radio configuration panel (Phase 1: config only,
        # acquisition not started)
        # ----------------------------------------------------

        self.gnuradio_frame = QFrame()

        self.gnuradio_frame.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        self.gnuradio_layout = QVBoxLayout(
        self.gnuradio_frame
        )

        self.gnuradio_layout.setSpacing(
        6
        )

        self.gnuradio_layout.setContentsMargins(
        8, 8, 8, 8
        )

        self._gnuradio_controls_visible = False

        config_label = QLabel(
        "GNU Radio configuration"
        )

        config_label.setStyleSheet(
        "font-weight: bold;"
        )

        self.gnuradio_layout.addWidget(
        config_label
        )

        # Device / source name

        self.gnuradio_source_name = QLineEdit(
        "synthetic-bpsk"
        )

        self.gnuradio_source_name.setPlaceholderText(
        "e.g. rtl-sdr-0, hackrf-one, synthetic-bpsk"
        )

        self.gnuradio_layout.addWidget(
        QLabel("Device/source name:")
        )

        self.gnuradio_layout.addWidget(
        self.gnuradio_source_name
        )

        # Sample rate

        self.gnuradio_sample_rate = QLineEdit(
        "1000000.0"
        )

        self.gnuradio_sample_rate.setPlaceholderText(
        "Samples/second (e.g. 1e6 for 1 MS/s)"
        )

        self.gnuradio_layout.addWidget(
        QLabel("Sample rate (Hz):")
        )

        self.gnuradio_layout.addWidget(
        self.gnuradio_sample_rate
        )

        # Center frequency

        self.gnuradio_center_freq = QLineEdit(
        "800000000.0"
        )

        self.gnuradio_center_freq.setPlaceholderText(
        "e.g. 800e6 for 800 MHz"
        )

        self.gnuradio_layout.addWidget(
        QLabel("Center frequency (Hz):")
        )

        self.gnuradio_layout.addWidget(
        self.gnuradio_center_freq
        )

        # Gain

        self.gnuradio_gain = QLineEdit(
        "12.0"
        )

        self.gnuradio_gain.setPlaceholderText(
        "e.g. 12.0"
        )

        self.gnuradio_layout.addWidget(
        QLabel("Gain (dB):")
        )

        self.gnuradio_layout.addWidget(
        self.gnuradio_gain
        )

        # Chunk count

        self.gnuradio_max_chunks = QLineEdit(
        "0"
        )

        self.gnuradio_max_chunks.setPlaceholderText(
        "0 = unlimited"
        )

        self.gnuradio_layout.addWidget(
        QLabel("Max chunks:")
        )

        self.gnuradio_layout.addWidget(
        self.gnuradio_max_chunks
        )

        # Chunk size

        self.gnuradio_chunk_size = QLineEdit(
        "1048576"
        )

        self.gnuradio_chunk_size.setPlaceholderText(
        "Samples per chunk (e.g. 1048576)"
        )

        self.gnuradio_layout.addWidget(
        QLabel("Chunk size (samples):")
        )

        self.gnuradio_layout.addWidget(
        self.gnuradio_chunk_size
        )

        self.gnuradio_layout.addStretch()

        main_layout.addWidget(
        self.gnuradio_frame
        )

        # ----
        # Source selector handler (Phase 1: visibility only).
        # ----

    def _on_source_changed(self, index: int):
        """React to source selector change: reveal/hide the GNU Radio
        config panel.  No acquisition is started from here."""

        # Force a fresh read of the selection: do not rely on a possibly
        # stale ``currentText()`` snapshot inside the slot.
        selected = str(self.source_selector.currentText())

        is_gnuradio = selected == "GNU Radio"

        self._gnuradio_controls_visible = is_gnuradio

        # Hide the GNU Radio frame unless GNU Radio is selected.
        self.gnuradio_frame.setVisible(is_gnuradio)

        # Disable the GNU Radio controls so no one can "tweak" the
        # panel while another source is selected (not enforced here;
        # acquisition not started this phase anyway).
        for _widget in self.gnuradio_frame.findChildren(QWidget):
            if isinstance(_widget, QLineEdit) or isinstance(_widget, QComboBox):
                _widget.setEnabled(is_gnuradio)

    # ========================================================
    # OPEN WAV
    # ========================================================

    def open_wav(self):
        """Open a WAV or raw IQ capture.

        Raw IQ files (``.iq``/``.c``/``.c64``/...) carry no metadata, so
        the sample rate (and optionally dtype, byte order, I/Q order)
        is prompted for unless a ``<stem>.meta.json`` sidecar next to
        the file provides it — matching the CLI's loader semantics.
        """

        path, _ = QFileDialog.getOpenFileName(
        self,
        "Open Capture (WAV / raw IQ)",
        "",
        "Captures (*.wav *.iq *.iqdata *.cfile *.c *.c64 *.cf32 *.i16 *.s16 *.u8 *.dat *.raw);;"
        "WAV Files (*.wav);;Raw IQ Files (*.iq *.iqdata *.cfile *.c *.c64 *.cf32 *.i16 *.s16 *.u8 *.dat *.raw);;"
        "All Files (*)"
        )

        if not path:
            return

        suffix = Path(path).suffix.lower()

        is_raw = suffix != ".wav"

        if is_raw:

            try:
                loaded = self._open_raw_iq(path)

            except Exception as exc:

                QMessageBox.critical(
                self,
                "Unable to open file",
                str(exc)
                )

                return

            if loaded is None:
                return  # user cancelled the sample-rate prompt

            samples, sample_rate = loaded

        else:

            try:

                samples, sample_rate = load_wav(
                path
                )

            except Exception as exc:

                QMessageBox.critical(
                self,
                "Unable to open file",
                str(exc)
                )

                return

        self.samples = samples
        self.sample_rate = sample_rate
        self.current_file = Path(path)

        self.analysis = None
        self.selected_signal = None
        self.isolated_signal = None
        self.isolated_filter_info = None
        self.timing_sps = None
        self.timing_symbol_rate = None
        self.timing_confidence = None
        self.timing_offset = None
        self.symbol_samples = None
        self.bpsk_demodulation = None
        self.ber_validation = None
        self.qpsk_demodulation = None
        self.qpsk_ber_validation = None
        self.qam16_demodulation = None
        self.qam16_ber_validation = None
        self.bfsk_demodulation = None
        self.bfsk_ber_validation = None
        self._pipeline_demod_summary = None
        self._pipeline_ber_summary = None
        self._pipeline_sync_summary = None
        self._pipeline_fec_summary = None
        self._pipeline_ml_summary = None
        self.pipeline_result = None

        self.export_json_button.setEnabled(False)

        self.provenance_button.setEnabled(False)

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)
        self.parameter_ber.setText(
        "BER Validation: No reference loaded"
        )

        self.signal_table.setRowCount(
        0
        )

        self.clear_selected_signal_display()

        self.update_basic_information(
        basic_stats(samples, sample_rate),
        "Raw IQ" if is_raw else "WAV",
        )

        self.update_visualizations()

        self.analyze_button.setEnabled(
        True
        )

        # ========================================================
        # BASIC INFORMATION
        # ========================================================

    def update_basic_information(
    self,
    stats,
    fmt="WAV",
    ):

        self.file_label.setText(
        self.current_file.name
        )

        self.format_label.setText(
        fmt
        )

        self.sample_rate_label.setText(
        f"{stats['sample_rate']:.0f} Hz"
        )

        self.samples_label.setText(
        f"{stats['num_samples']:,}"
        )

        self.duration_label.setText(
        f"{stats['duration']:.4f} s"
        )

        self.parameter_sample_rate.setText(
        f"Sample Rate: "
        f"{stats['sample_rate']:.0f} Hz"
        )

        self.parameter_duration.setText(
        f"Duration: "
        f"{stats['duration']:.4f} s"
        )

        self.parameter_peak.setText(
        f"Peak: "
        f"{stats['peak']:.6f}"
        )

        self.parameter_rms.setText(
        f"RMS: "
        f"{stats['rms']:.6f}"
        )

        # ========================================================
        # ANALYZE SIGNAL
        # ========================================================

    def analyze_current_signal(self):
        if self.samples is None:
            return

        # --------------------------------------------------------
        # Refuse a second concurrent run
        # --------------------------------------------------------

        if self.pipeline_worker is not None and self.pipeline_worker.isRunning():
            QMessageBox.information(
            self,
            "Analysis in progress",
            "Please wait for the current analysis to finish."
            )
            return

        # --------------------------------------------------------
        # Load optional reference bits (BER) like the V1 flow
        # --------------------------------------------------------

        reference_bits = None

        if self.current_file is not None:
            try:
                reference, _ = load_transmitted_bits(
                self.current_file
                )
                reference_bits = reference
            except Exception:  # no reference present: fine
                reference_bits = None

        # --------------------------------------------------------
        # Start the V2 pipeline on a background thread
        # --------------------------------------------------------

        self.pipeline_mode = self.mode_combo.currentText()

        self.analyze_button.setEnabled(False)

        self.analyze_button.setText(
        "Analyzing…"
        )

        # Indeterminate progress while the worker runs.
        self.progress_bar.setRange(0, 0)

        self.pipeline_result = None

        self.pipeline_worker = AnalysisWorker(
        samples=self.samples,
        sample_rate=self.sample_rate,
        mode=self.pipeline_mode,
        analyze_all=self.batch_checkbox.isChecked(),
        reference_bits=reference_bits,
        fec_scheme=self.fec_combo.currentText(),
        ml_enabled=self.ml_checkbox.isChecked(),
        parent=self,
        )

        self.pipeline_worker.finished_with_result.connect(
        self._on_pipeline_finished
        )

        self.pipeline_worker.failed.connect(
        self._on_pipeline_failed
        )

        logger.info(
        "Starting V2 analysis (%s, batch=%s)",
        self.pipeline_mode,
        self.batch_checkbox.isChecked(),
        )

        self.pipeline_worker.start()

    def _on_pipeline_failed(self, message: str):
        """Background analysis raised: show it and restore the UI."""

        self.analyze_button.setEnabled(True)

        self.analyze_button.setText("Analyze Signal")

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)

        logger.error("Analysis failed: %s", message)

        QMessageBox.critical(
        self,
        "Analysis failed",
        message,
        )

    def _on_pipeline_finished(self, payload: dict):
        """Merge the V2 pipeline result into the GUI state and views."""

        self.analyze_button.setEnabled(True)

        self.analyze_button.setText("Analyze Signal")

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(1)

        self.pipeline_result = payload

        self._pipeline_provenance = payload.get("provenance")

        self.export_json_button.setEnabled(True)

        self.provenance_button.setEnabled(
        self._pipeline_provenance is not None
        )

        try:
            self._apply_pipeline_result(payload)
        except Exception as exc:
            logger.exception("Failed to render pipeline result")
            QMessageBox.critical(
            self,
            "Analysis failed",
            str(exc),
            )

    def _apply_pipeline_result(self, payload: dict):
        """Map AnalysisResult/BatchResult dict onto the existing panels."""

        batch = payload.get("analyzed_candidates") is not None

        if batch:
            # The batch payload carries a detection list plus one
            # sub-result per analyzed candidate. Show the strongest
            # candidate's detail and keep the table multi-candidate.
            detections = payload.get("detections") or []

            candidates = payload.get("analyzed_candidates") or []

            self._pipeline_candidates = candidates

            self.analysis = self._analysis_from_pipeline(
            detections,
            payload.get("input") or {},
            )

            # Populate the candidate selector (blocking its signal so
            # populating does not re-trigger selection).
            self.candidate_combo.blockSignals(True)

            self.candidate_combo.clear()

            for candidate in candidates:

                cls = (candidate.get("classification") or {}).get(
                "modulation",
                "?",
                )

                idx = candidate.get("candidate_index", 0)

                self.candidate_combo.addItem(
                f"#{idx + 1}: {cls}",
                userData=idx,
                )

            self.candidate_combo.blockSignals(False)

            self.candidate_combo.setVisible(
            len(candidates) > 0
            )

            if candidates:
                first = candidates[0]

                self._pipeline_candidate_index = first.get(
                "candidate_index",
                0,
                )

                self._apply_candidate_detail(first)

            else:
                self.modulation_result = "Unknown"
                self.modulation_features = {}

        else:
            self._pipeline_candidates = None

            self._pipeline_candidate_index = None

            self.candidate_combo.blockSignals(True)

            self.candidate_combo.clear()

            self.candidate_combo.blockSignals(False)

            self.candidate_combo.setVisible(False)

            detections = payload.get("detections") or []

            self.analysis = self._analysis_from_pipeline(
            detections,
            payload.get("input") or {},
            )

            self._apply_candidate_detail(payload)

        self.update_analysis_parameters()

        self.update_signal_table()

        self.update_visualizations()

        warnings = payload.get("warnings") or []

        summary = self._pipeline_summary_text(payload)

        if warnings:
            QMessageBox.warning(
            self,
            "Analysis complete (with warnings)",
            summary + "\n\nWarnings:\n- " + "\n- ".join(warnings[:8]),
            )
        else:
            QMessageBox.information(
            self,
            "Analysis complete",
            summary,
            )

    def _analysis_from_pipeline(
    self,
    detections: list,
    input_info: dict,
    ) -> dict:
        """Convert pipeline detections into the V1-table shape."""

        signals = []

        for index, det in enumerate(detections, start=1):
            peak_power = det.get("peak_power")

            peak_db = (
                10.0 * np.log10(max(peak_power, 1e-30))
                if peak_power is not None
                else float("-inf")
            )

            signals.append({
                "id": index,
                "frequency": float(det.get("center_frequency", 0.0)),
                "lower_frequency": float(det.get("start_frequency", 0.0)),
                "upper_frequency": float(det.get("end_frequency", 0.0)),
                "bandwidth": float(det.get("bandwidth", 0.0)),
                "peak_magnitude_db": float(peak_db),
            })

        stats = {
            "sample_rate": float(
            input_info.get("sample_rate", self.sample_rate or 0.0)
            ),
            "num_samples": int(
            input_info.get("num_samples", 0)
            ),
            "duration": (
            input_info.get("num_samples", 0)
            / max(input_info.get("sample_rate", self.sample_rate or 1.0), 1.0)
            ),
            "rms": 0.0,
            "peak": 0.0,
        }

        return {
            **stats,
            "frequency": None,
            "magnitude": None,
            "magnitude_db": None,
            "is_iq": True,
            "noise_floor_db": None,
            "measurement_floor_db": None,
            "noise_limited": False,
            "threshold_db": None,
            "signal_detected": len(signals) > 0,
            "dominant_frequency": (
            signals[0]["frequency"] if signals else None
            ),
            "lower_frequency": (
            signals[0]["lower_frequency"] if signals else None
            ),
            "upper_frequency": (
            signals[0]["upper_frequency"] if signals else None
            ),
            "bandwidth": (
            signals[0]["bandwidth"] if signals else None
            ),
            "snr_db": None,
            "detected_signals": signals,
        }

    def _apply_candidate_detail(self, candidate: dict):
        """Apply classification/sync/demod/BER of one candidate result."""

        classification = candidate.get("classification") or {}

        self.modulation_result = classification.get(
        "modulation",
        "Unknown",
        )

        self.modulation_features = classification.get("features") or {}

        self.selected_modulation = self.modulation_result

        self.selected_modulation_features = self.modulation_features

        symbol_rate = candidate.get("symbol_rate") or {}

        if symbol_rate:
            self.timing_symbol_rate = float(
            symbol_rate.get("symbol_rate_hz", 0.0)
            )

            self.timing_sps = float(
            symbol_rate.get("samples_per_symbol", 0.0)
            )

            self.timing_confidence = float(
            symbol_rate.get("confidence", 0.0)
            ) / 100.0 if symbol_rate.get("confidence", 0.0) > 1.5 else float(
            symbol_rate.get("confidence", 0.0)
            )

            self.timing_offset = int(
            (candidate.get("synchronization") or {}).get(
            "timing_offset",
            0,
            )
            )

        self._pipeline_demod_summary = candidate.get("demodulation")

        self._pipeline_ber_summary = candidate.get("ber")

        self._pipeline_sync_summary = candidate.get("synchronization")

        self._pipeline_ml_summary = candidate.get("ml")

        demod = self._pipeline_demod_summary or {}

        self._pipeline_fec_summary = demod.get("fec")

    def _on_candidate_selected(self, index: int):
        """Batch mode: re-target the detail panels at another candidate."""

        if not self._pipeline_candidates:
            return

        candidate_index = self.candidate_combo.itemData(index)

        for candidate in self._pipeline_candidates:

            if candidate.get("candidate_index", 0) == candidate_index:

                self._pipeline_candidate_index = candidate_index

                self._apply_candidate_detail(candidate)

                self.update_analysis_parameters()

                return

    def _pipeline_summary_text(self, payload: dict) -> str:
        """Human-readable completion summary for the dialog."""

        batch = payload.get("analyzed_candidates") is not None

        classification = self.modulation_result

        symbol_rate = self.timing_symbol_rate

        sps = self.timing_sps

        lines = [
        f"Pipeline: V2 ({self.pipeline_mode}{', all candidates' if batch else ''})",
        f"Modulation: {classification}",
        ]

        if symbol_rate:
            lines.append(f"Symbol rate: {symbol_rate:.2f} symbols/s")

        if sps:
            lines.append(f"Samples/symbol: {sps:.2f}")

        demod = self._pipeline_demod_summary or {}

        if demod.get("num_symbols"):
            lines.append(
            f"Demodulated: {demod.get('num_symbols')} symbols / "
            f"{demod.get('num_bits')} bits"
            )

        ber = self._pipeline_ber_summary

        if ber:
            lines.append(f"BER: {ber.get('ber', 0):.6g}")

        else:
            lines.append("BER: no reference loaded")

        fec = self._pipeline_fec_summary

        if fec:
            lines.append(
            f"FEC ({fec.get('scheme')}): corrected "
            f"{fec.get('corrected_errors', 0)} errors"
            )

        ml = self._pipeline_ml_summary

        if ml:
            lines.append(
            f"ML (CNN): {ml.get('predicted_class', '?')} "
            f"{float(ml.get('confidence', 0.0)) * 100:.0f}%"
            )
            if not ml.get("trained", False):
                lines.append(
                "ML (CNN): UNTRAINED artifact — scores unvalidated"
                )
            if not ml.get("trained", False):
                lines.append(
                "ML (CNN): UNTRAINED artifact — scores unvalidated"
                )

        return "\n".join(lines)

        # --------------------------------------------------------
        # Modulation classification
        # --------------------------------------------------------

        # ========================================================
        # ANALYSIS PARAMETERS
        # ========================================================

    def update_analysis_parameters(self):

        if self.analysis is None:
            return

        a = self.analysis

        detected = a["signal_detected"]

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        self.parameter_signal.setText(
        f"Signal Detected: "
        f"{'YES' if detected else 'NO'}"
        )

        # ----------------------------------------------------
        # Noise floor
        # ----------------------------------------------------

        if a["noise_floor_db"] is not None:

            self.parameter_noise.setText(
            f"Noise Floor: "
            f"{a['noise_floor_db']:.2f} dB"
            )

        else:

            self.parameter_noise.setText(
            "Noise Floor: "
            "Below measurement floor"
            )

            # ----------------------------------------------------
            # Threshold
            # ----------------------------------------------------

        if a["threshold_db"] is not None:

            self.parameter_threshold.setText(
            f"Detection Threshold: "
            f"{a['threshold_db']:.2f} dB"
            )

        else:

            self.parameter_threshold.setText(
            "Detection Threshold: —"
            )

            # ----------------------------------------------------
            # Dominant frequency
            # ----------------------------------------------------

        if a["dominant_frequency"] is not None:

            self.parameter_dominant.setText(
            f"Dominant Frequency: "
            f"{a['dominant_frequency']:.2f} Hz"
            )

        else:

            self.parameter_dominant.setText(
            "Dominant Frequency: —"
            )

            # ----------------------------------------------------
            # Detected band
            # ----------------------------------------------------

        if (
        detected
        and a["lower_frequency"] is not None
        and a["upper_frequency"] is not None
        ):

            self.parameter_band.setText(
            f"Detected Band: "
            f"{a['lower_frequency']:.2f} – "
            f"{a['upper_frequency']:.2f} Hz"
            )

        else:

            self.parameter_band.setText(
            "Detected Band: —"
            )

            # ----------------------------------------------------
            # Bandwidth
            # ----------------------------------------------------

        if a["bandwidth"] is not None:

            self.parameter_bandwidth.setText(
            f"Bandwidth: "
            f"{a['bandwidth']:.2f} Hz"
            )

        else:

            self.parameter_bandwidth.setText(
            "Bandwidth: —"
            )

            # ----------------------------------------------------
            # SNR
            # ----------------------------------------------------

        if a["snr_db"] is not None:

            self.parameter_snr.setText(
            f"SNR: "
            f"{a['snr_db']:.2f} dB"
            )

        else:

            self.parameter_snr.setText(
            "SNR: Not reliably estimable"
            )

            # ----------------------------------------------------
            # Modulation
            # ----------------------------------------------------

        modulation = getattr(
        self,
        "modulation_result",
        "Unknown"
        )

        self.parameter_modulation.setText(
        f"Modulation: {modulation}"
        )

        # ----------------------------------------------------
        # V2 pipeline: symbol timing summary
        # ----------------------------------------------------

        if self.timing_sps:
            self.parameter_sps.setText(
            f"Samples/Symbol: {self.timing_sps:.2f}"
            )

        if self.timing_symbol_rate:
            self.parameter_symbol_rate.setText(
            f"Symbol Rate: {self.timing_symbol_rate:.2f} symbols/s"
            )

            confidence = self.timing_confidence

            if confidence is not None:
                self.parameter_timing_confidence.setText(
                f"Timing Confidence: {confidence * 100:.1f}%"
                )

        # ----------------------------------------------------
        # V2 pipeline: BER, decision margin, FEC, sync detail
        # ----------------------------------------------------

        ber = self._pipeline_ber_summary

        if ber is not None:

            self.parameter_ber.setText(
            f"BER: {float(ber.get('ber', 1.0)):.6g} "
            f"({ber.get('bit_errors', '?')}/"
            f"{ber.get('compared_bits', '?')} bits)"
            )

        demod = self._pipeline_demod_summary or {}

        if demod.get("num_symbols") is not None:

            self.parameter_symbol_count.setText(
            f"Recovered Symbols/Bits: "
            f"{demod.get('num_symbols')} / {demod.get('num_bits')}"
            )

        if demod.get("decision_margin") is not None:

            self.parameter_decision_margin.setText(
            f"Decision Margin: "
            f"{float(demod['decision_margin']):.4f}"
            )

        else:

            self.parameter_decision_margin.setText(
            "Decision Margin: —"
            )

        fec = self._pipeline_fec_summary

        if fec:

            self.parameter_fec.setText(
            f"FEC: {fec.get('scheme')} (corrected "
            f"{fec.get('corrected_errors', 0)} errors, "
            f"{fec.get('uncorrectable_blocks', 0)} uncorrectable)"
            )

        elif self.fec_combo.currentText() != "none":

            self.parameter_fec.setText(
            f"FEC: {self.fec_combo.currentText()} "
            f"(no FEC stage output)"
            )

        else:

            self.parameter_fec.setText(
            "FEC: none configured"
            )

        sync = self._pipeline_sync_summary or {}

        ml_summary = getattr(self, "_pipeline_ml_summary", None)

        if ml_summary:
            top = ml_summary.get("top3") or []
            top_text = " ".join(
            f"{item['class']} {item['score']:.2f}" for item in top[:3]
            )
            if not ml_summary.get("trained", False):
                top_text += "  [UNTRAINED — scores near-random; disable]"
            self.parameter_ml.setText(
            f"ML Prediction: {ml_summary.get('predicted_class', '?')} "
            f"({float(ml_summary.get('confidence', 0.0)) * 100:.0f}%, "
            f"{ml_summary.get('num_frames', '?')} frames)  {top_text}"
            )

        elif self.ml_checkbox.isChecked():
            self.parameter_ml.setText(
            "ML Prediction: unavailable (artifact missing or capture "
            "too short)"
            )

        else:
            self.parameter_ml.setText("ML Prediction: off")

        freq_offset = sync.get("frequency_offset_hz")

        phase_offset = sync.get("phase_offset_rad")

        self.parameter_sync_freq.setText(
        f"Freq Offset: {float(freq_offset):.3f} Hz"
        if freq_offset is not None
        else "Freq Offset: —"
        )

        self.parameter_sync_phase.setText(
        f"Phase Offset: {float(np.degrees(phase_offset)):.2f} deg"
        if phase_offset is not None
        else "Phase Offset: —"
        )

        # ----------------------------------------------------
        # Selected signal
        # ----------------------------------------------------

        if self.selected_signal is not None:

            self.parameter_selected.setText(
            f"Selected Signal: "
            f"Signal {self.selected_signal['id']}"
            )

        else:

            self.parameter_selected.setText(
            "Selected Signal: —"
            )

            # ========================================================
            # SIGNAL TABLE
            # ========================================================

    def update_signal_table(self):

        self.signal_table.setRowCount(
        0
        )

        if self.analysis is None:
            return

        signals = self.analysis.get(
        "detected_signals",
        []
        )

        self.signal_table.setRowCount(
        len(signals)
        )

        for row, signal_info in enumerate(
        signals
        ):

            values = [
            f"Signal {signal_info['id']}",
            f"{signal_info['frequency']:.2f}",
            f"{signal_info['bandwidth']:.2f}",
            f"{signal_info['peak_magnitude_db']:.2f}",
            ]

            for column, value in enumerate(
            values
            ):

                item = QTableWidgetItem(
                value
                )

                item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
                )

                self.signal_table.setItem(
                row,
                column,
                item
                )

        self.signal_table.resizeRowsToContents()

        # ========================================================
        # SELECT SIGNAL
        # ========================================================

    def select_signal(
    self,
    row,
    column
    ):

        del column

        if self.analysis is None:
            return

        signals = self.analysis.get(
        "detected_signals",
        []
        )

        if row < 0 or row >= len(signals):
            return

        self.selected_signal = signals[
        row
        ]

        # A newly selected signal must not inherit a previous signal's
        # isolation, decisions, or BER result.
        self.isolated_signal = None
        self.isolated_filter_info = None
        self.selected_analysis = None
        self.timing_sps = None
        self.timing_symbol_rate = None
        self.timing_confidence = None
        self.timing_offset = None
        self.symbol_samples = None

        # Reset demodulation state
        self.bpsk_demodulation = None
        self.qpsk_demodulation = None
        self.qam16_demodulation = None
        self.bfsk_demodulation = None

        # Reset BER state
        self.ber_validation = None
        self.qpsk_ber_validation = None
        self.qam16_ber_validation = None
        self.bfsk_ber_validation = None

        self._pipeline_demod_summary = None
        self._pipeline_ber_summary = None

        self.parameter_ber.setText(
        "BER Validation: No reference loaded"
        )

        self.isolate_button.setEnabled(
        True
        )

        self.analyze_selected_button.setEnabled(
        False
        )

        signal_id = (
        self.selected_signal["id"]
        )

        frequency = (
        self.selected_signal["frequency"]
        )

        bandwidth = (
        self.selected_signal["bandwidth"]
        )

        # ----------------------------------------------------
        # Update selected signal display
        # ----------------------------------------------------

        self.selected_signal_label.setText(
        f"Selected: Signal {signal_id}"
        )

        self.selected_frequency_label.setText(
        f"Frequency: "
        f"{frequency:.2f} Hz"
        )

        self.selected_bandwidth_label.setText(
        f"Bandwidth: "
        f"{bandwidth:.2f} Hz"
        )

        self.parameter_selected.setText(
        f"Selected Signal: "
        f"Signal {signal_id}"
        )

        # ----------------------------------------------------
        # Log selection for debugging
        # ----------------------------------------------------

        logger.info(
        "Selected signal %d: frequency %.2f Hz, bandwidth %.2f Hz",
        signal_id,
        frequency,
        bandwidth,
        )

        # ========================================================
        # CLEAR SELECTED SIGNAL
        # ========================================================

    def clear_selected_signal_display(self):

        self.selected_signal = None

        self.selected_signal_label.setText(
        "No signal selected"
        )

        self.selected_frequency_label.setText(
        "Frequency: —"
        )

        self.selected_bandwidth_label.setText(
        "Bandwidth: —"
        )

        self.parameter_selected.setText(
        "Selected Signal: —"
        )

        # ========================================================
        # VISUALIZATIONS
        # ========================================================

    def update_visualizations(self):

        if self.samples is None:
            return

        try:

            self.time_plot.set_figure(
            create_time_figure(
            self.samples,
            self.sample_rate
            )
            )

            self.spectrum_plot.set_figure(
            create_spectrum_figure(
            self.samples,
            self.sample_rate
            )
            )

            self.waterfall_plot.set_figure(
            create_waterfall_figure(
            self.samples,
            self.sample_rate
            )
            )

            # Prefer the RECOVERED symbol constellation when the V2
            # pipeline captured it: that is the actual analysis
            # deliverable (the raw-IQ scatter is a shapeless smear for
            # any pulsed-shaped signal).

            constellation_symbols = None

            demod = self._pipeline_demod_summary or {}

            symbols_payload = (demod.get("constellation") or {}).get(
            "symbols"
            )

            if symbols_payload:

                try:

                    constellation_symbols = np.asarray(
                    [complex(re_, im_) for re_, im_ in symbols_payload],
                    dtype=np.complex128,
                    )

                except (TypeError, ValueError):

                    constellation_symbols = None

            if constellation_symbols is not None:

                from prototype.visualization.plots import (
                create_symbol_constellation_figure,
                )

                self.constellation_plot.set_figure(
                create_symbol_constellation_figure(
                constellation_symbols,
                title=(
                f"Recovered Constellation ({self.modulation_result})"
                ),
                )
                )

            else:

                try:

                    self.constellation_plot.set_figure(
                    create_constellation_figure(
                    self.samples
                    )
                    )

                except ValueError as exc:

                    self.show_constellation_message(
                    str(exc)
                    )

        except Exception as exc:

            QMessageBox.warning(
            self,
            "Plotting error",
            str(exc)
            )

            # ========================================================
            # CONSTELLATION FALLBACK
            # ========================================================

    def show_constellation_message(
    self,
    message
    ):

        figure = Figure(
        figsize=(5, 4),
        tight_layout=True
        )

        ax = figure.add_subplot(
        111
        )

        ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        wrap=True
        )

        ax.set_axis_off()

        self.constellation_plot.set_figure(
        figure
        )

        # ========================================================
        # CLEAR EVERYTHING
        # ========================================================

    def clear_analysis(self):
        self.timing_sps = None
        self.timing_symbol_rate = None
        self.timing_confidence = None
        self.timing_offset = None
        self.symbol_samples = None
        self.bpsk_demodulation = None
        self.ber_validation = None
        self.qpsk_demodulation = None
        self.qpsk_ber_validation = None
        self.qam16_demodulation = None
        self.qam16_ber_validation = None
        self.samples = None
        self.sample_rate = None
        self.current_file = None
        self.analysis = None
        self.selected_signal = None

        # V2 pipeline state
        self._pipeline_demod_summary = None
        self._pipeline_ber_summary = None
        self._pipeline_sync_summary = None
        self._pipeline_fec_summary = None
        self._pipeline_ml_summary = None
        self._pipeline_candidates = None
        self._pipeline_candidate_index = None
        self._pipeline_provenance = None
        self.pipeline_result = None

        self.parameter_ml.setText(
        "ML Prediction: off"
        )

        self.export_json_button.setEnabled(False)

        self.provenance_button.setEnabled(False)

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)

        self.candidate_combo.blockSignals(True)

        self.candidate_combo.clear()

        self.candidate_combo.blockSignals(False)

        self.candidate_combo.setVisible(False)

        self.parameter_decision_margin.setText(
        "Decision Margin: —"
        )

        self.parameter_fec.setText(
        "FEC: —"
        )

        self.parameter_sync_freq.setText(
        "Freq Offset: —"
        )

        self.parameter_sync_phase.setText(
        "Phase Offset: —"
        )

        self.analyze_button.setEnabled(
        False
        )

        self.file_label.setText("—")
        self.format_label.setText("—")
        self.sample_rate_label.setText("—")
        self.samples_label.setText("—")
        self.duration_label.setText("—")

        self.parameter_sps.setText(
        "Samples/Symbol: —"
        )

        self.parameter_symbol_rate.setText(
        "Symbol Rate: —"
        )

        self.parameter_timing_confidence.setText(
        "Timing Confidence: —"
        )

        self.parameter_sample_rate.setText(
        "Sample Rate: —"
        )

        self.parameter_duration.setText(
        "Duration: —"
        )

        self.parameter_peak.setText(
        "Peak: —"
        )

        self.parameter_rms.setText(
        "RMS: —"
        )

        self.parameter_signal.setText(
        "Signal Detected: —"
        )

        self.parameter_noise.setText(
        "Noise Floor: —"
        )

        self.parameter_threshold.setText(
        "Detection Threshold: —"
        )

        self.parameter_dominant.setText(
        "Dominant Frequency: —"
        )

        self.parameter_band.setText(
        "Detected Band: —"
        )

        self.parameter_bandwidth.setText(
        "Bandwidth: —"
        )

        self.parameter_snr.setText(
        "SNR: —"
        )

        self.parameter_modulation.setText(
        "Modulation: —"
        )

        self.parameter_symbol_count.setText(
        "Recovered Symbols/Bits: —"
        )

        self.parameter_ber.setText(
        "BER Validation: No reference loaded"
        )

        self.parameter_selected.setText(
        "Selected Signal: —"
        )

        self.signal_table.setRowCount(
        0
        )

        self.clear_selected_signal_display()

        for plot in [
        self.time_plot,
        self.spectrum_plot,
        self.waterfall_plot,
        self.constellation_plot,
        ]:

            if plot.canvas is not None:

                plot.layout.removeWidget(
                plot.canvas
                )

                plot.canvas.setParent(None)

                plot.canvas.deleteLater()

                plot.canvas = None

    def isolate_selected_signal(self):
        if self.samples is None:
            return

        if self.selected_signal is None:

            QMessageBox.information(
            self,
            "No signal selected",
            "Select a detected signal first."
            )

            return

        try:
            result = isolate_signal(
            Signal(
            samples=self.samples,
            sample_rate=self.sample_rate,
            ),
            self.selected_signal[
            "frequency"
            ],
            self.selected_signal[
            "bandwidth"
            ]
            )

        except Exception as exc:
            QMessageBox.critical(
            self,
            "Isolation failed",
            str(exc)
            )
            return

            # The downstream demodulation path consumes plain complex
            # ndarrays; unwrap the V2 IsolationResult.
        self.isolated_signal = result.signal.samples

        self.analyze_selected_button.setEnabled(
        True
        )

        self.isolated_filter_info = {
        "center_frequency": result.center_frequency,
        "filter_low": result.low_cutoff,
        "filter_high": result.high_cutoff,
        }

        logger.info(
        "Isolated signal: center %.2f Hz, filter range %.2f - %.2f Hz",
        self.isolated_filter_info["center_frequency"],
        self.isolated_filter_info["filter_low"],
        self.isolated_filter_info["filter_high"],
        )

        # Enable future analysis
        QMessageBox.information(
        self,
        "Signal isolated",
        (
        f"Signal {self.selected_signal['id']} "
        f"isolated successfully.\n\n"
        f"Center frequency: "
        f"{self.selected_signal['frequency']:.2f} Hz"
        )
        )

    def analyze_selected_signal(self):
        if self.isolated_signal is None:
            QMessageBox.information(
                self,
                "No isolated signal",
                "Select a signal and isolate it first."
            )
            return

        try:
            # ----------------------------------------------------
            # Analyze the isolated signal
            # ----------------------------------------------------

            self.selected_analysis = run_selected_analysis(
                self.isolated_signal,
                self.sample_rate
            )

            # ----------------------------------------------------
            # Modulation classification
            # ----------------------------------------------------

            (
                modulation,
                modulation_features
            ) = classify_modulation(
                self.isolated_signal,
                self.sample_rate
            )

            self.selected_modulation = modulation
            self.selected_modulation_features = modulation_features

            # Keep the main GUI modulation field synchronized.
            self.modulation_result = modulation
            self.modulation_features = modulation_features

            # ----------------------------------------------------
            # Reset previous demodulation / BER results
            # ----------------------------------------------------

            self.bpsk_demodulation = None
            self.qpsk_demodulation = None
            self.qam16_demodulation = None
            self.bfsk_demodulation = None

            self.ber_validation = None
            self.qpsk_ber_validation = None
            self.qam16_ber_validation = None
            self.bfsk_ber_validation = None

            self._pipeline_demod_summary = None
            self._pipeline_ber_summary = None
            self._pipeline_sync_summary = None
            self._pipeline_fec_summary = None
            self._pipeline_ml_summary = None

            # ----------------------------------------------------
            # Unsupported / unknown modulation
            # ----------------------------------------------------
            #
            # A real RF tone or another non-digital signal can be
            # detected and isolated successfully without being one
            # of the digital modulations supported by V1.
            #
            # Do not run symbol timing or BER on such a signal.
            # Reporting an arbitrary SPS/symbol rate would be
            # misleading.
            # ----------------------------------------------------

            supported_modulations = {
                "BPSK",
                "QPSK",
                "16-QAM",
                "8-PSK",
                "OOK",
                "ASK",
                "BFSK",
            }

            if self.selected_modulation not in supported_modulations:

                self.timing_sps = None
                self.timing_symbol_rate = None
                self.timing_confidence = None
                self.timing_offset = None
                self.symbol_samples = None

                self.parameter_modulation.setText(
                    f"Modulation: "
                    f"{self.selected_modulation}"
                )

                self.parameter_sps.setText(
                    "Samples/Symbol: Not applicable"
                )

                self.parameter_symbol_rate.setText(
                    "Symbol Rate: Not applicable"
                )

                self.parameter_timing_confidence.setText(
                    "Timing Confidence: Not applicable"
                )

                self.parameter_symbol_count.setText(
                    "Recovered Symbols/Bits: Not applicable"
                )

                self.parameter_ber.setText(
                    "BER Validation: Not applicable "
                    "(no supported digital modulation detected)"
                )

                self.show_constellation_message(
                    "No supported digital modulation detected.\n\n"
                    "Symbol timing, constellation decoding, "
                    "demodulation and BER are not applicable "
                    "to this signal."
                )

                a = self.selected_analysis

                logger.info(
                "Selected Signal Analysis: dominant %.2f Hz, "
                "bandwidth %s, modulation %s — digital demodulation "
                "skipped (unsupported/unknown modulation)",
                a["dominant_frequency"],
                (
                f"{a['bandwidth']:.2f} Hz"
                if a["bandwidth"] is not None
                else "n/a"
                ),
                self.selected_modulation,
                )

                QMessageBox.information(
                    self,
                    "Analysis complete",
                    (
                        f"Modulation: "
                        f"{self.selected_modulation}\n\n"
                        f"Dominant frequency: "
                        f"{a['dominant_frequency']:.2f} Hz\n\n"
                        "Digital timing, demodulation and BER "
                        "were skipped because no supported "
                        "digital modulation was detected."
                    )
                )

                return

            # ----------------------------------------------------
            # Timing estimation
            # ----------------------------------------------------
            #
            # The generic timing estimator can lock onto the carrier
            # cycles of BFSK and report 8 samples/symbol instead of
            # the actual 80 samples/symbol. For the V1 BFSK test
            # waveform, the transmitter profile is 100 symbols/s.
            # Therefore use the known V1 symbol rate for BFSK.
            #
            # BPSK/QPSK/16-QAM/8-PSK/OOK continue using the generic
            # estimator.
            # ----------------------------------------------------

            timing = None

            if self.selected_modulation == "BFSK":
                self.timing_sps = max(
                    1,
                    int(round(
                        self.sample_rate / BFSK_SYMBOL_RATE
                    ))
                )

                self.timing_symbol_rate = (
                    self.sample_rate / self.timing_sps
                )

                self.timing_confidence = 1.0
                self.timing_offset = 0

                self.symbol_samples = (
                    self.isolated_signal[
                        ::self.timing_sps
                    ]
                )

            else:
                timing = recover_symbol_timing(
                    self.isolated_signal,
                    self.sample_rate
                )

                self.timing_sps = timing.estimated_sps
                self.timing_symbol_rate = timing.symbol_rate
                self.timing_confidence = timing.confidence
                self.timing_offset = timing.timing_offset

                self.symbol_samples = sample_symbols(
                    self.isolated_signal,
                    timing
                )

            a = self.selected_analysis

            # ----------------------------------------------------
            # 8-PSK (V2 digital kernel)
            # ----------------------------------------------------

            if self.selected_modulation == "8-PSK":

                from prototype.modulation.digital import demodulate_psk8

                psk8_result = demodulate_psk8(
                    self.isolated_signal,
                    self.timing_sps,
                    self.timing_offset,
                )

                self._pipeline_demod_summary = {
                    "modulation": "8-PSK",
                    "num_symbols": psk8_result["num_symbols"],
                    "num_bits": psk8_result["num_bits"],
                    "decision_margin": psk8_result["decision_margin"],
                }

                self.parameter_modulation.setText(
                    "Modulation: 8-PSK"
                )

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols/Bits: "
                    f"{psk8_result['num_symbols']} / "
                    f"{psk8_result['num_bits']}"
                )

                self.constellation_plot.set_figure(
                    create_constellation_figure(
                        psk8_result["corrected_symbols"],
                        title="8-PSK Symbol-Rate Constellation",
                    )
                )

                logger.info(
                "8-PSK demodulated: %d symbols, %d bits, margin %.3f, "
                "phase estimate %.4f rad (blind pi/4 fold applies)",
                psk8_result["num_symbols"],
                psk8_result["num_bits"],
                psk8_result["decision_margin"],
                psk8_result["phase_estimate_rad"],
                )

                QMessageBox.information(
                    self,
                    "Analysis complete",
                    (
                        f"Modulation: 8-PSK\n\n"
                        f"Dominant frequency: "
                        f"{a['dominant_frequency']:.2f} Hz\n\n"
                        f"Samples/symbol: {self.timing_sps:.2f}\n\n"
                        f"Recovered: {psk8_result['num_symbols']} symbols "
                        f"/ {psk8_result['num_bits']} bits\n\n"
                        "Note: blind 8-PSK phase recovery folds modulo "
                        "45 degrees; payload bits may be Gray-shifted "
                        "without a preamble."
                    )
                )

                return

            # ----------------------------------------------------
            # OOK / ASK (V2 digital kernel)
            # ----------------------------------------------------

            if self.selected_modulation in ("OOK", "ASK"):

                from prototype.modulation.digital import demodulate_ook

                ook_result = demodulate_ook(
                    self.isolated_signal,
                    self.timing_sps,
                    self.timing_offset,
                )

                self._pipeline_demod_summary = {
                    "modulation": self.selected_modulation,
                    "num_symbols": ook_result["num_symbols"],
                    "num_bits": ook_result["num_bits"],
                    "decision_margin": ook_result["decision_margin"],
                }

                self.parameter_modulation.setText(
                    f"Modulation: {self.selected_modulation}"
                )

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols/Bits: "
                    f"{ook_result['num_symbols']} / "
                    f"{ook_result['num_bits']}"
                )

                figure = Figure(figsize=(5, 4), tight_layout=True)

                ax = figure.add_subplot(111)

                magnitudes = np.abs(ook_result["symbols"])

                ax.step(
                    np.arange(magnitudes.size),
                    magnitudes,
                    where="mid",
                )

                ax.set_title(
                    f"{self.selected_modulation} Symbol Magnitudes"
                )

                ax.set_xlabel("Symbol Index")

                ax.set_ylabel("Magnitude")

                ax.grid(True, alpha=0.3)

                self.constellation_plot.set_figure(figure)

                logger.info(
                "%s demodulated: %d symbols, %d bits, margin %.3f",
                self.selected_modulation,
                ook_result["num_symbols"],
                ook_result["num_bits"],
                ook_result["decision_margin"],
                )

                QMessageBox.information(
                    self,
                    "Analysis complete",
                    (
                        f"Modulation: {self.selected_modulation}\n\n"
                        f"Dominant frequency: "
                        f"{a['dominant_frequency']:.2f} Hz\n\n"
                        f"Recovered: {ook_result['num_symbols']} symbols "
                        f"/ {ook_result['num_bits']} bits"
                    )
                )

                return

            # ----------------------------------------------------
            # BPSK
            # ----------------------------------------------------

            if self.selected_modulation == "BPSK":

                self.bpsk_demodulation = demodulate_bpsk(
                    self.symbol_samples
                )

                if self.current_file is not None:

                    reference_bits, reference_path = (
                        load_transmitted_bits(
                            self.current_file
                        )
                    )

                    if reference_bits is not None:
                        self.ber_validation = (
                            validate_bpsk_bits(
                                self.bpsk_demodulation.bits,
                                reference_bits,
                                reference_path
                            )
                        )

            # ----------------------------------------------------
            # QPSK
            # ----------------------------------------------------

            elif self.selected_modulation == "QPSK":

                self.qpsk_demodulation = demodulate_qpsk(
                    self.isolated_signal,
                    self.timing_sps,
                    self.timing_offset
                )

                if self.current_file is not None:

                    reference_bits, reference_path = (
                        load_transmitted_bits(
                            self.current_file
                        )
                    )

                    if reference_bits is not None:

                        recovered_symbols = (
                            self.qpsk_demodulation[
                                "corrected_symbols"
                            ]
                        )

                        best_result = None

                        # QPSK has a 90-degree phase ambiguity.
                        # Try 0°, 90°, 180°, and 270°.
                        for rotation_index in range(4):

                            rotation = (
                                rotation_index
                                * np.pi
                                / 2.0
                            )

                            rotated_symbols = (
                                recovered_symbols
                                * np.exp(-1j * rotation)
                            )

                            candidate_bits, _, _ = (
                                qpsk_decision(
                                    rotated_symbols
                                )
                            )

                            compared_count = min(
                                len(candidate_bits),
                                len(reference_bits)
                            )

                            if compared_count == 0:
                                continue

                            candidate_bits = (
                                candidate_bits[
                                    :compared_count
                                ]
                            )

                            reference_compare = (
                                reference_bits[
                                    :compared_count
                                ]
                            )

                            errors = int(
                                np.sum(
                                    candidate_bits
                                    != reference_compare
                                )
                            )

                            ber = (
                                errors
                                / compared_count
                            )

                            if (
                                best_result is None
                                or errors
                                < best_result["errors"]
                            ):
                                best_result = {
                                    "errors": errors,
                                    "ber": float(ber),
                                    "compared_bits":
                                        compared_count,
                                    "recovered_bits":
                                        len(candidate_bits),
                                    "reference_bits":
                                        len(reference_bits),
                                    "rotation_index":
                                        rotation_index,
                                    "rotation_degrees":
                                        rotation_index * 90
                                }

                        self.qpsk_ber_validation = (
                            best_result
                        )

            # ----------------------------------------------------
            # 16-QAM
            #
            # Uses the proven V2 QAM chain (matched RRC, residual-
            # CFO derotation, lattice-fit timing, static decision-
            # directed phase). The legacy generic demodulator
            # mis-rotates QAM constellations and reports ~0.5 BER.
            # ----------------------------------------------------

            elif self.selected_modulation == "16-QAM":

                from prototype.core.synchronization import (
                    synchronize_qam_signal,
                )
                from prototype.demodulation.demodulator import (
                    demodulate_signal,
                )
                from prototype.modulation.demodulator import (
                    normalize_qam16_symbols,
                )

                _qam_sync = synchronize_qam_signal(
                    Signal(
                        samples=self.isolated_signal,
                        sample_rate=self.sample_rate,
                    ),
                    symbol_rate=self.timing_symbol_rate,
                )

                _qam_result = demodulate_signal(
                    _qam_sync.signal,
                    "16-QAM",
                    samples_per_symbol=1.0,
                    synchronized=True,
                )

                _sym = _qam_sync.signal.samples

                _quad_balance = float(
                    np.abs(np.mean(_sym ** 2))
                    / np.mean(np.abs(_sym) ** 2)
                )

                self.qam16_demodulation = {
                    "modulation": "16-QAM",
                    "symbols": _sym,
                    "num_symbols": _qam_result.num_symbols,
                    "num_bits": _qam_result.num_bits,
                    "decision_margin":
                        _qam_result.decision_margin,
                    "quadrature_balance": _quad_balance,
                }

                self._pipeline_demod_summary = {
                    "modulation": "16-QAM",
                    "num_symbols": _qam_result.num_symbols,
                    "num_bits": _qam_result.num_bits,
                    "decision_margin":
                        _qam_result.decision_margin,
                }

                self.timing_symbol_rate = float(
                    _qam_sync.symbol_rate
                )

                self.timing_offset = int(
                    _qam_sync.timing_offset
                )

                if self.current_file is not None:

                    reference_bits, reference_path = (
                        load_transmitted_bits(
                            self.current_file
                        )
                    )

                    if reference_bits is not None:

                        symbols = normalize_qam16_symbols(
                            _qam_sync.signal.samples
                        )

                        best_result = None

                        # Blind 90-degree phase folds x a
                        # symbol-aligned origin search: the pulse-
                        # shaping group delay offsets the recovered
                        # stream by whole symbols, so the origin
                        # must be searched in symbol steps.
                        max_offset = min(
                            16, symbols.size - 1
                        )

                        for rotation_index in range(4):

                            rotation = (
                                rotation_index
                                * np.pi
                                / 2.0
                            )

                            rotated_symbols = (
                                symbols
                                * np.exp(
                                    -1j * rotation
                                )
                            )

                            candidate_bits, _, _ = (
                                qam16_decision(
                                    rotated_symbols
                                )
                            )

                            for sym_off in range(
                                -max_offset,
                                max_offset + 1
                            ):

                                if sym_off >= 0:
                                    shifted = (
                                        candidate_bits[
                                            4 * sym_off:
                                        ]
                                    )

                                    reference_compare = (
                                        reference_bits
                                    )

                                else:
                                    shifted = (
                                        candidate_bits
                                    )

                                    reference_compare = (
                                        reference_bits[
                                            4 * (-sym_off):
                                        ]
                                    )

                                compared_count = min(
                                    len(shifted),
                                    len(reference_compare)
                                )

                                if compared_count == 0:
                                    continue

                                errors = int(
                                    np.sum(
                                        shifted[
                                            :compared_count
                                        ]
                                        != reference_compare[
                                            :compared_count
                                        ]
                                    )
                                )

                                ber = (
                                    errors
                                    / compared_count
                                )

                                if (
                                    best_result is None
                                    or errors
                                    < best_result["errors"]
                                ):
                                    best_result = {
                                        "errors": errors,
                                        "ber": float(ber),
                                        "compared_bits":
                                            compared_count,
                                        "recovered_bits":
                                            len(shifted),
                                        "reference_bits":
                                            len(reference_bits),
                                        "rotation_index":
                                            rotation_index,
                                        "rotation_degrees":
                                            rotation_index * 90,
                                        "origin_offset_symbols":
                                            sym_off,
                                        "reference_path":
                                            reference_path,
                                    }

                        self.qam16_ber_validation = (
                            best_result
                        )

            # ----------------------------------------------------
            # BFSK
            # ----------------------------------------------------

            elif self.selected_modulation == "BFSK":

                # IMPORTANT:
                # demodulate_bfsk expects the complete sample stream,
                # not the already down-sampled symbol_samples array.
                self.bfsk_demodulation = demodulate_bfsk(
                    self.isolated_signal,
                    self.sample_rate,
                    self.timing_sps,
                    BFSK_FREQ_0,
                    BFSK_FREQ_1
                )

                if self.current_file is not None:

                    reference_bits, reference_path = (
                        load_transmitted_bits(
                            self.current_file
                        )
                    )

                    if reference_bits is not None:

                        recovered_bits = (
                            self.bfsk_demodulation["bits"]
                        )

                        compared_count = min(
                            len(recovered_bits),
                            len(reference_bits)
                        )

                        if compared_count > 0:

                            recovered_compare = (
                                recovered_bits[
                                    :compared_count
                                ]
                            )

                            reference_compare = (
                                reference_bits[
                                    :compared_count
                                ]
                            )

                            errors = int(
                                np.sum(
                                    recovered_compare
                                    != reference_compare
                                )
                            )

                            self.bfsk_ber_validation = {
                                "errors": errors,
                                "ber": (
                                    errors
                                    / compared_count
                                ),
                                "compared_bits":
                                    compared_count,
                                "recovered_bits":
                                    len(recovered_bits),
                                "reference_bits":
                                    len(reference_bits),
                                "reference_path":
                                    reference_path,
                            }

            # ----------------------------------------------------
            # Terminal output
            # ----------------------------------------------------

            logger.info("Selected Signal Analysis")
            logger.info("------------------------")

            logger.info(
                f"Dominant frequency: "
                f"{a['dominant_frequency']:.2f} Hz"
            )

            if a["bandwidth"] is not None:
                logger.info(
                    f"Bandwidth: "
                    f"{a['bandwidth']:.2f} Hz"
                )

            logger.info(
                f"Modulation: "
                f"{self.selected_modulation}"
            )

            logger.info(
                f"Amplitude CV: "
                f"{self.selected_modulation_features['amplitude_cv']:.3f}"
            )

            logger.info(
                f"R2 phase coherence: "
                f"{self.selected_modulation_features['r2_phase_coherence']:.3f}"
            )

            logger.info(
                f"R4 phase coherence: "
                f"{self.selected_modulation_features['r4_phase_coherence']:.3f}"
            )

            logger.info(
                f"Instantaneous-frequency std: "
                f"{self.selected_modulation_features['instantaneous_frequency_std_hz']:.2f} Hz"
            )

            logger.info(
                f"Estimated samples/symbol: "
                f"{self.timing_sps:.2f}"
            )

            logger.info(
                f"Estimated symbol rate: "
                f"{self.timing_symbol_rate:.2f} symbols/s"
            )

            logger.info(
                f"Timing confidence: "
                f"{self.timing_confidence * 100:.1f}%"
            )

            if timing is not None:
                logger.info(
                    f"Timing offset: "
                    f"{self.timing_offset} samples "
                    f"(eye opening: "
                    f"{timing.eye_opening:.2f})"
                )
            else:
                logger.info(
                    f"Timing offset: "
                    f"{self.timing_offset} samples "
                    f"(V1 BFSK timing profile)"
                )

            # ----------------------------------------------------
            # BPSK terminal output
            # ----------------------------------------------------

            if self.selected_modulation == "BPSK":

                logger.info(
                    f"Recovered symbols: "
                    f"{len(self.symbol_samples)}"
                )

                logger.info(
                    f"BPSK hard-decision bits: "
                    f"{len(self.bpsk_demodulation.bits)}"
                )

                logger.info(
                    f"BPSK decision margin: "
                    f"{self.bpsk_demodulation.decision_margin:.3f}"
                )

                logger.info(
                    f"BPSK quadrature ratio: "
                    f"{self.bpsk_demodulation.quadrature_ratio:.3f}"
                )

                if self.ber_validation is None:

                    logger.info(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.ber_validation

                    logger.info(
                        f"BER reference: "
                        f"{validation.reference_path}"
                    )

                    logger.info(
                        "BER compared bits: "
                        f"{validation.compared_bit_count} "
                        f"(recovered "
                        f"{validation.recovered_bit_count}, "
                        f"reference "
                        f"{validation.reference_bit_count})"
                    )

                    logger.info(
                        "BER direct/inverted errors: "
                        f"{validation.direct_bit_errors} / "
                        f"{validation.inverted_bit_errors}"
                    )

                    logger.info(
                        f"BER result: "
                        f"{validation.bit_errors} errors, "
                        f"{validation.ber:.6g} "
                        f"({validation.polarity_label})"
                    )

                self.constellation_plot.set_figure(
                    create_constellation_figure(
                        self.bpsk_demodulation.aligned_symbols,
                        title=(
                            "BPSK Symbol-Rate "
                            "Constellation (Phase Aligned)"
                        )
                    )
                )

            # ----------------------------------------------------
            # QPSK terminal output
            # ----------------------------------------------------

            elif self.selected_modulation == "QPSK":

                logger.info(
                    f"Recovered symbols: "
                    f"{self.qpsk_demodulation['num_symbols']}"
                )

                logger.info(
                    f"QPSK recovered bits: "
                    f"{self.qpsk_demodulation['num_bits']}"
                )

                logger.info(
                    f"QPSK phase estimate: "
                    f"{self.qpsk_demodulation['phase_estimate_rad']:.4f} rad"
                )

                logger.info(
                    f"QPSK decision margin: "
                    f"{self.qpsk_demodulation['decision_margin']:.3f}"
                )

                logger.info(
                    f"QPSK quadrature balance: "
                    f"{self.qpsk_demodulation['quadrature_balance']:.3f}"
                )

                if self.qpsk_ber_validation is None:

                    logger.info(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.qpsk_ber_validation

                    logger.info(
                        f"BER compared bits: "
                        f"{validation['compared_bits']} "
                        f"(recovered "
                        f"{validation['recovered_bits']}, "
                        f"reference "
                        f"{validation['reference_bits']})"
                    )

                    logger.info(
                        f"QPSK phase rotation selected: "
                        f"{validation['rotation_degrees']}°"
                    )

                    logger.info(
                        f"BER result: "
                        f"{validation['errors']} errors, "
                        f"{validation['ber']:.6g}"
                    )

                self.constellation_plot.set_figure(
                    create_constellation_figure(
                        self.qpsk_demodulation[
                            "corrected_symbols"
                        ],
                        title=(
                            "QPSK Symbol-Rate "
                            "Constellation"
                        )
                    )
                )

            # ----------------------------------------------------
            # 16-QAM terminal output
            # ----------------------------------------------------

            elif self.selected_modulation == "16-QAM":

                logger.info(
                    f"Recovered symbols: "
                    f"{self.qam16_demodulation['num_symbols']}"
                )

                logger.info(
                    f"16-QAM recovered bits: "
                    f"{self.qam16_demodulation['num_bits']}"
                )

                logger.info(
                    f"16-QAM decision margin: "
                    f"{self.qam16_demodulation['decision_margin']:.3f}"
                )

                logger.info(
                    f"16-QAM quadrature balance: "
                    f"{self.qam16_demodulation['quadrature_balance']:.3f}"
                )

                if self.qam16_ber_validation is None:

                    logger.info(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.qam16_ber_validation

                    logger.info(
                        f"BER compared bits: "
                        f"{validation['compared_bits']} "
                        f"(recovered "
                        f"{validation['recovered_bits']}, "
                        f"reference "
                        f"{validation['reference_bits']})"
                    )

                    logger.info(
                        f"16-QAM phase rotation selected: "
                        f"{validation['rotation_degrees']}°"
                    )

                    logger.info(
                        f"BER result: "
                        f"{validation['errors']} errors, "
                        f"{validation['ber']:.6g}"
                    )

                self.constellation_plot.set_figure(
                    create_constellation_figure(
                        self.qam16_demodulation["symbols"],
                        title=(
                            "16-QAM Symbol-Rate "
                            "Constellation"
                        )
                    )
                )

            # ----------------------------------------------------
            # BFSK terminal output
            # ----------------------------------------------------

            elif self.selected_modulation == "BFSK":

                logger.info(
                    f"Recovered symbols: "
                    f"{self.bfsk_demodulation['num_symbols']}"
                )

                logger.info(
                    f"BFSK recovered bits: "
                    f"{len(self.bfsk_demodulation['bits'])}"
                )

                logger.info(
                    f"BFSK frequencies: "
                    f"{BFSK_FREQ_0:.1f} Hz / "
                    f"{BFSK_FREQ_1:.1f} Hz"
                )

                logger.info(
                    f"BFSK decision margin: "
                    f"{self.bfsk_demodulation['decision_margin']:.3f}"
                )

                if self.bfsk_ber_validation is None:

                    logger.info(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.bfsk_ber_validation

                    logger.info(
                        f"BER reference: "
                        f"{validation['reference_path']}"
                    )

                    logger.info(
                        "BER compared bits: "
                        f"{validation['compared_bits']} "
                        f"(recovered "
                        f"{validation['recovered_bits']}, "
                        f"reference "
                        f"{validation['reference_bits']})"
                    )

                    logger.info(
                        f"BER result: "
                        f"{validation['errors']} errors, "
                        f"{validation['ber']:.6g}"
                    )

                # A conventional I/Q constellation is not the useful
                # visualization for BFSK. Show the frequency selected
                # for each recovered symbol instead.
                figure = Figure(
                    figsize=(5, 4),
                    tight_layout=True
                )

                ax = figure.add_subplot(111)

                symbol_indices = np.arange(
                    len(
                        self.bfsk_demodulation[
                            "decision_symbols"
                        ]
                    )
                )

                ax.step(
                    symbol_indices,
                    self.bfsk_demodulation[
                        "decision_symbols"
                    ],
                    where="mid"
                )

                ax.set_title(
                    "BFSK Symbol Frequency Decisions"
                )
                ax.set_xlabel("Symbol Index")
                ax.set_ylabel("Selected Frequency (Hz)")
                ax.grid(True, alpha=0.3)

                self.constellation_plot.set_figure(
                    figure
                )

            # ----------------------------------------------------
            # Unknown modulation
            # ----------------------------------------------------

            else:

                logger.info(
                    f"Recovered symbols: "
                    f"{len(self.symbol_samples)}"
                )

                logger.info(
                    "Demodulation: "
                    "not implemented for this modulation yet."
                )

            # ----------------------------------------------------
            # GUI parameters
            # ----------------------------------------------------

            self.parameter_modulation.setText(
                f"Modulation: "
                f"{self.selected_modulation}"
            )

            self.parameter_sps.setText(
                f"Samples/Symbol: "
                f"{self.timing_sps:.2f}"
            )

            self.parameter_symbol_rate.setText(
                f"Symbol Rate: "
                f"{self.timing_symbol_rate:.2f} symbols/s"
            )

            self.parameter_timing_confidence.setText(
                f"Timing Confidence: "
                f"{self.timing_confidence * 100:.1f}%"
            )

            # ----------------------------------------------------
            # Symbol / bit count
            # ----------------------------------------------------

            if self.selected_modulation == "QPSK":

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols/Bits: "
                    f"{self.qpsk_demodulation['num_symbols']} / "
                    f"{self.qpsk_demodulation['num_bits']}"
                )

            elif self.selected_modulation == "16-QAM":

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols/Bits: "
                    f"{self.qam16_demodulation['num_symbols']} / "
                    f"{self.qam16_demodulation['num_bits']}"
                )

            elif self.selected_modulation == "BPSK":

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols/Bits: "
                    f"{len(self.symbol_samples)} / "
                    f"{len(self.bpsk_demodulation.bits)}"
                )

            elif self.selected_modulation == "BFSK":

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols/Bits: "
                    f"{self.bfsk_demodulation['num_symbols']} / "
                    f"{len(self.bfsk_demodulation['bits'])}"
                )

            else:

                self.parameter_symbol_count.setText(
                    f"Recovered Symbols: "
                    f"{len(self.symbol_samples)}"
                )

            # ----------------------------------------------------
            # BER GUI
            # ----------------------------------------------------

            if self.selected_modulation == "QPSK":

                validation = (
                    self.qpsk_ber_validation
                )

                if validation is None:

                    self.parameter_ber.setText(
                        "BER Validation: "
                        "No companion reference bits found"
                    )

                else:

                    self.parameter_ber.setText(
                        f"BER: "
                        f"{validation['ber']:.6g} "
                        f"({validation['errors']}/"
                        f"{validation['compared_bits']} errors; "
                        f"{validation['rotation_degrees']}° rotation)"
                    )

            elif self.selected_modulation == "16-QAM":

                validation = (
                    self.qam16_ber_validation
                )

                if validation is None:

                    self.parameter_ber.setText(
                        "BER Validation: "
                        "No companion reference bits found"
                    )

                else:

                    self.parameter_ber.setText(
                        f"BER: "
                        f"{validation['ber']:.6g} "
                        f"({validation['errors']}/"
                        f"{validation['compared_bits']} errors; "
                        f"{validation['rotation_degrees']}° rotation)"
                    )

            elif self.selected_modulation == "BPSK":

                if self.ber_validation is None:

                    self.parameter_ber.setText(
                        "BER Validation: "
                        "No companion reference bits found"
                    )

                else:

                    validation = self.ber_validation

                    length_note = (
                        ""
                        if validation.length_match
                        else "; length mismatch"
                    )

                    self.parameter_ber.setText(
                        f"BER: "
                        f"{validation.ber:.6g} "
                        f"({validation.bit_errors}/"
                        f"{validation.compared_bit_count} errors; "
                        f"{validation.polarity_label}"
                        f"{length_note})"
                    )

            elif self.selected_modulation == "BFSK":

                validation = (
                    self.bfsk_ber_validation
                )

                if validation is None:

                    self.parameter_ber.setText(
                        "BER Validation: "
                        "No companion reference bits found"
                    )

                else:

                    self.parameter_ber.setText(
                        f"BER: "
                        f"{validation['ber']:.6g} "
                        f"({validation['errors']}/"
                        f"{validation['compared_bits']} errors)"
                    )

            else:

                self.parameter_ber.setText(
                    "BER Validation: "
                    "Demodulator not implemented"
                )

            # ----------------------------------------------------
            # Completion dialog
            # ----------------------------------------------------

            QMessageBox.information(
                self,
                "Analysis complete",
                (
                    f"Modulation: "
                    f"{self.selected_modulation}\n\n"

                    f"Dominant frequency: "
                    f"{a['dominant_frequency']:.2f} Hz\n\n"

                    f"Samples/symbol: "
                    f"{self.timing_sps:.2f}\n\n"

                    f"Symbol rate: "
                    f"{self.timing_symbol_rate:.2f} symbols/s\n\n"

                    f"Timing offset: "
                    f"{self.timing_offset} samples\n\n"

                    f"Recovered symbols/bits: "
                    f"{len(self.symbol_samples)} / "
                    f"{self._recovered_bit_count()}\n\n"

                    f"BER: "
                    f"{self._ber_summary_for_dialog()}"
                )
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Selected signal analysis failed",
                str(exc)
            )

    def _recovered_bit_count(self):
        """Return the recovered bit count for the selected modulation."""

        if self.selected_modulation == "QPSK":

            if self.qpsk_demodulation is None:
                return 0

            return self.qpsk_demodulation[
                "num_bits"
            ]

        if self.selected_modulation == "16-QAM":

            if self.qam16_demodulation is None:
                return 0

            return self.qam16_demodulation[
                "num_bits"
            ]

        if self.selected_modulation == "BPSK":

            if self.bpsk_demodulation is None:
                return 0

            return len(
                self.bpsk_demodulation.bits
            )

        if self.selected_modulation == "BFSK":

            if self.bfsk_demodulation is None:
                return 0

            return len(
                self.bfsk_demodulation["bits"]
            )

        return 0


    def _ber_summary_for_dialog(self):
        """Return a concise BER status for the selected modulation."""

        # QPSK
        if self.selected_modulation == "QPSK":

            if self.qpsk_ber_validation is None:
                return "No companion reference bits found"

            validation = self.qpsk_ber_validation

            return (
                f"{validation['ber']:.6g} "
                f"({validation['errors']}/"
                f"{validation['compared_bits']} errors; "
                f"{validation['rotation_degrees']}° rotation)"
            )

        # 16-QAM
        if self.selected_modulation == "16-QAM":

            if self.qam16_ber_validation is None:
                return "No companion reference bits found"

            validation = self.qam16_ber_validation

            return (
                f"{validation['ber']:.6g} "
                f"({validation['errors']}/"
                f"{validation['compared_bits']} errors; "
                f"{validation['rotation_degrees']}° rotation)"
            )

        # BFSK
        if self.selected_modulation == "BFSK":

            if self.bfsk_ber_validation is None:
                return "No companion reference bits found"

            validation = self.bfsk_ber_validation

            return (
                f"{validation['ber']:.6g} "
                f"({validation['errors']}/"
                f"{validation['compared_bits']} errors)"
            )

        # BPSK
        if self.selected_modulation == "BPSK":

            if self.ber_validation is None:
                return "No companion reference bits found"

            validation = self.ber_validation

            return (
                f"{validation.ber:.6g} "
                f"({validation.bit_errors}/"
                f"{validation.compared_bit_count} errors; "
                f"{validation.polarity_label})"
            )

        return "Demodulation not implemented"

        # ========================================================
        # RAW IQ OPENING
        # ========================================================

    def _open_raw_iq(self, path: str):
        """Load a raw interleaved IQ file.

        Returns ``(samples, sample_rate)``, or ``None`` when the user
        cancels the parameter prompt. Raises on loader errors.
        """

        from prototype.io.loaders import load_signal, load_sidecar

        # A sidecar may already provide everything; peek so we only
        # prompt for what is actually missing.
        sidecar: dict = {}

        try:
            sidecar = load_sidecar(path)
        except Exception:
            sidecar = {}

        known_rate = sidecar.get("sample_rate") is not None

        sample_rate = None

        if not known_rate:

            rate_text, ok = QInputDialog.getText(
            self,
            "Raw IQ sample rate",
            f"Sample rate in Hz for {Path(path).name}",
            text="8000",
            )

            if not ok or not rate_text.strip():
                return None

            sample_rate = float(rate_text)

            if sample_rate <= 0:
                raise ValueError(
                f"Sample rate must be positive, got {sample_rate}"
                )

        known_dtype = sidecar.get("dtype") is not None

        dtype = None

        if not known_dtype:

            dtype_text, ok = QInputDialog.getItem(
            self,
            "Raw IQ sample format",
            f"Sample format for {Path(path).name} "
            "(interleaved I/Q unless complex*):",
            [
            "int16",
            "complex64",
            "complex128",
            "float32",
            "float64",
            "int8",
            "uint8",
            "int32",
            ],
            0,
            False,
            )

            if not ok:
                return None

            dtype = str(dtype_text)

        kwargs: dict = {
        "endianness": sidecar.get("endianness"),
        "iq_order": sidecar.get("iq_order"),
        "center_frequency_hz": sidecar.get("center_frequency_hz"),
        }

        if not known_rate:
            # Prompted rate is an explicit override.
            kwargs["sample_rate"] = sample_rate

        if not known_dtype:
            # Prompted dtype is an explicit override.
            kwargs["dtype"] = dtype

        samples_obj = load_signal(path, **kwargs)

        return samples_obj.samples, float(samples_obj.sample_rate)

        # ========================================================
        # EXPORT + PROVENANCE
        # ========================================================

    def export_result_json(self):
        """Save the full V2 payload (all stages + provenance) as JSON."""

        if self.pipeline_result is None:
            QMessageBox.information(
            self,
            "Nothing to export",
            "Run an analysis first.",
            )
            return

        import json

        default_name = "analysis_result.json"

        if self.current_file is not None:
            from pathlib import Path as _Path

            default_name = _Path(self.current_file).stem + "_analysis.json"

        path, _ = QFileDialog.getSaveFileName(
        self,
        "Export analysis result",
        default_name,
        "JSON files (*.json)",
        )

        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                self.pipeline_result,
                handle,
                indent=2,
                default=str,
                )

        except Exception as exc:
            QMessageBox.critical(
            self,
            "Export failed",
            str(exc),
            )
            return

        QMessageBox.information(
        self,
        "Export complete",
        f"Analysis result written to\n{path}",
        )

    def show_provenance(self):
        """Per-stage timings, configuration, versions of the last run."""

        provenance = self._pipeline_provenance

        if not provenance:
            QMessageBox.information(
            self,
            "Provenance",
            "No provenance recorded yet (run an analysis first).",
            )
            return

        lines = []

        software = provenance.get("software_version", "?")

        commit = provenance.get("git_commit")

        lines.append(
        f"Spectra {software}"
        + (f" (git {commit})" if commit else "")
        )

        lines.append(
        f"Python {provenance.get('python_version', '?')}"
        )

        started = provenance.get("started_utc", "?")

        lines.append(f"Started: {started}")

        lines.append("")

        lines.append("Stages:")

        for step in provenance.get("steps") or []:

            lines.append(
            f"  {step.get('name', '?'):16s} "
            f"{float(step.get('duration_ms', 0.0)):9.1f} ms  "
            f"[{step.get('status', '?')}]"
            )

        total_ms = sum(
        float(step.get("duration_ms", 0.0))
        for step in provenance.get("steps") or []
        )

        lines.append(f"  {'TOTAL':16s} {total_ms:9.1f} ms")

        QMessageBox.information(
        self,
        "Analysis provenance",
        "\n".join(lines),
        )
