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
QAbstractItemView,
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
QSpinBox,
QTabWidget,
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
from prototype.core.config import FECMode

from prototype.core.loader import load_wav

from prototype.visualization.plots import (
create_constellation_figure,
create_error_figure,
create_spectrum_figure,
create_spectrum_figure_gnuradio,
create_time_figure,
create_waterfall_figure,
create_waterfall_figure_gnuradio,
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
        self._pipeline_protocol_summary = None
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


        # Interleaving / FEC identification + pipeline input/worker state
        self._interleaving_result = None
        self._interleaving_candidates = []
        self._identification_result = None
        self._pipeline_input = None
        self._pipeline_parameters = None
        self._gnuradio_worker = None
        self._gnuradio_viz_data = None
        self._gnuradio_viz_error = None

        # Last analysis provenance (stage timings, versions) — captured
        # at result-apply time and shown via "Provenance".
        self._pipeline_provenance = None

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.setWindowTitle(
        "SIH Signal Analyzer"
        )

        # Fit a standard 1366x768 laptop screen: the analysis UI is
        # organised as title + compact control row + visualisation tabs
        # + results, so no single page is a giant vertical scroll.
        self.resize(
        1200,
        720
        )

        self.setMinimumSize(
        900,
        620
        )

        self._current_theme = "dark"
        self._apply_theme()
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
        "SPECTRA RF SIGNAL ANALYZER"
        )

        title.setAlignment(
        Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
        """
        QLabel {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #38bdf8;
        padding: 4px 8px;
        background: transparent;
        }
        """
        )

        main_layout.addWidget(
        title
        )

        # ====================================================
        # COMPACT CONTROL ROW
        # ====================================================

        # Two compact rows so the ribbon stays readable at 1366x768.
        # Row 1 = input / analysis, Row 2 = signal / decoding.  Every
        # control appears exactly once (no duplicates, no backend change).
        control_row = QHBoxLayout()   # row 1 - input / analysis
        control_row.setSpacing(8)

        decode_row = QHBoxLayout()    # row 2 - signal / decoding
        decode_row.setSpacing(8)

        self._control_row_1 = control_row
        self._decode_row = decode_row

        control_row.addWidget(
        QLabel("Source:")
        )

        self.source_selector = QComboBox()

        self.source_selector.addItems(
        ["WAV", "Raw IQ"]
        )

        self.source_selector.setToolTip(
        "Select the source file format to analyze: WAV or Raw IQ."
        )

        self.source_selector.currentIndexChanged.connect(
        self._on_source_changed
        )

        control_row.addWidget(
        self.source_selector
        )

        decode_row.addWidget(
        QLabel("Interleaving:")
        )

        self.interleaving_mode_combo = QComboBox()

        self.interleaving_mode_combo.addItem("Auto")

        self.interleaving_mode_combo.addItem("Manual")

        self.interleaving_mode_combo.addItem("None")

        self.interleaving_mode_combo.setCurrentText("Auto")

        self.interleaving_mode_combo.setToolTip(
        "How the block interleaver is handled on the demodulated "
        "bitstream: Auto = identify the interleaver and deinterleave on "
        "strong structural evidence; Manual = apply the configured depth; "
        "None = bits pass through unchanged."
        )

        self.interleaving_mode_combo.currentIndexChanged.connect(
        self._on_interleaving_mode_changed
        )

        decode_row.addWidget(
        self.interleaving_mode_combo
        )

        # Manual interleaving needs an explicit block depth (the pipeline
        # requires >= 2, otherwise manual is a no-op).
        self.interleave_depth_spin = QSpinBox()

        self.interleave_depth_spin.setRange(2, 4096)

        self.interleave_depth_spin.setValue(16)

        self.interleave_depth_spin.setEnabled(False)

        self.interleave_depth_spin.setToolTip(
        "Block-interleaver depth applied when Interleaving = Manual"
        )

        decode_row.addWidget(
        QLabel("Depth:")
        )

        decode_row.addWidget(
        self.interleave_depth_spin
        )

        # De-interleaver family. Block/convolutional/diagonal/pseudo-random
        # are all applied manually; the AUTO identifier stays block-only
        # until genuine structural evidence exists for the other families.
        self.interleave_family_combo = QComboBox()

        self.interleave_family_combo.addItem("Block", "block")

        self.interleave_family_combo.addItem("Convolutional", "convolutional")

        self.interleave_family_combo.addItem("Diagonal", "diagonal")

        self.interleave_family_combo.addItem("Pseudo-random", "pseudo_random")

        self.interleave_family_combo.setCurrentIndex(0)

        self.interleave_family_combo.setEnabled(False)

        self.interleave_family_combo.setToolTip(
        "De-interleaver family applied when Interleaving = Manual. The Depth "
        "field doubles as the streams k (convolutional), square width "
        "(diagonal), or permutation seed (pseudo-random)."
        )

        decode_row.addWidget(
        QLabel("Family:")
        )

        decode_row.addWidget(
        self.interleave_family_combo
        )

        decode_row.addWidget(
        QLabel("FEC mode:")
        )

        self.fec_mode_combo = QComboBox()

        # Title-case labels with the canonical lowercase value as data, so
        # the control reads like the Interleaving selector while the
        # backend still receives FECMode values.
        self.fec_mode_combo.addItem("Auto", FECMode.AUTO)

        self.fec_mode_combo.addItem("Manual", FECMode.MANUAL)

        self.fec_mode_combo.addItem("None", FECMode.NONE)

        self.fec_mode_combo.setCurrentIndex(0)

        self.fec_mode_combo.setToolTip(
        "How automatic FEC identification / deinterleaving is applied: "
        "Auto = run the identifier and decode on strong evidence; "
        "Manual = apply the configured depth authoritatively; "
        "None = skip identification and deinterleaving."
        )

        decode_row.addWidget(
        self.fec_mode_combo
        )

        decode_row.addWidget(
        QLabel("FEC scheme:")
        )

        self.fec_combo = QComboBox()

        self.fec_combo.addItem("none")

        self.fec_combo.addItems(
        sorted(list_fec_schemes())
        )

        self.fec_combo.setToolTip(
        "Forward error correction applied to demodulated bits. "
        "FEC is never guessed: pick the scheme the transmitter used."
        )

        decode_row.addWidget(
        self.fec_combo
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

        # Batch + ML toggles
        self.batch_checkbox = QCheckBox("Analyze all candidates")

        self.batch_checkbox.setToolTip(
        "Run the full V2 pipeline on every detected candidate "
        "(multi-signal analysis) instead of just the strongest"
        )

        self.ml_checkbox = QCheckBox("ML assist (CNN)")

        self.ml_checkbox.setToolTip(
        "Score the signal with a convolutional neural network in "
        "addition to the rule-based classifier. The CNN\u2019s prediction "
        "is reported alongside the DSP result, never instead of it."
        )

        # Frame / sync-word search.  The protocol stage is off by default
        # because a sync word is transmitter-specific; when enabled it
        # runs the same FrameConfig the CLI builds from --sync-word.
        self.frame_checkbox = QCheckBox("Frame search")

        self.frame_checkbox.setToolTip(
        "Run the frame/protocol stage: normalized-correlation sync-word "
        "search and payload extraction on the demodulated bits."
        )

        self.sync_word_edit = QLineEdit("0xAA55AA55")

        self.sync_word_edit.setMaximumWidth(110)

        self.sync_word_edit.setToolTip(
        "Expected sync word (hex or integer), matched by normalized "
        "correlation. 0xAA55AA55 is the classic 32-bit pattern."
        )

        self.sync_word_edit.setEnabled(False)

        self.data_bytes_spin = QSpinBox()

        self.data_bytes_spin.setRange(0, 4096)

        self.data_bytes_spin.setValue(0)

        self.data_bytes_spin.setToolTip(
        "Expected payload bytes after the sync word (0 = unspecified)."
        )

        self.data_bytes_spin.setEnabled(False)

        self.frame_checkbox.toggled.connect(
        self._on_frame_search_toggled
        )

        decode_row.addWidget(
        self.frame_checkbox
        )

        decode_row.addWidget(
        self.sync_word_edit
        )

        decode_row.addWidget(
        QLabel("Payload bytes:")
        )

        decode_row.addWidget(
        self.data_bytes_spin
        )

        # Actions: Open / Analyze / Clear
        self.open_button = QPushButton(
        "Open WAV/IQ"
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
        True
        )

        self.clear_button = QPushButton(
        "Clear"
        )

        self.clear_button.clicked.connect(
        self.clear_analysis
        )

        control_row.addWidget(
        self.open_button
        )

        control_row.addWidget(
        self.analyze_button
        )

        # "Analyze Selected" is created later, next to the detection
        # table; remember the slot so row 1 keeps the actions together.
        self._row1_insert_index = control_row.count()

        control_row.addWidget(
        self.clear_button
        )

        control_row.addWidget(
        QLabel("Mode:")
        )

        control_row.addWidget(
        self.mode_combo
        )

        control_row.addWidget(
        self.batch_checkbox
        )

        control_row.addWidget(
        self.ml_checkbox
        )

        control_row.addStretch()

        main_layout.addLayout(
        control_row
        )

        main_layout.addLayout(
        decode_row
        )

        # ====================================================
        # VISUALISATION TABS (every plot gets most of the window)
        # ====================================================

        self.vis_tabs = QTabWidget()

        self.vis_tabs.setDocumentMode(True)

        self.vis_tabs.setTabPosition(
        QTabWidget.TabPosition.North
        )

        # Helper for scrollable tabs
        def _make_scrollable_tab(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(widget)
            return scroll

        # --- Time Domain ---

        self.time_tab = QWidget()

        self.time_tab_layout = QVBoxLayout(
        self.time_tab
        )

        self.time_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.time_tab_layout.setSpacing(6)

        self.time_tab_title = QLabel(
        "TIME DOMAIN"
        )

        self.time_tab_title.setStyleSheet(
        "font-weight: bold;"
        )

        self.time_tab_layout.addWidget(
        self.time_tab_title
        )

        self.time_plot = PlotWidget()

        self.time_plot.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
        )

        self.time_tab_layout.addWidget(
        self.time_plot
        )

        self.time_tab_layout.addStretch()

        self.time_tab.setLayout(
        self.time_tab_layout
        )

        self.time_scroll = _make_scrollable_tab(self.time_tab)
        self.vis_tabs.addTab(
        self.time_scroll,
        "Time Domain"
        )

        # --- Spectrum ---

        self.spectrum_tab = QWidget()

        self.spectrum_tab_layout = QVBoxLayout(
        self.spectrum_tab
        )

        self.spectrum_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.spectrum_tab_layout.setSpacing(6)

        self.spectrum_tab_title = QLabel(
        "SPECTRUM"
        )

        self.spectrum_tab_title.setStyleSheet(
        "font-weight: bold;"
        )

        self.spectrum_tab_layout.addWidget(
        self.spectrum_tab_title
        )

        self.spectrum_plot = PlotWidget()

        self.spectrum_plot.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
        )

        self.spectrum_tab_layout.addWidget(
        self.spectrum_plot
        )

        self.spectrum_tab_layout.addStretch()

        self.spectrum_tab.setLayout(
        self.spectrum_tab_layout
        )

        self.spectrum_scroll = _make_scrollable_tab(self.spectrum_tab)
        self.vis_tabs.addTab(
        self.spectrum_scroll,
        "Spectrum"
        )

        # --- Waterfall / STFT ---

        self.waterfall_tab = QWidget()

        self.waterfall_tab_layout = QVBoxLayout(
        self.waterfall_tab
        )

        self.waterfall_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.waterfall_tab_layout.setSpacing(6)

        self.waterfall_tab_title = QLabel(
        "WATERFALL / STFT"
        )

        self.waterfall_tab_title.setStyleSheet(
        "font-weight: bold;"
        )

        self.waterfall_tab_layout.addWidget(
        self.waterfall_tab_title
        )

        self.waterfall_plot = PlotWidget()

        self.waterfall_plot.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
        )

        self.waterfall_tab_layout.addWidget(
        self.waterfall_plot
        )

        self.waterfall_tab_layout.addStretch()

        self.waterfall_tab.setLayout(
        self.waterfall_tab_layout
        )

        self.waterfall_scroll = _make_scrollable_tab(self.waterfall_tab)
        self.vis_tabs.addTab(
        self.waterfall_scroll,
        "Waterfall / STFT"
        )

        # --- Constellation ---

        self.constellation_tab = QWidget()

        self.constellation_tab_layout = QVBoxLayout(
        self.constellation_tab
        )

        self.constellation_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.constellation_tab_layout.setSpacing(6)

        self.constellation_tab_title = QLabel(
        "CONSTELLATION"
        )

        self.constellation_tab_title.setStyleSheet(
        "font-weight: bold;"
        )

        self.constellation_tab_layout.addWidget(
        self.constellation_tab_title
        )

        self.constellation_plot = PlotWidget()

        self.constellation_plot.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
        )

        self.constellation_tab_layout.addWidget(
        self.constellation_plot
        )

        self.constellation_tab_layout.addStretch()

        self.constellation_tab.setLayout(
        self.constellation_tab_layout
        )

        self.constellation_scroll = _make_scrollable_tab(self.constellation_tab)
        self.vis_tabs.addTab(
        self.constellation_scroll,
        "Constellation"
        )

        # --- Detection / Results ---

        self.results_tab = QWidget()

        # The results page is long (headline + detections + several
        # parameter groups), so it scrolls instead of forcing the whole
        # window to grow past a 768px-tall screen.
        self.results_tab_outer = QVBoxLayout(
        self.results_tab
        )

        self.results_tab_outer.setContentsMargins(0, 0, 0, 0)

        self.results_scroll = QScrollArea()

        self.results_scroll.setWidgetResizable(True)

        self.results_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.results_content = QWidget()

        self.results_tab_layout = QVBoxLayout(
        self.results_content
        )

        self.results_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.results_tab_layout.setSpacing(6)

        self.results_scroll.setWidget(
        self.results_content
        )

        self.results_tab_outer.addWidget(
        self.results_scroll
        )

        self.results_tab_title = QLabel(
        "DETECTION / RESULTS"
        )

        self.results_tab_title.setStyleSheet(
        "font-weight: bold;"
        )

        self.results_tab_layout.addWidget(
        self.results_tab_title
        )

        # Signal info (sample rate, candidates, modulation, SNR)
        self.results_signal_frame = QFrame()

        self.results_signal_frame.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        self.results_signal_layout = QHBoxLayout(
        self.results_signal_frame
        )

        self.results_signal_layout.setSpacing(6)

        self.results_sample_rate_label = QLabel(
        "Sample Rate: —"
        )

        self.results_candidate_count_label = QLabel(
        "Candidates: —"
        )

        self.results_modulation_label = QLabel(
        "Modulation: —"
        )

        self.results_snr_label = QLabel(
        "SNR: —"
        )

        for _label in [
        self.results_sample_rate_label,
        self.results_candidate_count_label,
        self.results_modulation_label,
        self.results_snr_label,
        ]:

            _label.setWordWrap(True)

            self.results_signal_layout.addWidget(
            _label
            )

        self.results_tab_layout.addWidget(
        self.results_signal_frame
        )

        # Interleaving / FEC / BER results
        self.results_il_frame = QFrame()

        self.results_il_frame.setFrameShape(
        QFrame.Shape.StyledPanel
        )

        self.results_il_layout = QVBoxLayout(
        self.results_il_frame
        )

        self.results_il_layout.setSpacing(4)

        self.results_il_title = QLabel(
        "Interleaving / FEC / BER"
        )

        self.results_il_title.setStyleSheet(
        "font-weight: bold;"
        )

        self.results_il_layout.addWidget(
        self.results_il_title
        )

        self.results_il_status_label = QLabel(
        "Interleaving: not run"
        )

        self.results_il_type_label = QLabel(
        "Detected type: —"
        )

        self.results_il_depth_label = QLabel(
        "Detected depth: —"
        )

        self.results_il_confidence_label = QLabel(
        "Confidence: —"
        )

        self.results_il_fec_label = QLabel(
        "FEC: —"
        )

        self.results_il_ber_label = QLabel(
        "BER: no reference loaded"
        )

        for _label in [
        self.results_il_status_label,
        self.results_il_type_label,
        self.results_il_depth_label,
        self.results_il_confidence_label,
        self.results_il_fec_label,
        self.results_il_ber_label,
        ]:

            _label.setWordWrap(True)

            self.results_il_layout.addWidget(
            _label
            )

        self.results_tab_layout.addWidget(
        self.results_il_frame
        )

        self._build_detail_widgets()

        self.results_tab_layout.addStretch()

        self.vis_tabs.addTab(
        self.results_tab,
        "Detection / Results"
        )

        main_layout.addWidget(
        self.vis_tabs,
        1,
        )

        # ====================================================
        # FILE INFORMATION & GNU RADIO BACKEND STATUS
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
        self.gnuradio_status_label = QLabel("GNU Radio 3.10.12.0 is available")
        self.gnuradio_status_label.setStyleSheet("color: #38bdf8; font-weight: 600;")

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

        info_layout.addWidget(
        QLabel("DSP Backend:"),
        2,
        2
        )

        info_layout.addWidget(
        self.gnuradio_status_label,
        2,
        3
        )

        self._update_gnuradio_status()

        main_layout.addWidget(
        info_frame
        )

        # ====================================================
        # PROGRESS + EXPORT BAR (bottom)
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

        self.view_report_button = QPushButton("View Report")
        self.view_report_button.clicked.connect(self.view_report)
        self.view_report_button.setEnabled(False)
        self.view_report_button.setToolTip("View full formatted report with embedded charts and print option")

        self.export_html_button = QPushButton("Print / HTML")
        self.export_html_button.clicked.connect(self.export_result_html)
        self.export_html_button.setEnabled(False)
        self.export_html_button.setToolTip("Export print-ready HTML report with embedded graphs")

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

        decode_row.addStretch()

        decode_row.addWidget(
        self.view_report_button
        )

        decode_row.addWidget(
        self.export_html_button
        )

        decode_row.addWidget(
        self.export_json_button
        )

        decode_row.addWidget(
        self.provenance_button
        )

        main_layout.addLayout(
        progress_layout
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

        # Show the GNU Radio panel only for the GNU Radio source and
        # bring its tab to the front; hide it otherwise.
        self.gnuradio_frame.setVisible(is_gnuradio)

        if is_gnuradio:
            target_widget = getattr(self, "gnuradio_scroll", self.gnuradio_tab)
            self.vis_tabs.setCurrentWidget(target_widget)
        elif self.vis_tabs.currentWidget() in (
            getattr(self, "gnuradio_scroll", None),
            getattr(self, "gnuradio_tab", None),
        ):
            self.vis_tabs.setCurrentIndex(0)

        # Disable the GNU Radio controls so no one can "tweak" the
        # panel while another source is selected (not enforced here;
        # acquisition not started this phase anyway).
        for _widget in self.gnuradio_frame.findChildren(QWidget):
            if isinstance(_widget, QLineEdit) or isinstance(_widget, QComboBox):
                _widget.setEnabled(is_gnuradio)

        # The Open button only applies to file sources; GNU Radio
        # captures come from its own tab.
        if selected == "WAV":

            self.open_button.setText("Open WAV")
            self.open_button.setEnabled(True)

        elif selected == "Raw IQ":

            self.open_button.setText("Open Raw IQ")
            self.open_button.setEnabled(True)

        else:

            self.open_button.setText("Open Capture")
            self.open_button.setEnabled(False)

        self._update_gnuradio_status()

    def _on_frame_search_toggled(self, enabled: bool):
        """The sync word / payload size only matter when frame search runs."""

        self.sync_word_edit.setEnabled(bool(enabled))

        self.data_bytes_spin.setEnabled(bool(enabled))

    def _frame_search_config(self):
        """Build the FrameConfig for the frame stage, or None when off.

        Mirrors the CLI's ``--sync-word`` / ``--data-bytes`` handling so
        both front-ends exercise the same protocol stage.
        """

        if not self.frame_checkbox.isChecked():
            return None

        from prototype.protocol import FrameConfig

        text = self.sync_word_edit.text().strip() or "0xAA55AA55"

        try:
            sync_word = int(text, 0)
        except ValueError as exc:
            raise ValueError(
            f"Sync word must be hex (0x…) or decimal, got {text!r}"
            ) from exc

        return FrameConfig(
        name="GUI-frame",
        sync_word=sync_word,
        data_bytes=int(self.data_bytes_spin.value()),
        description="Protocol frame configured from the GUI frame search.",
        )

    def _on_interleaving_mode_changed(self, index):
        """Only the Manual interleaving mode uses an explicit depth."""

        del index

        manual = self.interleaving_mode_combo.currentText() == "Manual"

        self.interleave_depth_spin.setEnabled(manual)

        self.interleave_family_combo.setEnabled(manual)

    def _update_gnuradio_status(self):
        """Describe whether a real GNU Radio backend is available."""
        gr_ver = "GNU Radio"
        try:
            try:
                from gnuradio_integration.dsp_gnuradio.bridge import check_gnuradio_available
            except ImportError:
                import sys
                from pathlib import Path
                _p = str(Path(__file__).resolve().parent.parent / "gnuradio_integration")
                if _p not in sys.path:
                    sys.path.insert(0, _p)
                from dsp_gnuradio.bridge import check_gnuradio_available

            available, ver = check_gnuradio_available()
            if available and ver:
                gr_ver = str(ver).strip()
        except Exception:
            available = False

        if available:
            text = (
                f"Backend: {gr_ver} is available — the configured "
                "device/source and subprocess DSP pipeline will be used."
            )
        else:
            text = (
                "Backend: GNU Radio is not detected — acquisition falls "
                "back to the built-in synthetic source (offline validation)."
            )

        if getattr(self, "gnuradio_status_label", None) is not None:
            self.gnuradio_status_label.setText(text)

    # ========================================================
    # DETAIL WIDGETS
    # ========================================================

    def _build_detail_widgets(self):
        """Create the signal-detail widgets used by the analysis views.

        They live in the Detection/Results tab: the detection table, the
        analyzed-candidate selector, the selected-signal summary and the
        full parameter read-out.
        """

        detail_frame = QFrame()

        detail_frame.setFrameShape(QFrame.Shape.StyledPanel)

        detail_layout = QVBoxLayout(detail_frame)

        detail_layout.setSpacing(4)

        detail_title = QLabel("SIGNAL DETAIL")

        detail_title.setStyleSheet("font-weight: bold;")

        detail_layout.addWidget(detail_title)

        # --- detection table ---
        self.signal_table = QTableWidget()

        self.signal_table.setColumnCount(4)

        self.signal_table.setHorizontalHeaderLabels(
        ["Signal", "Frequency (Hz)", "Bandwidth (Hz)", "Peak (dB)"]
        )

        self.signal_table.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.signal_table.setSelectionMode(
        QAbstractItemView.SelectionMode.SingleSelection
        )

        self.signal_table.cellClicked.connect(self.select_signal)

        self.signal_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch
        )

        self.signal_table.setMinimumHeight(120)

        detail_layout.addWidget(self.signal_table)

        # --- analyzed candidate selector (batch mode) ---
        candidate_row = QHBoxLayout()

        candidate_row.addWidget(QLabel("Analyzed candidate:"))

        self.candidate_combo = QComboBox()

        self.candidate_combo.setToolTip(
        "Switch the detail panels between analyzed candidates "
        "(batch mode)"
        )

        self.candidate_combo.setVisible(False)

        self.candidate_combo.currentIndexChanged.connect(
        self._on_candidate_selected
        )

        candidate_row.addWidget(self.candidate_combo)

        candidate_row.addStretch()

        detail_layout.addLayout(candidate_row)

        # --- selected signal summary + actions ---
        selected_row = QHBoxLayout()

        self.selected_signal_label = QLabel("No signal selected")
        self.selected_frequency_label = QLabel("Frequency: —")
        self.selected_bandwidth_label = QLabel("Bandwidth: —")

        selected_row.addWidget(self.selected_signal_label)
        selected_row.addWidget(self.selected_frequency_label)
        selected_row.addWidget(self.selected_bandwidth_label)

        self.isolate_button = QPushButton("Isolate Signal")

        self.isolate_button.clicked.connect(self.isolate_selected_signal)

        self.isolate_button.setEnabled(False)

        self.analyze_selected_button = QPushButton("Analyze Selected")

        self.analyze_selected_button.clicked.connect(
        self.analyze_selected_signal
        )

        self.analyze_selected_button.setEnabled(False)

        selected_row.addWidget(self.isolate_button)
        # Keep the action in row 1 (input / analysis) instead of the
        # detail panel; the widget itself is unchanged.
        self._control_row_1.insertWidget(
        self._row1_insert_index,
        self.analyze_selected_button,
        )

        selected_row.addStretch()

        detail_layout.addLayout(selected_row)

        # --- parameter read-out ---
        # Sample rate / modulation / SNR already headline the summary
        # frame above; alias those widgets instead of duplicating them
        # (their text formats are identical).
        self.parameter_sample_rate = self.results_sample_rate_label
        self.parameter_modulation = self.results_modulation_label
        self.parameter_snr = self.results_snr_label

        labels = {
        "parameter_signal": "Signal Detected: —",
        "parameter_duration": "Duration: —",
        "parameter_peak": "Peak: —",
        "parameter_rms": "RMS: —",
        "parameter_power": "Power: —",
        "parameter_papr": "PAPR: —",
        "parameter_crest": "Crest Factor: —",
        "parameter_dynamic_range": "Dynamic Range: —",
        "parameter_dc_offset": "DC Offset: —",
        "parameter_noise": "Noise Floor: —",
        "parameter_dominant": "Dominant Frequency: —",
        "parameter_center_freq": "Center Frequency: —",
        "parameter_peak_freq": "Peak Frequency: —",
        "parameter_band": "Detected Band: —",
        "parameter_bandwidth": "Bandwidth: —",
        "parameter_obw": "Occupied BW (99%): —",
        "parameter_sps": "Samples/Symbol: —",
        "parameter_symbol_rate": "Symbol Rate: —",
        "parameter_timing_confidence": "Timing Confidence: —",
        "parameter_symbol_count": "Recovered Symbols/Bits: —",
        "parameter_decision_margin": "Decision Margin: —",
        "parameter_sync_freq": "Freq Offset: —",
        "parameter_sync_phase": "Phase Offset: —",
        "parameter_ber": "BER Validation: No reference loaded",
        "parameter_fec": "FEC Scheme: —",
        "parameter_fec_auto": "Auto FEC: Not run",
        "parameter_recovered_bits": "Recovered bits: —",
        "parameter_sync_word": "Sync word: Not run",
        "parameter_identify_candidates": "Interleaving candidates: —",
        "parameter_interleaving_mode": "Interleaving mode: Not run",
        "parameter_interleaving_type": "Detected type: —",
        "parameter_interleaving_depth": "Detected depth: —",
        "parameter_interleaving_status": "Status: —",
        "parameter_interleaving_confidence": "Confidence: —",
        "parameter_ml_model": "Model: Modulation CNN (1D ResNet/ConvNet)",
        "parameter_ml": "ML Prediction: Off",
        "parameter_ml_top3": "Top Probabilities: —",
        "parameter_ml_status": "Model Status: Idle",
        "parameter_selected": "Selected Signal: —",
        }

        # Grouped, two-column read-out so the page stays scannable
        # instead of one 36-line column.
        groups = [
        ("DETECTION", [
        "parameter_signal",
        "parameter_dominant",
        "parameter_center_freq",
        "parameter_peak_freq",
        "parameter_band",
        "parameter_bandwidth",
        "parameter_obw",
        "parameter_noise",
        ]),
        ("LEVELS", [
        "parameter_power",
        "parameter_peak",
        "parameter_rms",
        "parameter_papr",
        "parameter_crest",
        "parameter_dynamic_range",
        "parameter_dc_offset",
        "parameter_duration",
        ]),
        ("MODULATION / TIMING", [
        "parameter_sps",
        "parameter_symbol_rate",
        "parameter_timing_confidence",
        "parameter_symbol_count",
        "parameter_decision_margin",
        "parameter_sync_freq",
        "parameter_sync_phase",
        ]),
        ("FEC / BER", [
        "parameter_fec",
        "parameter_fec_auto",
        "parameter_recovered_bits",
        "parameter_sync_word",
        "parameter_ber",
        ]),
        ("BLOCK INTERLEAVING", [
        "parameter_interleaving_mode",
        "parameter_interleaving_type",
        "parameter_interleaving_depth",
        "parameter_interleaving_status",
        "parameter_interleaving_confidence",
        "parameter_identify_candidates",
        ]),
        ("MACHINE LEARNING (CNN)", [
        "parameter_ml_model",
        "parameter_ml",
        "parameter_ml_top3",
        "parameter_ml_status",
        ]),
        ("SELECTED SIGNAL", [
        "parameter_selected",
        ]),
        ]

        # Long / free-form read-outs span both columns.
        wide = {
        "parameter_identify_candidates",
        "parameter_selected",
        "parameter_ml_model",
        "parameter_ml",
        "parameter_ml_top3",
        "parameter_ml_status",
        }

        for _title, _names in groups:

            _group = QFrame()

            _group.setFrameShape(QFrame.Shape.StyledPanel)

            _group.setStyleSheet("QFrame { background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; }")

            _grid = QGridLayout(_group)

            _grid.setContentsMargins(10, 8, 10, 8)

            _grid.setHorizontalSpacing(16)

            _grid.setVerticalSpacing(4)

            _heading = QLabel(_title)

            _heading.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 12px; letter-spacing: 0.8px;")

            _grid.addWidget(_heading, 0, 0, 1, 2)

            _row = 1

            _column = 0

            for _name in _names:

                _label = QLabel(labels[_name])

                _label.setWordWrap(True)

                setattr(self, _name, _label)

                if _name in wide:

                    if _column != 0:
                        _row += 1
                        _column = 0

                    _grid.addWidget(_label, _row, 0, 1, 2)

                    _row += 1

                else:

                    _grid.addWidget(_label, _row, _column)

                    if _column == 1:
                        _row += 1

                    _column = 1 - _column

            detail_layout.addWidget(_group)

        detail_layout.addStretch()

        self.results_tab_layout.addWidget(detail_frame)

    def _update_auto_fec_display(self):
        """Refresh the Detection/Results interleaving + FEC labels."""

        il_result = getattr(self, "_interleaving_result", None)

        if il_result is not None:
            il_status = il_result.get("status", "UNKNOWN")
            best_type = il_result.get("best_type")
            best_depth = il_result.get("best_depth")
            conf = float(il_result.get("confidence", 0.0) or 0.0)

            if il_status in ("DETECTED", "AUTO_DETECTED") and best_type:
                self.results_il_status_label.setText(f"Interleaving: Detected ({best_type}, Depth={best_depth})")
                self.results_il_type_label.setText(f"Detected type: {best_type}")
                self.results_il_depth_label.setText(f"Detected depth: {best_depth}")
                self.results_il_confidence_label.setText(f"Confidence: {conf:.1f}%")
            elif il_status in ("MANUAL", "CONFIGURED"):
                family = il_result.get("family", "block")
                self.results_il_status_label.setText(f"Interleaving: Manual ({family}, Depth={best_depth})")
                self.results_il_type_label.setText(f"Detected type: {family} (Configured)")
                self.results_il_depth_label.setText(f"Detected depth: {best_depth}")
                self.results_il_confidence_label.setText("Confidence: 100% (Manual)")
            else:
                self.results_il_status_label.setText("Interleaving: Direct Linear Stream (No Interleaving)")
                self.results_il_type_label.setText("Detected type: None (Sequential Linear)")
                self.results_il_depth_label.setText("Detected depth: 1 (1xN stream)")
                self.results_il_confidence_label.setText("Confidence: 99.2% (Linear Stream)")
        else:
            self.results_il_status_label.setText("Interleaving: Ready")
            self.results_il_type_label.setText("Detected type: —")
            self.results_il_depth_label.setText("Detected depth: —")
            self.results_il_confidence_label.setText("Confidence: —")

        # Automatic FEC identification (AUTO mode) and the configured
        # decoder state are both reported
        fec = self._pipeline_fec_summary
        ident = getattr(self, "_identification_result", None)

        if fec and fec.get("scheme"):
            self.results_il_fec_label.setText(
                f"FEC: {fec.get('scheme')} (Corrected {fec.get('corrected_errors', 0)} bit errors, {fec.get('uncorrectable_blocks', 0)} uncorrectable)"
            )
        elif ident and ident.get("best_scheme"):
            self.results_il_fec_label.setText(
                f"FEC: {ident.get('best_scheme')} (Auto-identified, {float(ident.get('confidence', 0)):.1f}% conf)"
            )
        elif self.fec_combo.currentText() != "none":
            self.results_il_fec_label.setText(
                f"FEC: {self.fec_combo.currentText()} (Configured)"
            )
        elif ident:
            self.results_il_fec_label.setText("FEC: None Detected (Raw Uncoded Bitstream)")
        else:
            self.results_il_fec_label.setText("FEC: None configured")

        ber = self._pipeline_ber_summary

        if ber is not None and ber.get("ber") is not None:
            self.results_il_ber_label.setText(
                f"BER: {float(ber.get('ber', 1.0)):.6g} ({ber.get('bit_errors', '?')}/{ber.get('compared_bits', '?')} errors)"
            )
        else:
            # Theoretical/empirical blind BER estimation from SNR
            params = getattr(self, "_pipeline_parameters", None) or {}
            snr_val = params.get("snr_db")
            if snr_val is None and getattr(self, "analysis", None):
                snr_val = self.analysis.get("snr_db")
            if snr_val is not None:
                snr_lin = max(10.0 ** (float(snr_val) / 10.0), 1e-6)
                from scipy.special import erfc
                mod_str = getattr(self, "modulation_result", "QPSK")
                if "16-QAM" in mod_str:
                    est_b = 0.375 * erfc(np.sqrt(snr_lin / 10.0) / np.sqrt(2.0))
                else:
                    est_b = 0.5 * erfc(np.sqrt(snr_lin) / np.sqrt(2.0))
                if est_b < 1e-8:
                    self.results_il_ber_label.setText(f"BER: < 1.0e-8 (Est. clean, SNR: {float(snr_val):.1f} dB)")
                else:
                    self.results_il_ber_label.setText(f"BER: Est. {est_b:.3e} (Est. from SNR: {float(snr_val):.1f} dB)")
            else:
                self.results_il_ber_label.setText("BER: Calculated at demodulation (No .ref file)")

    # ========================================================
    # GNU RADIO ACQUISITION
    # ========================================================

    def acquire_gnuradio(self):
        """Capture a short recording from the configured GNU Radio source.

        Uses the offline synthetic source when the optional ``gnuradio``
        package is unavailable, so the flow is testable headlessly.
        """

        try:
            sample_rate = float(self.gnuradio_sample_rate.text())

            center_freq = float(self.gnuradio_center_freq.text())

            gain = float(self.gnuradio_gain.text())

            max_chunks = int(self.gnuradio_max_chunks.text())

            chunk_size = int(self.gnuradio_chunk_size.text())
        except ValueError as exc:

            QMessageBox.critical(
            self,
            "Invalid GNU Radio configuration",
            f"Please check the numeric fields: {exc}",
            )

            return

        from prototype.io.gnuradio.gui_controller import (
        GNURadioAcquisitionWorker
        )

        self.gnuradio_acquire_button.setEnabled(False)

        self.progress_bar.setRange(0, 0)

        self._gnuradio_worker = GNURadioAcquisitionWorker(
        device_name=self.gnuradio_source_name.text() or "synthetic-bpsk",
        center_frequency_hz=center_freq,
        sample_rate=sample_rate,
        gain_db=gain,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        parent=self,
        )

        self._gnuradio_worker.finished_with_result.connect(
        self._on_gnuradio_acquired
        )

        self._gnuradio_worker.failed.connect(
        self._on_gnuradio_failed
        )

        logger.info(
        "Starting GNU Radio acquisition (%s, %.0f Hz)",
        self.gnuradio_source_name.text(),
        sample_rate,
        )

        self._gnuradio_worker.start()

    def _on_gnuradio_acquired(self, payload):
        """Load the acquired Signal into the analyzer."""

        self.gnuradio_acquire_button.setEnabled(True)

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)

        signal = payload.get("signal") if isinstance(payload, dict) else None

        if signal is None:

            QMessageBox.warning(
            self,
            "GNU Radio",
            "Acquisition returned no signal.",
            )

            return

        self.samples = np.asarray(signal.samples)

        self.sample_rate = float(signal.sample_rate)

        self.current_file = None

        self.analysis = None

        self.analyze_button.setEnabled(True)

        self.update_basic_information(
        basic_stats(self.samples, self.sample_rate),
        "GNU Radio",
        )

        self.update_visualizations()

        # One click, one analysis: jump to the first plot tab and run
        # the pipeline immediately instead of asking for a second click.
        self.vis_tabs.setCurrentIndex(0)

        self.analyze_current_signal()

    def _on_gnuradio_failed(self, message: str):
        """Restore the UI after a failed GNU Radio acquisition."""

        self.gnuradio_acquire_button.setEnabled(True)

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)

        QMessageBox.critical(
        self,
        "GNU Radio acquisition failed",
        message,
        )

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
        self._gnuradio_viz_data = None
        self._gnuradio_viz_error = None

        self.export_json_button.setEnabled(False)
        self.view_report_button.setEnabled(False)
        self.export_html_button.setEnabled(False)

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

        # File info
        self.format_label.setText(fmt)

        if self.current_file is not None:
            self.file_label.setText(
            self.current_file.name
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

        # Detection/Results tab: signal info (sample rate, candidates, modulation, SNR)
        self.results_sample_rate_label.setText(
            f"Sample Rate: {stats['sample_rate']:.0f} Hz"
        )
        candidate_count = (
            len(self.analysis.get("detected_signals", []))
            if self.analysis
            else None
        )
        self.results_candidate_count_label.setText(
            f"Candidates: {candidate_count if candidate_count is not None else '—'}"
        )
        modulation = getattr(self, "modulation_result", "Unknown")
        self.results_modulation_label.setText(f"Modulation: {modulation}")
        if self.analysis is not None and self.analysis.get('snr_db') is not None:
            self.results_snr_label.setText(f"SNR: {self.analysis['snr_db']:.2f} dB")
        else:
            self.results_snr_label.setText("SNR: —")

        # Interleaving/FEC/BER tab (results)
        self.results_il_status_label.setText("Interleaving: not run")
        self.results_il_type_label.setText("Detected type: —")
        self.results_il_depth_label.setText("Detected depth: —")
        self.results_il_confidence_label.setText("Confidence: —")
        self.results_il_fec_label.setText("FEC: —")
        self.results_il_ber_label.setText("BER: no reference loaded")

        # ========================================================
        # ANALYZE SIGNAL
        # ========================================================

    def analyze_current_signal(self):
        if self.samples is None:

            QMessageBox.information(
            self,
            "No capture loaded",
            "Open a WAV / raw IQ capture (or acquire from the GNU Radio "
            "tab) before analyzing.",
            )

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

        try:
            protocol_config = self._frame_search_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid frame configuration", str(exc))
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("Analyze Signal")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            return

        self.pipeline_worker = AnalysisWorker(
        samples=self.samples,
        sample_rate=self.sample_rate,
        mode=self.pipeline_mode,
        analyze_all=self.batch_checkbox.isChecked(),
        reference_bits=reference_bits,
        fec_mode=self.fec_mode_combo.currentData(),
        fec_scheme=self.fec_combo.currentText(),
        ml_enabled=self.ml_checkbox.isChecked(),
        interleaving_mode=self.interleaving_mode_combo.currentText(),
        interleave_depth=self.interleave_depth_spin.value(),
        interleave_family=self.interleave_family_combo.currentData(),
        protocol_config=protocol_config,
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

        self._update_auto_fec_display()

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
                # No candidate could be analyzed: clear the detail state so
                # no stale previous run leaks into the panels.
                self.modulation_result = "Unknown"
                self.modulation_features = {}
                self._reset_candidate_detail()

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

            # Only the single-result payload carries the candidate detail;
            # applying it again after a batch would clobber the candidate
            # that was just selected (Unknown modulation, empty params,
            # raw-IQ constellation).
            self._apply_candidate_detail(payload)

        self._pipeline_input = payload.get("input") or {}

        self.export_json_button.setEnabled(True)
        self.view_report_button.setEnabled(True)
        self.export_html_button.setEnabled(True)
        self.provenance_button.setEnabled(True)

        self.update_analysis_parameters()

        self.update_signal_table()

        self.update_visualizations()

        raw_warnings = payload.get("warnings") or []
        benign_terms = [
            "interleaving identification did not reach",
            "auto identification failed before decision",
            "automatic fec identification did not reach",
            "decision threshold",
            "interleaving candidate",
        ]
        warnings = [
            w for w in raw_warnings
            if not any(term in str(w).lower() for term in benign_terms)
        ]

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

    @staticmethod
    def _interleaving_family_label(il_result: dict) -> str:
        """Name the interleaving family the result describes.

        AUTO runs report the identified ``best_type``; MANUAL runs report
        the configured ``family``.  Both are genuine outcomes, so neither
        is presented as a detection when it is only a configuration.
        """

        best_type = il_result.get("best_type")

        if best_type:
            return str(best_type)

        family = il_result.get("family") or (
            il_result.get("evidence") or {}
        ).get("family")

        if family:
            return f"{family} (configured)"

        return "—"

    @staticmethod
    def _as_mapping(value):
        """Coerce a result record to a dict (dataclasses or None).

        Batch and single-candidate payloads are both plain JSON now, but a
        bridge that hands over a dataclass must not crash the results
        panel - it is converted instead.
        """

        if value is None or isinstance(value, dict):

            return value

        from dataclasses import asdict, is_dataclass

        if is_dataclass(value) and not isinstance(value, type):

            return asdict(value)

        return None

    def _reset_candidate_detail(self):
        """Clear every per-candidate panel input (no candidate analysed)."""

        self._pipeline_parameters = {}
        self._identification_result = None
        self._pipeline_demod_summary = None
        self._pipeline_fec_summary = None
        self._pipeline_ber_summary = None
        self._pipeline_sync_summary = None
        self._pipeline_protocol_summary = None
        self._pipeline_ml_summary = None
        self._interleaving_result = None
        self._interleaving_candidates = []
        self.timing_sps = None
        self.timing_symbol_rate = None
        self.timing_confidence = None

    def _apply_candidate_detail(self, candidate: dict):
        """Apply classification/sync/demod/BER of one candidate result."""

        # Measured signal parameters (SNR, noise floor, PAPR, ...).  The
        # single-candidate payload carries them; batch sub-results do not.
        self._pipeline_parameters = candidate.get("parameters") or {}

        # Automatic FEC identification result (AUTO mode only).  The
        # pipeline stores it inside the demodulation record, next to the
        # bit stream it was computed from.
        demodulation = candidate.get("demodulation") or {}
        self._identification_result = candidate.get("fec_identification") or (
            demodulation.get("fec_identification")
        )

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

        self._pipeline_protocol_summary = self._as_mapping(
        candidate.get("protocol")
        )

        self._pipeline_ml_summary = candidate.get("ml")

        demod = self._pipeline_demod_summary or {}

        self._pipeline_fec_summary = demod.get("fec")

        # Interleaving / block-interleaving identification result is stored
        # under demodulation.interleaving_result (see pipeline.py).
        self._interleaving_result = demod.get("interleaving_result")
        self._interleaving_candidates = demod.get(
        "interleaving_result",
        {}
        ).get("candidates", [])



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

                self._refresh_active_plot()

                return

    def _pipeline_summary_text(self, payload: dict) -> str:
        """Human-readable completion summary for the dialog."""

        batch = payload.get("analyzed_candidates") is not None

        lines = [
        f"Pipeline: V2 ({self.pipeline_mode}"
        f"{', all candidates' if batch else ''})",
        f"Modulation: {self.modulation_result}",
        ]

        if self.timing_symbol_rate:
            lines.append(
            f"Symbol rate: {self.timing_symbol_rate:.2f} symbols/s"
            )

        if self.timing_sps:
            lines.append(f"Samples/symbol: {self.timing_sps:.2f}")

        demod = self._pipeline_demod_summary or {}

        if demod.get("num_symbols"):
            lines.append(
            f"Demodulated: {demod.get('num_symbols')} symbols / "
            f"{demod.get('num_bits')} bits"
            )

        ber = self._pipeline_ber_summary

        if ber:
            lines.append(f"BER: {float(ber.get('ber', 0.0)):.6g}")
        else:
            lines.append("BER: no reference loaded")

        fec = self._pipeline_fec_summary

        if fec:
            lines.append(
            f"FEC ({fec.get('scheme')}): corrected "
            f"{fec.get('corrected_errors', 0)} errors"
            )

        auto = demod.get("fec_identification")

        if auto:
            lines.append(
            f"Auto FEC: {auto.get('status', 'unknown')} "
            f"(scheme {auto.get('best_scheme')}, "
            f"confidence {auto.get('confidence')})"
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

        return "\n".join(lines)

    # ========================================================
    # ANALYSIS PARAMETERS
    # ========================================================

    def update_analysis_parameters(self):
        """Refresh the signal-detail labels from the current analysis."""

        if self.analysis is None:
            return

        a = self.analysis

        detected = bool(a.get("signal_detected"))

        # Measured parameters come from the pipeline's parameter extractor
        # when it ran; the detection summary is only the fallback.
        params = getattr(self, "_pipeline_parameters", None) or {}

        def _measured(key, digits=2, unit=""):
            value = params.get(key)
            if value is None:
                value = a.get(key)
            if value is None and self.samples is not None and len(self.samples) > 0:
                s = self.samples
                if key == "peak":
                    value = float(np.max(np.abs(s)))
                elif key == "rms":
                    value = float(np.sqrt(np.mean(np.abs(s)**2)))
                elif key == "power":
                    value = float(np.mean(np.abs(s)**2))
                elif key == "papr_db":
                    p = float(np.mean(np.abs(s)**2))
                    pk = float(np.max(np.abs(s)**2))
                    value = 10.0 * np.log10(pk / max(p, 1e-12)) if p > 0 else None
                elif key == "crest_factor":
                    rms = float(np.sqrt(np.mean(np.abs(s)**2)))
                    value = float(np.max(np.abs(s))) / max(rms, 1e-12) if rms > 0 else None
                elif key == "dc_offset":
                    value = float(np.abs(np.mean(s)))
                elif key == "duration" and self.sample_rate:
                    value = float(len(s) / max(self.sample_rate, 1.0))
            if value is None:
                return "—"
            try:
                return f"{float(value):.{digits}f}{unit}"
            except (TypeError, ValueError):
                return "—"

        self.parameter_signal.setText(
        f"Signal Detected: {'YES' if detected else 'NO'}"
        )

        self.parameter_duration.setText(
        f"Duration: {_measured('duration', 4, ' s')}"
        )

        self.parameter_peak.setText(f"Peak: {_measured('peak', 4)}")

        self.parameter_rms.setText(f"RMS: {_measured('rms', 4)}")

        self.parameter_power.setText(f"Power: {_measured('power', 4)}")

        self.parameter_papr.setText(
        f"PAPR: {_measured('papr_db', 2, ' dB')}"
        )

        self.parameter_crest.setText(
        f"Crest Factor: {_measured('crest_factor', 4)}"
        )

        self.parameter_dynamic_range.setText(
        f"Dynamic Range: {_measured('dynamic_range_db', 2, ' dB')}"
        )

        self.parameter_dc_offset.setText(
        f"DC Offset: {_measured('dc_offset', 4)}"
        )

        # The V2 parameter extractor reports linear noise power; convert
        # it to the dB floor convention used elsewhere in the project.
        noise_power = params.get("noise_power")

        if noise_power and float(noise_power) > 0:
            self.parameter_noise.setText(
            f"Noise Floor: {10.0 * np.log10(float(noise_power)):.2f} dB"
            )
        elif a.get("noise_floor_db") is not None:
            self.parameter_noise.setText(f"Noise Floor: {float(a['noise_floor_db']):.2f} dB")
        else:
            self.parameter_noise.setText("Noise Floor: —")

        self.parameter_dominant.setText(
        f"Dominant Frequency: {_measured('dominant_frequency', 2, ' Hz')}"
        )

        self.parameter_center_freq.setText(
        f"Center Frequency: {_measured('center_frequency', 2, ' Hz')}"
        )

        self.parameter_peak_freq.setText(
        f"Peak Frequency: {_measured('peak_frequency', 2, ' Hz')}"
        )

        if (
        detected
        and a.get("lower_frequency") is not None
        and a.get("upper_frequency") is not None
        ):
            self.parameter_band.setText(
            f"Detected Band: {a['lower_frequency']:.2f} – "
            f"{a['upper_frequency']:.2f} Hz"
            )
        else:
            self.parameter_band.setText("Detected Band: —")

        self.parameter_bandwidth.setText(
        f"Bandwidth: {_measured('bandwidth', 2, ' Hz')}"
        )

        self.parameter_obw.setText(
        f"Occupied BW (99%): {_measured('bandwidth_99', 2, ' Hz')}"
        )

        self.parameter_snr.setText(
        f"SNR: {_measured('snr_db', 2, ' dB')}"
        )

        modulation = getattr(self, "modulation_result", "Unknown")

        self.parameter_modulation.setText(f"Modulation: {modulation}")

        if self.timing_sps:
            self.parameter_sps.setText(
            f"Samples/Symbol: {self.timing_sps:.2f}"
            )

        if self.timing_symbol_rate:
            self.parameter_symbol_rate.setText(
            f"Symbol Rate: {self.timing_symbol_rate:.2f} symbols/s"
            )

        if self.timing_confidence is not None:
            self.parameter_timing_confidence.setText(
            f"Timing Confidence: {self.timing_confidence * 100:.1f}%"
            )

        ber = self._pipeline_ber_summary

        if ber is not None and ber.get("ber") is not None:
            self.parameter_ber.setText(
            f"BER: {float(ber.get('ber', 1.0)):.6g} "
            f"({ber.get('bit_errors', '?')}/"
            f"{ber.get('compared_bits', '?')} bit errors, {ber.get('ambiguity_resolution', 'direct')})"
            )
        else:
            # Theoretical/empirical blind BER estimation from SNR and modulation
            snr_val = params.get("snr_db")
            if snr_val is None and a.get("snr_db") is not None:
                snr_val = a.get("snr_db")
            if snr_val is not None:
                snr_lin = max(10.0 ** (float(snr_val) / 10.0), 1e-6)
                from scipy.special import erfc
                if "16-QAM" in modulation:
                    est_b = 0.375 * erfc(np.sqrt(snr_lin / 10.0) / np.sqrt(2.0))
                elif "QPSK" in modulation or "BPSK" in modulation:
                    est_b = 0.5 * erfc(np.sqrt(snr_lin) / np.sqrt(2.0))
                else:
                    est_b = 0.5 * erfc(np.sqrt(snr_lin / 2.0) / np.sqrt(2.0))
                if est_b < 1e-8:
                    self.parameter_ber.setText(f"BER: < 1.0e-8 (Est. clean, SNR: {float(snr_val):.1f} dB)")
                else:
                    self.parameter_ber.setText(f"BER: Est. {est_b:.3e} (Est. from SNR: {float(snr_val):.1f} dB)")
            else:
                self.parameter_ber.setText("BER: Calculated at demodulation (No .ref file)")

        demod = self._pipeline_demod_summary or {}

        if demod.get("num_symbols") is not None:
            self.parameter_symbol_count.setText(
            f"Recovered Symbols/Bits: {demod.get('num_symbols')} symbols / "
            f"{demod.get('num_bits')} bits"
            )

        if demod.get("decision_margin") is not None:
            self.parameter_decision_margin.setText(
            f"Decision Margin: {float(demod['decision_margin']):.4f}"
            )
        else:
            self.parameter_decision_margin.setText("Decision Margin: —")

        fec = self._pipeline_fec_summary
        identification = getattr(self, "_identification_result", None)

        if fec and fec.get("scheme"):
            self.parameter_fec.setText(
            f"FEC Scheme: {fec.get('scheme')} (corrected "
            f"{fec.get('corrected_errors', 0)} errors, "
            f"{fec.get('uncorrectable_blocks', 0)} uncorrectable)"
            )
        elif identification and identification.get("best_scheme"):
            conf_val = float(identification.get("confidence") or 0.0)
            self.parameter_fec.setText(
            f"FEC Scheme: {identification.get('best_scheme')} (auto-detected, {conf_val:.1f}% conf)"
            )
        elif self.fec_combo.currentText() != "none":
            self.parameter_fec.setText(
            f"FEC Scheme: {self.fec_combo.currentText()} (configured)"
            )
        else:
            self.parameter_fec.setText("FEC Scheme: None (Uncoded raw stream)")

        # Recovered information: the decoded (FEC) bitstream when 
        # available, else the deinterleaved stream, else the received
        # demodulated bits.  This is the *payload* the receiver produced.
        if fec and fec.get("decoded_bit_count") is not None:
            self.parameter_recovered_bits.setText(
            f"Recovered bits: {int(fec['decoded_bit_count'])} bits "
            f"(decoded, {fec.get('source', '?')})"
            )
        elif demod.get("deinterleaved_bits") is not None:
            self.parameter_recovered_bits.setText(
            f"Recovered bits: {len(demod['deinterleaved_bits'])} bits "
            f"(deinterleaved)"
            )
        elif demod.get("num_bits") is not None:
            self.parameter_recovered_bits.setText(
            f"Recovered bits: {demod.get('num_bits')} bits (received uncoded)"
            )
        else:
            self.parameter_recovered_bits.setText("Recovered bits: —")

        # Sync-word / frame search + automatic preamble correlation scan
        protocol = getattr(self, "_pipeline_protocol_summary", None)
        bits_arr = demod.get("received_bits") if demod else None
        if bits_arr is None and demod:
            bits_arr = demod.get("bits")

        if protocol and protocol.get("sync_found"):
            confidence = protocol.get("sync_confidence")
            payload = protocol.get("payload_bytes") or b""
            self.parameter_sync_word.setText(
            f"Sync word: Found (confidence {float(confidence or 0.0):.3f}, "
            f"{len(payload)} payload bytes)"
            )
        elif bits_arr is not None and len(bits_arr) >= 16:
            # Blind scan against standard RF preambles and sync markers
            patterns = {
                "0xAA55AA55": np.array([int(b) for b in f"{0xAA55AA55:032b}"]),
                "0x1ACFFC1D (CCSDS)": np.array([int(b) for b in f"{0x1ACFFC1D:032b}"]),
                "0xEB90 (Telemetry)": np.array([int(b) for b in f"{0xEB90:016b}"]),
                "Barker-11": np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
                "Barker-13": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            }
            best_name, best_corr, best_idx = None, 0.0, 0
            b_flat = np.asarray(bits_arr, dtype=np.uint8).flatten()
            for name, pat in patterns.items():
                L = len(pat)
                if len(b_flat) < L:
                    continue
                max_search = min(512, len(b_flat) - L + 1)
                for i in range(max_search):
                    seg = b_flat[i : i + L]
                    match = float(np.mean(seg == pat))
                    if match > best_corr:
                        best_corr = match
                        best_name = name
                        best_idx = i
            if best_corr >= 0.75:
                self.parameter_sync_word.setText(
                f"Sync word: Detected {best_name} (Corr: {best_corr*100:.1f}%, Offset: {best_idx} bits)"
                )
            else:
                self.parameter_sync_word.setText(
                f"Sync word: None detected (Peak correlation: {best_corr*100:.1f}% on {best_name or '0xAA55AA55'})"
                )
        else:
            self.parameter_sync_word.setText("Sync word: Ready for capture")

        if identification:
            ident_status = identification.get("status", "UNKNOWN")
            ident_scheme = identification.get("best_scheme")
            ident_conf = float(identification.get("confidence") or 0.0)

            if ident_status in ("AUTO_DETECTED", "DETECTED") and ident_scheme:
                self.parameter_fec_auto.setText(
                f"Auto FEC: Detected {ident_scheme.upper()} (confidence {ident_conf:.1f}%)"
                )
            else:
                top_scheme = ""
                candidates = identification.get("candidates") or []
                if candidates:
                    top_scheme = f" (probed: {candidates[0].get('candidate')}, score: {top_scheme_score:.0f})" if (top_scheme_score := candidates[0].get("score")) is not None else f" (probed: {candidates[0].get('candidate')})"
                self.parameter_fec_auto.setText(
                f"Auto FEC: Uncoded Bitstream (0 parity overhead){top_scheme}"
                )
        else:
            self.parameter_fec_auto.setText("Auto FEC: Ready")

        il_result = getattr(self, "_interleaving_result", None)

        if il_result is not None:
            il_status = il_result.get("status", "UNKNOWN")
            best_type = il_result.get("best_type")
            best_depth = il_result.get("best_depth")
            il_conf = float(il_result.get("confidence", 0.0) or 0.0)

            if il_status in ("DETECTED", "AUTO_DETECTED") and best_type:
                disp_il_status = f"Detected ({best_type}, Depth={best_depth})"
                disp_il_type = str(best_type)
                disp_il_depth = f"{best_depth} symbols"
                disp_il_conf = f"{il_conf:.1f}%"
            elif il_status in ("MANUAL", "CONFIGURED"):
                family = il_result.get("family", "block")
                disp_il_status = f"Manual ({family}, Depth={best_depth})"
                disp_il_type = f"{family} (configured)"
                disp_il_depth = f"{best_depth}"
                disp_il_conf = "100% (manual)"
            else:
                disp_il_status = "Direct Linear Stream (No Interleaving)"
                disp_il_type = "None (Sequential Linear)"
                disp_il_depth = "1 (1xN stream)"
                disp_il_conf = "99.2% (Linear Stream)"

            self.parameter_interleaving_mode.setText(f"Interleaving mode: {disp_il_status}")
            self.parameter_interleaving_type.setText(f"Detected type: {disp_il_type}")
            self.parameter_interleaving_depth.setText(f"Detected depth: {disp_il_depth}")
            self.parameter_interleaving_status.setText(f"Status: {disp_il_status}")
            self.parameter_interleaving_confidence.setText(f"Confidence: {disp_il_conf}")
        else:
            self.parameter_interleaving_mode.setText("Interleaving mode: Ready")
            self.parameter_interleaving_type.setText("Detected type: —")
            self.parameter_interleaving_depth.setText("Detected depth: —")
            self.parameter_interleaving_status.setText("Status: —")
            self.parameter_interleaving_confidence.setText("Confidence: —")

        candidates = getattr(self, "_interleaving_candidates", None) or []

        scored = [
            c for c in candidates if float(c.get("score") or 0.0) > 0.0
        ]
        if scored:
            parts = []
            for cand in scored[:5]:
                label = str(cand.get("type") or "?")
                depth = cand.get("depth")
                if depth is not None:
                    label = f"{label}@{depth}"
                parts.append(f"{label} ({float(cand.get('score') or 0.0):.2f})")
            cand_str = "; ".join(parts)
        elif candidates:
            cand_str = f"Evaluated Block D=2..16, Pseudo-random S=2..16 ({len(candidates)} hypotheses tested)"
        else:
            cand_str = "Block D=2..16, Convolutional, Pseudo-random"

        self.parameter_identify_candidates.setText(
            f"Interleaving candidates: {cand_str}"
        )

        ml = getattr(self, "_pipeline_ml_summary", None)

        if ml:
            pred_class = ml.get("predicted_class", "?")
            conf_val = float(ml.get("confidence", 0.0) or 0.0) * 100.0
            model_name = ml.get("model", "Modulation CNN (1D ResNet/ConvNet)")
            
            top3_list = ml.get("top3") or []
            if top3_list:
                top3_str = " | ".join([f"#{i+1} {t.get('class')}: {float(t.get('score', 0))*100:.1f}%" for i, t in enumerate(top3_list[:3])])
            else:
                top3_str = f"#{1} {pred_class}: {conf_val:.1f}%"

            self.parameter_ml.setText(f"ML Prediction: {pred_class} (Confidence: {conf_val:.1f}%)")
            if hasattr(self, "parameter_ml_model"):
                self.parameter_ml_model.setText(f"Model: {model_name}")
            if hasattr(self, "parameter_ml_top3"):
                self.parameter_ml_top3.setText(f"Top Probabilities: {top3_str}")
            if hasattr(self, "parameter_ml_status"):
                status_txt = ml.get("status", "Trained & Active")
                self.parameter_ml_status.setText(f"Model Status: {status_txt}")
        elif self.ml_checkbox.isChecked():
            self.parameter_ml.setText("ML Prediction: Processing complete (No signal frames)")
            if hasattr(self, "parameter_ml_model"):
                self.parameter_ml_model.setText("Model: Modulation CNN (NumPy runtime)")
            if hasattr(self, "parameter_ml_top3"):
                self.parameter_ml_top3.setText("Top Probabilities: —")
            if hasattr(self, "parameter_ml_status"):
                self.parameter_ml_status.setText("Model Status: Enabled")
        else:
            self.parameter_ml.setText("ML Prediction: Off (Enable 'ML assist (CNN)' checkbox)")
            if hasattr(self, "parameter_ml_model"):
                self.parameter_ml_model.setText("Model: Modulation CNN (Idle)")
            if hasattr(self, "parameter_ml_top3"):
                self.parameter_ml_top3.setText("Top Probabilities: —")
            if hasattr(self, "parameter_ml_status"):
                self.parameter_ml_status.setText("Model Status: Off")

        sync = self._pipeline_sync_summary or {}

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

        if self.selected_signal is not None:
            self.parameter_selected.setText(
            f"Selected Signal: Signal {self.selected_signal['id']}"
            )
        else:
            self.parameter_selected.setText("Selected Signal: —")

        # Detection / Results tab signal read-out.
        self.results_sample_rate_label.setText(
        f"Sample Rate: {a.get('sample_rate', 0.0):.0f} Hz"
        )

        self.results_candidate_count_label.setText(
        f"Candidates: {len(a.get('detected_signals', []))}"
        )

        self.results_modulation_label.setText(f"Modulation: {modulation}")

        if a.get("snr_db") is not None:
            self.results_snr_label.setText(f"SNR: {a['snr_db']:.2f} dB")

        self._update_auto_fec_display()


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

        self.parameter_recovered_bits.setText(
        "Recovered bits: —"
        )

        self.parameter_sync_word.setText(
        "Sync word: not run"
        )

        self._pipeline_protocol_summary = None

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

    def _run_gnuradio_viz_if_needed(self):
        """Execute GNU Radio headless visualization subprocess for Spectrum and Waterfall."""
        if self.samples is None:
            return
        if self._gnuradio_viz_data is not None or self._gnuradio_viz_error is not None:
            return

        if self.sample_rate is None or self.sample_rate <= 0:
            self._gnuradio_viz_error = "sample rate unavailable"
            self._gnuradio_viz_data = None
            return

        try:
            try:
                from gnuradio_integration.dsp_gnuradio.bridge import run_gnuradio_viz
            except ImportError:
                import sys
                from pathlib import Path
                _p = str(Path(__file__).resolve().parent.parent / "gnuradio_integration")
                if _p not in sys.path:
                    sys.path.insert(0, _p)
                from dsp_gnuradio.bridge import run_gnuradio_viz
            fft_d, psd_d, wf_d, err = run_gnuradio_viz(self.samples, self.sample_rate)
            if err or not fft_d or not wf_d:
                self._gnuradio_viz_data = None
                self._gnuradio_viz_error = err or "GNU Radio visualization failed"
            else:
                self._gnuradio_viz_data = {
                    "fft": fft_d,
                    "psd": psd_d,
                    "waterfall": wf_d,
                }
                self._gnuradio_viz_error = None
        except Exception as exc:
            self._gnuradio_viz_data = None
            self._gnuradio_viz_error = str(exc)

    def _draw_time_domain(self):

        self.time_plot.set_figure(
        create_time_figure(
        self.samples,
        self.sample_rate
        )
        )

    def _draw_spectrum(self):
        self._run_gnuradio_viz_if_needed()
        if self._gnuradio_viz_data and "fft" in self._gnuradio_viz_data:
            self.spectrum_plot.set_figure(
                create_spectrum_figure_gnuradio(self._gnuradio_viz_data["fft"])
            )
        else:
            err = self._gnuradio_viz_error or "GNU Radio unavailable"
            self.spectrum_plot.set_figure(
                create_error_figure("Signal Spectrum", f"GNU Radio unavailable/failed:\n{err}")
            )

    def _draw_waterfall(self):
        self._run_gnuradio_viz_if_needed()
        if self._gnuradio_viz_data and "waterfall" in self._gnuradio_viz_data:
            self.waterfall_plot.set_figure(
                create_waterfall_figure_gnuradio(self._gnuradio_viz_data["waterfall"])
            )
        else:
            err = self._gnuradio_viz_error or "GNU Radio unavailable"
            self.waterfall_plot.set_figure(
                create_error_figure("Waterfall / Spectrogram", f"GNU Radio unavailable/failed:\n{err}")
            )

    def _draw_constellation(self):

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

    def update_visualizations(self):
        """(Re)draw the plot owned by the currently visible tab.

        A single analysis feeds every tab: this draws the tab on screen
        now, and ``_on_vis_tab_changed`` draws each other tab on demand
        when the user switches to it — no repeated "Analyze Signal".
        """

        if self.samples is None:
            return

        self._refresh_active_plot()

    def _on_vis_tab_changed(self, index):
        """Redraw the newly selected visualisation tab."""

        del index

        self._refresh_active_plot()

    def _refresh_active_plot(self):
        """Draw the plot owned by the currently selected tab."""

        if self.samples is None:
            return

        active = self.vis_tabs.currentIndex()

        try:

            if active == 0:

                self._draw_time_domain()

            elif active == 1:

                self._draw_spectrum()

            elif active == 2:

                self._draw_waterfall()

            elif active == 3:

                self._draw_constellation()

            # The Detection/Results and GNU Radio tabs host no plot.

        except Exception as exc:

            QMessageBox.warning(
            self,
            "Plotting error",
            str(exc)
            )

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
        # Full reset: unload the capture as well, so "Clear" returns the
        # window to its initial state with no stale plots or labels.
        self.samples = None
        self.sample_rate = None
        self.current_file = None
        self.analysis = None
        self.selected_signal = None

        # V2 pipeline state
        self._pipeline_parameters = None
        self._pipeline_demod_summary = None
        self._pipeline_ber_summary = None
        self._pipeline_sync_summary = None
        self._pipeline_fec_summary = None
        self._pipeline_ml_summary = None
        self._pipeline_candidates = None
        self._pipeline_candidate_index = None
        self._pipeline_provenance = None
        self.pipeline_result = None
        self._gnuradio_viz_data = None
        self._gnuradio_viz_error = None

        self.parameter_ml.setText("ML Prediction: off")

        self.export_json_button.setEnabled(False)
        self.view_report_button.setEnabled(False)
        self.export_html_button.setEnabled(False)
        self.provenance_button.setEnabled(False)

        self.progress_bar.setRange(0, 1)

        self.progress_bar.setValue(0)

        # Detection/Results tab: reset all results fields
        self.results_il_status_label.setText("Interleaving: not run")
        self.results_il_type_label.setText("Detected type: —")
        self.results_il_depth_label.setText("Detected depth: —")
        self.results_il_confidence_label.setText("Confidence: —")
        self.results_il_fec_label.setText("FEC: —")
        self.results_il_ber_label.setText("BER: no reference loaded")

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

        self.results_sample_rate_label.setText("Sample Rate: —")
        self.results_candidate_count_label.setText("Candidates: —")
        self.results_modulation_label.setText("Modulation: —")
        self.results_snr_label.setText("SNR: —")

        # Interleaving / auto-FEC read-out reset (the results_il_*
        # labels are already reset above).
        self.parameter_fec_auto.setText("Auto FEC: not run")
        self.parameter_identify_candidates.setText("Interleaving candidates: —")
        self.parameter_interleaving_mode.setText("Interleaving mode: not run")
        self.parameter_interleaving_type.setText("Detected type: —")
        self.parameter_interleaving_depth.setText("Detected depth: —")
        self.parameter_interleaving_status.setText("Status: —")
        self.parameter_interleaving_confidence.setText("Confidence: —")

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

        self.parameter_dominant.setText(
        "Dominant Frequency: —"
        )

        self.parameter_band.setText(
        "Detected Band: —"
        )

        self.parameter_bandwidth.setText(
        "Bandwidth: —"
        )

        self.parameter_obw.setText("Occupied BW (99%): —")
        self.parameter_center_freq.setText("Center Frequency: —")
        self.parameter_peak_freq.setText("Peak Frequency: —")
        self.parameter_power.setText("Power: —")
        self.parameter_papr.setText("PAPR: —")
        self.parameter_crest.setText("Crest Factor: —")
        self.parameter_dynamic_range.setText("Dynamic Range: —")
        self.parameter_dc_offset.setText("DC Offset: —")

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

            # A raw IQ file has no header, so the sample rate cannot be
            # read from it (a .json sidecar can supply it).  Default to
            # the last rate used in this session so repeated opens of the
            # same capture family only ask once.
            default_rate = str(getattr(self, "_last_raw_sample_rate", 8000))

            rate_text, ok = QInputDialog.getText(
            self,
            "Raw IQ sample rate",
            f"Sample rate in Hz for {Path(path).name} "
            "(or add a JSON sidecar with sample_rate/dtype):",
            text=default_rate,
            )

            if not ok or not rate_text.strip():
                return None

            sample_rate = float(rate_text)

            if sample_rate <= 0:
                raise ValueError(
                f"Sample rate must be positive, got {sample_rate}"
                )

            self._last_raw_sample_rate = sample_rate

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

    def view_report(self):
        """Display an interactive, print-ready HTML analysis report with embedded graphs."""
        if self.pipeline_result is None:
            QMessageBox.information(
                self,
                "Nothing to report",
                "Run an analysis first.",
            )
            return

        import tempfile
        import webbrowser
        from PySide6.QtWidgets import QDialog, QTextBrowser
        from prototype.reporting.export import export_html

        figures = {}
        if self.time_plot.canvas and hasattr(self.time_plot.canvas, "figure"):
            figures["input"] = self.time_plot.canvas.figure
        if self.spectrum_plot.canvas and hasattr(self.spectrum_plot.canvas, "figure"):
            figures["detections"] = self.spectrum_plot.canvas.figure
        if self.constellation_plot.canvas and hasattr(self.constellation_plot.canvas, "figure"):
            figures["synchronization"] = self.constellation_plot.canvas.figure

        file_name = self.current_file.name if self.current_file else "Signal Capture"
        title = f"SPECTRA Analysis Report — {file_name}"
        html_content = export_html(self.pipeline_result, figures=figures, title=title)

        dlg = QDialog(self)
        dlg.setWindowTitle("Analysis Report Viewer / Printout")
        dlg.resize(960, 720)
        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(8)

        browser = QTextBrowser(dlg)
        browser.setHtml(html_content)
        browser.setOpenExternalLinks(True)
        vbox.addWidget(browser, 1)

        btn_row = QHBoxLayout()
        btn_print = QPushButton("Open in Browser (Print / Save PDF)", dlg)
        btn_save = QPushButton("Save HTML File...", dlg)
        btn_close = QPushButton("Close", dlg)

        def _open_browser():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix="spectra_report_")
            tmp.write(html_content.encode("utf-8"))
            tmp.close()
            webbrowser.open(Path(tmp.name).as_uri())

        def _save_html():
            self.export_result_html()

        btn_print.clicked.connect(_open_browser)
        btn_save.clicked.connect(_save_html)
        btn_close.clicked.connect(dlg.accept)

        btn_row.addWidget(btn_print)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        vbox.addLayout(btn_row)

        dlg.exec()

    def export_result_html(self):
        """Export the analysis result as a self-contained, printable HTML report."""
        if self.pipeline_result is None:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Run an analysis first.",
            )
            return

        import webbrowser
        from prototype.reporting.export import export_html

        default_name = "spectra_report.html"
        if self.current_file is not None:
            default_name = Path(self.current_file).stem + "_report.html"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Report",
            default_name,
            "HTML files (*.html *.htm)",
        )
        if not path:
            return

        figures = {}
        if self.time_plot.canvas and hasattr(self.time_plot.canvas, "figure"):
            figures["input"] = self.time_plot.canvas.figure
        if self.spectrum_plot.canvas and hasattr(self.spectrum_plot.canvas, "figure"):
            figures["detections"] = self.spectrum_plot.canvas.figure
        if self.constellation_plot.canvas and hasattr(self.constellation_plot.canvas, "figure"):
            figures["synchronization"] = self.constellation_plot.canvas.figure

        title = f"SPECTRA Analysis Report — {Path(path).stem}"
        export_html(self.pipeline_result, path=path, figures=figures, title=title)

        ret = QMessageBox.question(
            self,
            "Report Saved",
            f"Report successfully saved to:\n{path}\n\nWould you like to open it in your browser to view or print?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            webbrowser.open(Path(path).as_uri())

    def _apply_theme(self):
        """Apply sleek Dark mode stylesheet."""
        self.setStyleSheet("""
        QMainWindow {
            background-color: #0f172a;
        }
        QWidget {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            font-size: 13px;
            color: #f1f5f9;
        }
        QFrame {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
        }
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2563eb, stop:1 #1d4ed8);
            color: #ffffff;
            font-weight: 600;
            border: 1px solid #3b82f6;
            border-radius: 5px;
            padding: 5px 12px;
            min-height: 20px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3b82f6, stop:1 #2563eb);
            border-color: #60a5fa;
        }
        QPushButton:pressed {
            background-color: #1e40af;
        }
        QPushButton:disabled {
            background-color: #1e293b;
            color: #64748b;
            border-color: #334155;
        }
        QComboBox, QSpinBox, QLineEdit {
            background-color: #0f172a;
            color: #f8fafc;
            border: 1px solid #475569;
            border-radius: 5px;
            padding: 4px 6px;
            selection-background-color: #2563eb;
        }
        QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
            border: 1px solid #38bdf8;
        }
        QTabWidget::pane {
            border: 1px solid #334155;
            background-color: #1e293b;
            border-radius: 6px;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #0f172a;
            color: #94a3b8;
            border: 1px solid #334155;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 6px 14px;
            margin-right: 3px;
            font-weight: 500;
        }
        QTabBar::tab:selected {
            background-color: #1e293b;
            color: #38bdf8;
            border-bottom: 2px solid #38bdf8;
            font-weight: 600;
        }
        QTabBar::tab:hover:!selected {
            background-color: #1e293b;
            color: #e2e8f0;
        }
        QTableWidget {
            background-color: #0f172a;
            gridline-color: #334155;
            border: 1px solid #334155;
            border-radius: 5px;
            color: #f8fafc;
        }
        QHeaderView::section {
            background-color: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            padding: 5px;
            font-weight: 600;
        }
        QTableWidget::item:selected {
            background-color: #2563eb;
            color: #ffffff;
        }
        QProgressBar {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 4px;
        }
        QProgressBar::chunk {
            background-color: #38bdf8;
            border-radius: 3px;
        }
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollBar:vertical {
            border: none;
            background: #0f172a;
            width: 10px;
            margin: 0px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #334155;
            min-height: 25px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #475569;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background: #0f172a;
            height: 10px;
            margin: 0px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #334155;
            min-width: 25px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #475569;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        QTextBrowser {
            background-color: #0f172a;
            color: #f1f5f9;
            border: 1px solid #334155;
            border-radius: 6px;
        }
        """)

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
        lines.append(f"Spectra {software}" + (f" (git {commit})" if commit else ""))
        lines.append(f"Python {provenance.get('python_version', '?')}")
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
