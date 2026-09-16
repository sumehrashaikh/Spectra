import numpy as np
from pathlib import Path

from prototype.core.timing import recover_symbol_timing, sample_symbols
from prototype.core.bpsk import demodulate_bpsk
from prototype.core.ber import load_transmitted_bits, validate_bpsk_bits

from prototype.modulation.demodulator import (
demodulate_qpsk,
qpsk_decision,
demodulate_qam16,
qam16_decision,
demodulate_bfsk,
)

# V1 synthetic BFSK test profile.
BFSK_FREQ_0 = 500.0
BFSK_FREQ_1 = 700.0
BFSK_SYMBOL_RATE = 100.0

from PySide6.QtCore import Qt
from prototype.core.selected_analyzer import (
analyze_selected_signal as run_selected_analysis
)

from prototype.modulation.classifier import (
classify_modulation
)
from prototype.core.isolator import isolate_signal
from PySide6.QtWidgets import (
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
)


from matplotlib.backends.backend_qtagg import (
FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure

from prototype.core.analyzer import (
analyze_signal,
basic_stats,
)

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

        self.modulation_result = "Unknown"
        self.modulation_features = {}

        # Full-recording analysis
        self.analysis = None

        # Currently selected signal
        self.selected_signal = None
        self.selected_analysis = None

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

        button_layout.addWidget(
        self.clear_button
        )

        button_layout.addStretch()

        main_layout.addLayout(
        button_layout
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
        # OPEN WAV
        # ========================================================

    def open_wav(self):

        path, _ = QFileDialog.getOpenFileName(
        self,
        "Open WAV File",
        "",
        "WAV Files (*.wav);;All Files (*)"
        )

        if not path:
            return

        try:

            samples, sample_rate = load_wav(
            path
            )

            stats = basic_stats(
            samples,
            sample_rate
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
        self.selected_analysis = None
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
        self.bfsk_demodulation = None
        self.bfsk_ber_validation = None
        self.parameter_ber.setText(
        "BER Validation: No reference loaded"
        )

        self.signal_table.setRowCount(
        0
        )

        self.clear_selected_signal_display()

        self.update_basic_information(
        stats
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
    stats
    ):

        self.file_label.setText(
        self.current_file.name
        )

        self.format_label.setText(
        "WAV"
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

        try:

            self.analysis = analyze_signal(
            self.samples,
            self.sample_rate
            )

        except Exception as exc:
            QMessageBox.critical(
            self,
            "Analysis failed",
            str(exc)
            )

            return

            # --------------------------------------------------------
            # Modulation classification
            # --------------------------------------------------------

        try:

            modulation, modulation_features = (
            classify_modulation(
            self.samples,
            self.sample_rate
            )
            )


            self.modulation_result = modulation
            self.modulation_features = (
            modulation_features
            )

        except Exception as exc:
            self.modulation_result = "Unknown"
            self.modulation_features = {}

            print(
            "Modulation classification error:",
            exc
            )

            # --------------------------------------------------------
            # Update GUI
            # --------------------------------------------------------

        self.update_analysis_parameters()

        self.update_signal_table()

        self.update_visualizations()

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
        # Print to terminal for debugging
        # ----------------------------------------------------

        print(
        "\nSelected signal:"
        )

        print(
        f"Signal {signal_id}"
        )

        print(
        f"Frequency: "
        f"{frequency:.2f} Hz"
        )

        print(
        f"Bandwidth: "
        f"{bandwidth:.2f} Hz"
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
            isolated, filter_info = (
            isolate_signal(
            self.samples,
            self.sample_rate,
            self.selected_signal[
            "frequency"
            ],
            self.selected_signal[
            "bandwidth"
            ]
            )
            )

        except Exception as exc:
            QMessageBox.critical(
            self,
            "Isolation failed",
            str(exc)
            )
            return

            # Store isolated signal
        self.isolated_signal = isolated

        self.analyze_selected_button.setEnabled(
        True
        )

        self.isolated_filter_info = filter_info

        print("\nIsolated signal")
        print("----------------")
        print(
        f"Center frequency: "
        f"{filter_info['center_frequency']:.2f} Hz"
        )

        print(
        f"Filter range: "
        f"{filter_info['filter_low']:.2f} - "
        f"{filter_info['filter_high']:.2f} Hz"
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

                print()
                print("Selected Signal Analysis")
                print("------------------------")
                print(
                    f"Dominant frequency: "
                    f"{a['dominant_frequency']:.2f} Hz"
                )

                if a["bandwidth"] is not None:
                    print(
                        f"Bandwidth: "
                        f"{a['bandwidth']:.2f} Hz"
                    )

                print(
                    f"Modulation: "
                    f"{self.selected_modulation}"
                )

                print(
                    "Digital demodulation: skipped "
                    "(unsupported/unknown modulation)"
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
            # BPSK/QPSK/16-QAM continue using the generic estimator.
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
            # ----------------------------------------------------

            elif self.selected_modulation == "16-QAM":

                self.qam16_demodulation = demodulate_qam16(
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
                            self.qam16_demodulation[
                                "symbols"
                            ]
                        )

                        best_result = None

                        # Try all four common carrier phase
                        # rotations.
                        for rotation_index in range(4):

                            rotation = (
                                rotation_index
                                * np.pi
                                / 2.0
                            )

                            rotated_symbols = (
                                recovered_symbols
                                * np.exp(
                                    -1j * rotation
                                )
                            )

                            candidate_bits, _, _ = (
                                qam16_decision(
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
                                        rotation_index * 90,
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

            print()
            print("Selected Signal Analysis")
            print("------------------------")

            print(
                f"Dominant frequency: "
                f"{a['dominant_frequency']:.2f} Hz"
            )

            if a["bandwidth"] is not None:
                print(
                    f"Bandwidth: "
                    f"{a['bandwidth']:.2f} Hz"
                )

            print(
                f"Modulation: "
                f"{self.selected_modulation}"
            )

            print(
                f"Amplitude CV: "
                f"{self.selected_modulation_features['amplitude_cv']:.3f}"
            )

            print(
                f"R2 phase coherence: "
                f"{self.selected_modulation_features['r2_phase_coherence']:.3f}"
            )

            print(
                f"R4 phase coherence: "
                f"{self.selected_modulation_features['r4_phase_coherence']:.3f}"
            )

            print(
                f"Instantaneous-frequency std: "
                f"{self.selected_modulation_features['instantaneous_frequency_std_hz']:.2f} Hz"
            )

            print(
                f"Estimated samples/symbol: "
                f"{self.timing_sps:.2f}"
            )

            print(
                f"Estimated symbol rate: "
                f"{self.timing_symbol_rate:.2f} symbols/s"
            )

            print(
                f"Timing confidence: "
                f"{self.timing_confidence * 100:.1f}%"
            )

            if timing is not None:
                print(
                    f"Timing offset: "
                    f"{self.timing_offset} samples "
                    f"(eye opening: "
                    f"{timing.eye_opening:.2f})"
                )
            else:
                print(
                    f"Timing offset: "
                    f"{self.timing_offset} samples "
                    f"(V1 BFSK timing profile)"
                )

            # ----------------------------------------------------
            # BPSK terminal output
            # ----------------------------------------------------

            if self.selected_modulation == "BPSK":

                print(
                    f"Recovered symbols: "
                    f"{len(self.symbol_samples)}"
                )

                print(
                    f"BPSK hard-decision bits: "
                    f"{len(self.bpsk_demodulation.bits)}"
                )

                print(
                    f"BPSK decision margin: "
                    f"{self.bpsk_demodulation.decision_margin:.3f}"
                )

                print(
                    f"BPSK quadrature ratio: "
                    f"{self.bpsk_demodulation.quadrature_ratio:.3f}"
                )

                if self.ber_validation is None:

                    print(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.ber_validation

                    print(
                        f"BER reference: "
                        f"{validation.reference_path}"
                    )

                    print(
                        "BER compared bits: "
                        f"{validation.compared_bit_count} "
                        f"(recovered "
                        f"{validation.recovered_bit_count}, "
                        f"reference "
                        f"{validation.reference_bit_count})"
                    )

                    print(
                        "BER direct/inverted errors: "
                        f"{validation.direct_bit_errors} / "
                        f"{validation.inverted_bit_errors}"
                    )

                    print(
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

                print(
                    f"Recovered symbols: "
                    f"{self.qpsk_demodulation['num_symbols']}"
                )

                print(
                    f"QPSK recovered bits: "
                    f"{self.qpsk_demodulation['num_bits']}"
                )

                print(
                    f"QPSK phase estimate: "
                    f"{self.qpsk_demodulation['phase_estimate_rad']:.4f} rad"
                )

                print(
                    f"QPSK decision margin: "
                    f"{self.qpsk_demodulation['decision_margin']:.3f}"
                )

                print(
                    f"QPSK quadrature balance: "
                    f"{self.qpsk_demodulation['quadrature_balance']:.3f}"
                )

                if self.qpsk_ber_validation is None:

                    print(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.qpsk_ber_validation

                    print(
                        f"BER compared bits: "
                        f"{validation['compared_bits']} "
                        f"(recovered "
                        f"{validation['recovered_bits']}, "
                        f"reference "
                        f"{validation['reference_bits']})"
                    )

                    print(
                        f"QPSK phase rotation selected: "
                        f"{validation['rotation_degrees']}°"
                    )

                    print(
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

                print(
                    f"Recovered symbols: "
                    f"{self.qam16_demodulation['num_symbols']}"
                )

                print(
                    f"16-QAM recovered bits: "
                    f"{self.qam16_demodulation['num_bits']}"
                )

                print(
                    f"16-QAM decision margin: "
                    f"{self.qam16_demodulation['decision_margin']:.3f}"
                )

                print(
                    f"16-QAM quadrature balance: "
                    f"{self.qam16_demodulation['quadrature_balance']:.3f}"
                )

                if self.qam16_ber_validation is None:

                    print(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.qam16_ber_validation

                    print(
                        f"BER compared bits: "
                        f"{validation['compared_bits']} "
                        f"(recovered "
                        f"{validation['recovered_bits']}, "
                        f"reference "
                        f"{validation['reference_bits']})"
                    )

                    print(
                        f"16-QAM phase rotation selected: "
                        f"{validation['rotation_degrees']}°"
                    )

                    print(
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

                print(
                    f"Recovered symbols: "
                    f"{self.bfsk_demodulation['num_symbols']}"
                )

                print(
                    f"BFSK recovered bits: "
                    f"{len(self.bfsk_demodulation['bits'])}"
                )

                print(
                    f"BFSK frequencies: "
                    f"{BFSK_FREQ_0:.1f} Hz / "
                    f"{BFSK_FREQ_1:.1f} Hz"
                )

                print(
                    f"BFSK decision margin: "
                    f"{self.bfsk_demodulation['decision_margin']:.3f}"
                )

                if self.bfsk_ber_validation is None:

                    print(
                        "BER validation: "
                        "no companion reference bits found."
                    )

                else:

                    validation = self.bfsk_ber_validation

                    print(
                        f"BER reference: "
                        f"{validation['reference_path']}"
                    )

                    print(
                        "BER compared bits: "
                        f"{validation['compared_bits']} "
                        f"(recovered "
                        f"{validation['recovered_bits']}, "
                        f"reference "
                        f"{validation['reference_bits']})"
                    )

                    print(
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

                print(
                    f"Recovered symbols: "
                    f"{len(self.symbol_samples)}"
                )

                print(
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
