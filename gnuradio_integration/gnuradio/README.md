# SPECTRA — GNU Radio Flowgraphs & Bridge

This directory contains the GNU Radio flowgraphs and headless bridge scripts used by SPECTRA for interactive signal inspection, headless BPSK/QPSK demodulation, and spectral visualization.

GNU Radio execution is run **exclusively in a separate subprocess** via `backend/gnuradio/bridge.py` using `SPECTRA_GNURADIO_PYTHON` (defaulting to `C:\Users\karee\radioconda\python.exe` on this system). `gnuradio` is never imported directly into the backend FastAPI server process.

---

## Folder Structure

```
gnuradio/
├── spectra_dsp.grc                  # Interactive DSP & visualization flowgraph (GRC design)
├── spectra_dsp.py                   # Interactive DSP flowgraph (PyQt5 GUI)
├── spectra_demod.grc                # Interactive QPSK demodulation flowgraph (GRC design)
├── spectra_demod.py                 # Interactive QPSK demodulation flowgraph (PyQt5 GUI)
├── qpsk_mapping.json                # Experimentally verified QPSK symbol-to-bit mapping
├── headless/
│   ├── spectra_demod_headless.py    # Headless BPSK/QPSK symbol demodulator
│   ├── spectra_viz_headless.py      # Headless FFT / PSD / Waterfall matrix generator
│   └── verify_qpsk_mapping.py       # Script that verifies constellation decoder mapping
└── README.md                        # Documentation
```

---

## 1. Flowgraph Architectures

### A. Headless Demodulation (`headless/spectra_demod_headless.py`)
- **Pipeline**: Complex64 File Source ─► Symbol Sync (Gardner) ─► Costas Loop (Order 4 for QPSK, 2 for BPSK) ─► Constellation Decoder ─► File Sink (raw `uint8` symbols).
- **Symbol File Contract**: The output file contains raw unsigned bytes (`uint8`), exactly **one byte per decoded symbol** without headers or padding. Downstream components map symbols to bitstrings using `qpsk_mapping.json`.
- **Phase Ambiguity**: Costas loops exhibit phase ambiguity (4-fold for QPSK, 2-fold for BPSK). Absolute bit polarity is subject to rotation unless resolved by higher-layer framing or preambles. This is explicitly documented in the demodulation result (`phase_ambiguity: true`).

### B. Headless Visualization (`headless/spectra_viz_headless.py`)
- **Pipeline**: Complex64 File Source ─► Stream to Vector (FFT size) ─► Forward FFT (Blackman-Harris window, shifted) ─► Complex to Mag Squared ─► Vector File Sink.
- Computes power spectral density and STFT waterfall matrices with native GNU Radio blocks.

### C. Interactive Qt GUI Flowgraphs (`spectra_dsp.py`, `spectra_demod.py`)
- Designed for standalone interactive analysis in GNU Radio Companion / PyQt5 desktop windows.

---

## 2. Python Bridge & Truth in Provenance

- Demodulation and visualization results carry explicit `backend_used` and `source` labels:
  - `backend_used: "gnuradio"` / `source: "GNU Radio"` is returned **only** when a GNU Radio flowgraph successfully executes via subprocess and produces the output data.
  - When GNU Radio is unavailable or if parameter estimation (such as symbol rate / SPS) is undetermined, the system falls back cleanly to NumPy DSP and truthfully reports `backend_used: "numpy"` / `source: "NumPy"`.

---

## 3. Verified GNU Radio Environment

- **GNU Radio Version**: 3.10.12.0 (radioconda)
- **Environment Path**: `C:\Users\karee\radioconda\python.exe`
- **Verified Capabilities**:
  - `verify_qpsk_mapping.py` verified symbol mappings: `0 -> "00"`, `1 -> "10"`, `2 -> "01"`, `3 -> "11"`.
  - Headless QPSK & BPSK demodulation with Gardner symbol timing recovery and Costas carrier synchronization.
  - Spectral peak frequency accuracy verified against known synthetic test tones (< 20 Hz error across a 100 kHz span, within FFT bin resolution).