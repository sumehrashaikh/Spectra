# Standalone GNU Radio Integration Package

This package allows any external Python or Web application to easily execute GNU Radio signal processing (clock recovery, Costas loop demodulation, constellation decoding, and spectral waterfall analysis) with automatic environment detection and NumPy fallback.

## Folder Structure
```text
├── gnuradio/                          # Standalone flowgraphs & headless scripts
│   ├── headless/
│   │   ├── spectra_demod_headless.py  # Headless demodulation & clock recovery
│   │   ├── spectra_viz_headless.py    # Headless PSD & STFT waterfall generation
│   │   └── verify_qpsk_mapping.py     # Constellation self-test script
│   ├── spectra_demod.grc              # GRC Flowgraph (editable in GNU Radio Companion)
│   └── spectra_flowgraph.grc
│
├── dsp_gnuradio/                      # Python Bridge & Orchestrator
│   ├── __init__.py
│   ├── bridge.py                      # Subprocess runner, auto-locator & IPC
│   └── flowgraph.py                   # GNU Radio top_block wrappers
│
├── example_usage.py                   # Ready-to-run quick start test
└── README.md
```

## Quick Start
1. Place `gnuradio` and `dsp_gnuradio` into your project root.
2. Run `python example_usage.py` in your terminal.
3. The bridge will automatically locate Radioconda, GNU Radio 3.10+, or system Python.
