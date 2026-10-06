# SPECTRA — RF / Signal Analysis Platform

Spectra loads IQ/WAV captures, detects signals, isolates candidates,
extracts parameters, estimates symbol rate, synchronizes carrier and
timing, classifies modulation, demodulates, de-interleaves and FEC-decodes
(when configured), runs an explicit frame/sync-word layer, and measures
BER against a reference — from the Python API, the command line, or the
desktop GUI.

**Scope and validation status (read this first).** Spectra is an
engineering prototype validated primarily on *synthetic* signals. It is
**not** certified, flight-qualified, mission-qualified, or approved for
operational deployment by any organization. Benchmark numbers in this
repository characterize the processing chain on simulated channels; they
are not real-world or hardware validation. The repository also ships an
*optional* external real-world validation harness (third-party dataset,
downloaded locally and never committed) and a public-benchmark evidence
pack under `prototype/docs/evidence/` — neither turns the tool into a
validated receiver. See `docs/VALIDATION.md` for exactly what is and is
not claimed.

## Quick start

```bash
# from the repository root (package folder: prototype/)
pip install -e .[dev]

# CLI self-check (synthetic QPSK through the full pipeline)
spectra validate

# Analyze a capture (WAV or raw IQ)
spectra analyze path/to/capture.wav
spectra analyze capture.c64 --sample-rate 8000 --dtype complex64

# Detection / classification / parameters only
spectra detect capture.wav
spectra classify capture.wav --synchronized
spectra parameters capture.wav

# Demodulation + BER against a transmitted-bit reference (NPZ: bits=...)
spectra demodulate capture.wav --reference capture.reference.npz

# FEC / de-interleaving (explicit configuration, never guessed)
spectra analyze capture.wav --fec-mode manual --fec-scheme conv12 \
    --interleaving-mode manual --interleave-family block --interleave-depth 8

# Frame / sync-word search
spectra analyze capture.wav --sync-word 0xAA55AA55 --data-bytes 4

# HTML + JSON report
spectra report capture.wav --html report.html --json results.json

# SNR benchmark (synthetic AWGN channel)
spectra benchmark --modulation QPSK --snr-db 30,20,10,0

# Desktop GUI
spectra gui                  # or: python -m prototype.main

# ML tooling (optional; runtime inference is NumPy-only)
spectra ml-eval              # independent hold-out + NumPy/PyTorch parity
spectra ml-train             # NumPy trainer (gradient-checked)

# Optional external real-world validation harness (needs h5py + local data)
spectra dataset-inspect      # shapes, labels, balance of the dataset
spectra dataset-eval         # run the existing chain on a sampled subset
```

Every analytic command accepts `--json <path>` for machine-readable
output and `--mode quick|balanced|deep|realtime` for processing depth.

## Feature overview

- **Signal chain** — preprocessing, FFT candidate detection, DDC
  isolation, parameter extraction, symbol-rate estimation, carrier +
  timing synchronization, classification, demodulation, BER.
- **Modulations** — BPSK, QPSK, 8-PSK, 16-QAM, BFSK, OOK/ASK; anything
  the evidence does not support is a first-class `Unknown`.
- **FEC** — six registered schemes (`hamming74`, `repetition3`, `conv12`
  K=7 Viterbi, `reedsolomon` GF(256) t=4, a didactic `ldpc`, and
  `concatenated` RS+conv); explicit MANUAL selection, NONE, and an
  evidence-based AUTO identifier that returns `UNKNOWN` unless a decoder
  actually corroborates a hypothesis.
- **De-interleaving** — block, convolutional, diagonal and pseudo-random
  families, wired through the CLI/GUI/API; automatic identification is
  block-only (the other families require explicit configuration — no
  fabricated structural evidence).
- **Frame layer** — declarative `FrameConfig` (sync word, payload bytes,
  optional CRC); never guesses a protocol.
- **Provenance** — every run records hashes, config, stage timings,
  warnings and the git commit; JSON/CSV/HTML export.
- **GNU Radio** — optional ingest layer plus a spectrum/waterfall
  visualization bridge that runs headless flowgraphs from the standalone
  `gnuradio_integration/` package in a subprocess (the NumPy DSP chain is
  never replaced).
- **ML assist (optional)** — a 16-class temporal CNN consumed by a
  NumPy-only runtime. The promoted artifact is trained inside this
  project (PyTorch trainer, exported to `.npz`) and declares its holdout;
  below the 0.60 validation floor it is shown as evidence, never as a
  result.
- **GUI** — PySide6 application with batch candidate analysis, FEC /
  interleaving / frame controls, recovered-constellation view, symbol and
  bit inspectors, light/dark theme, JSON export and a provenance dialog.

## Python API

```python
from prototype.pipeline import analyze_samples

result = analyze_samples(samples, sample_rate, reference_bits=bits)
data = result.to_dict()   # JSON-serializable structured results
```

`data["classification"]["modulation"]`, `data["symbol_rate"]`,
`data["synchronization"]`, `data["demodulation"]`, `data["ber"]`,
`data["fec_identification"]`, `data["protocol"]`, `data["provenance"]` —
every estimate carries confidence and method; stages that cannot produce
evidence return `None` or `"Unknown"` and a warning. Nothing is
fabricated.

Optional ML assist (`config.ml.enabled = True`) adds `data["ml"]`.
It is supplementary evidence alongside the rule-based classifier; see
`docs/USER_GUIDE.md` ("Machine learning").

## Supported inputs

- WAV (PCM 8/16/32-bit, float32): mono → Hilbert-analytic, stereo → IQ
- Raw interleaved IQ: complex64/128, float32/64, int8/uint8/int16/int32,
  either endianness, I-first or Q-first
- Separate I/Q files
- JSON sidecars (`<name>.meta.json`) for raw captures (sample rate,
  dtype, center frequency, …)
- Chunked streaming via `prototype.io.loaders.stream_raw_iq`
- Optional GNU Radio source (synthetic fallback when the runtime is not
  installed; live SDR capture is roadmap)
- Optional external real-world HDF5 dataset for validation only (not
  committed; see `prototype/dataset/README.md`)

## Supported modulation pipeline

| Modulation | Detection | Classification | Sync | Demodulation | BER |
|---|---|---|---|---|---|
| BPSK | yes | yes | M²-power + timing | yes | yes (polarity-searched) |
| QPSK | yes | yes | M⁴-power + timing + 45°/90° ambiguity search | yes | yes |
| 8-PSK | yes | yes | M⁸-power + timing | yes | yes (rotation-searched) |
| 16-QAM | yes | yes | M⁴-power + timing + lattice-fit re-sync | yes | yes (unambiguous) |
| BFSK | yes | yes (bimodal-IF rule) | timing via run-length rate | yes (coherent tones) | yes |
| OOK / ASK | yes | yes (envelope rule) | timing-only | yes | — |
| Unknown | yes | first-class result | n/a | refused (never guessed) | n/a |

## Architecture

```
SpectraV2/
├── prototype/               Python package (pip install -e .)
│   ├── core/                Signal model, config, provenance, preprocessing,
│   │                        isolation/DDC, synchronization, BER
│   ├── io/                  WAV/raw-IQ/pair loaders, sidecars, streaming,
│   │                        GNU Radio ingest + visualization bridge
│   ├── dsp/                 PSD/STFT, filters (FIR/IIR/polyphase), baseband,
│   │                        correlation
│   ├── detection/           FFT candidate detection + artifact suppression
│   ├── parameters/          Parameter extraction, symbol-rate estimators
│   ├── classification/      Waveform-feature + constellation classifiers
│   ├── demodulation/        Modulation dispatch, ambiguity resolution
│   ├── modulation/          Demodulator kernels + digital variants
│   ├── fec/                 CRC, Hamming, repetition, conv+Viterbi,
│   │                        Reed-Solomon, LDPC, concatenated, interleavers,
│   │                        automatic FEC/interleaving identification
│   ├── protocol/            Explicit frame/sync-word layer
│   ├── simulation/          Channel impairment simulator (ground-truthed)
│   ├── reporting/           JSON / CSV / HTML export
│   ├── benchmarking/        SNR sweep harness
│   ├── ml/                  NumPy CNN runtime, synthetic dataset generator,
│   │                        NumPy + PyTorch trainers, artifact validation
│   ├── external_validation/ Optional real-world dataset harness (needs h5py)
│   ├── visualization/       Matplotlib figures
│   ├── gui/                 PySide6 desktop application + theme
│   ├── pipeline.py          End-to-end orchestration + provenance
│   ├── pipeline_batch.py    Every-candidate batch analysis
│   └── cli.py               Command-line interface
├── gnuradio_integration/    Standalone GNU Radio flowgraphs + headless
│                            scripts (used by the visualization bridge)
├── docs/                    Architecture / user guides / validation /
│                            SIH requirement matrix
└── prototype/docs/          SIH deck pack + evidence pack + dataset notes
```

See `docs/ARCHITECTURE.md` for the data-flow diagrams and design
decisions, and `AGENTBRAIN.md` for the engineering journal.

## Development

```bash
pip install -e .[dev]
python -m pytest -q                      # full suite (headless GUI tests
                                         # need QT_QPA_PLATFORM=offscreen)
```

The optional components install separately: `pip install -e .[ml-tools]`
(h5py, one-time pickle conversion), `.[gnuradio]` (real GNU Radio
runtime), `.venv-mltrain` (CPU PyTorch, training only — never needed for
inference).

## Documentation index

- `docs/ARCHITECTURE.md` — modules, data flow, design decisions
- `docs/USER_GUIDE.md` — task-oriented usage (CLI + API + ML + dataset)
- `docs/GUI_USER_GUIDE.md` — launch-and-operate guide for the desktop GUI
- `docs/VALIDATION.md` — what is and is not validated (validation matrix)
- `docs/SIH_REQUIREMENTS.md` — SIH-147 requirement-by-requirement status
- `prototype/docs/SIH26147_SPECTRA_DECK.md` — SIH deck pack (claims traced
  to `prototype/docs/evidence/`)
- `prototype/docs/evidence/EVIDENCE.md` — generated evidence pack
- `prototype/dataset/README.md` — optional external real-world dataset
- `CHANGELOG.md` — release history

## License

Proprietary — see project governance. No telemetry, no network I/O;
all processing is offline.
