# SPECTRA — RF / Signal Analysis Platform

Spectra loads IQ/WAV captures, detects signals, isolates candidates,
extracts parameters, estimates symbol rate, synchronizes carrier and
timing, classifies modulation, demodulates, and reports BER — from the
Python API, the command line, or the desktop GUI.

**Scope and validation status (read this first).** Spectra is an
engineering prototype validated exclusively on *synthetic* signals. It
is **not** certified, flight-qualified, mission-qualified, or approved
for operational deployment by any organization. Benchmark numbers in
this repository characterize the processing chain on simulated
channels; they are not real-world or hardware validation.

## Quick start

```bash
# from the repository root (package folder: prototype/)
pip install -e .

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

# HTML + JSON report
spectra report capture.wav --html report.html --json results.json

# SNR benchmark (synthetic AWGN channel)
spectra benchmark --modulation QPSK --snr-db 30,20,10,0

# Desktop GUI
spectra gui                  # or: python -m prototype.main
```

Every analytic command accepts `--json <path>` for machine-readable
output and `--mode quick|balanced|deep|realtime` for processing depth.

## Python API

```python
from prototype.pipeline import analyze_samples

result = analyze_samples(samples, sample_rate, reference_bits=bits)
data = result.to_dict()   # JSON-serializable structured results
```

`data["classification"]["modulation"]`, `data["symbol_rate"]`,
`data["synchronization"]`, `data["demodulation"]`, `data["ber"]`,
`data["provenance"]` — every estimate carries confidence and method;
stages that cannot produce evidence return `None` or `"Unknown"` and a
warning. Nothing is fabricated.

Optional ML assist (`config.ml.enabled = True`) adds `data["ml"]`: a
CNN's modulation scores computed by a NumPy-only runtime. It is
supplementary evidence alongside the rule-based classifier, never a
replacement; see `docs/USER_GUIDE.md` ("Machine learning").

## Supported inputs

- WAV (PCM 8/16/32-bit, float32): mono → Hilbert-analytic, stereo → IQ
- Raw interleaved IQ: complex64/128, float32/64, int8/uint8/int16/int32,
  either endianness, I-first or Q-first
- Separate I/Q files
- JSON sidecars (`<name>.meta.json`) for raw captures (sample rate,
  dtype, center frequency, …)
- Chunked streaming via `prototype.io.loaders.stream_raw_iq`
- Optional SDR hardware: not integrated (offline-only tool). The loader
  abstraction accepts any file-based capture.

## Supported modulation pipeline

| Modulation | Detection | Classification | Sync | Demodulation | BER |
|---|---|---|---|---|---|
| BPSK | yes | yes | M²-power + timing | yes | yes (polarity-searched) |
| QPSK | yes | yes | M⁴-power + timing + 90° ambiguity search | yes | yes |
| 16-QAM | yes | yes | M⁴-power + timing | yes | yes (unambiguous) |
| BFSK | yes | yes (bimodal-IF rule) | timing via run-length rate | yes (coherent tones) | yes |
| Unknown | yes | first-class result | n/a | refused (never guessed) | n/a |

## Architecture

```
prototype/
├── core/           Signal model, exceptions, config, provenance,
│                   preprocessing, isolation/DDC, sync, BER
├── io/             Acquisition: WAV/raw-IQ/pair loaders, sidecars, streaming
├── dsp/            PSD/STFT, filters (FIR/IIR/polyphase), baseband, correlation
├── detection/      FFT candidate detection + artifact suppression
├── parameters/     Parameter extraction, symbol-rate estimators
├── classification/ Waveform-feature + constellation-geometry classifiers
├── demodulation/   Modulation dispatch, QPSK ambiguity resolution
├── modulation/     Demodulator kernels (V1-derived, validated)
├── fec/            CRC16/32, Hamming(7,4), repetition, K=7 conv+Viterbi, interleaver
├── simulation/     Channel impairment simulator (ground-truthed)
├── reporting/      JSON / CSV / HTML export
├── benchmarking/   SNR sweep harness
├── ml/             ML subsystem: NumPy CNN runtime, synthetic dataset
│                   generator, gradient-checked trainer, conversion tool
├── visualization/  Matplotlib figures (spectrum/waterfall/constellation)
├── gui/            PySide6 desktop application
├── pipeline.py     End-to-end orchestration + provenance
└── cli.py          Command-line interface
```

See `docs/ARCHITECTURE.md` for the data-flow diagrams and design
decisions, and `AGENTBRAIN.md` for the engineering journal.

## Development

```bash
pip install -e .[dev]
python -m pytest -q          # full test suite
```

## Documentation index

- `docs/ARCHITECTURE.md` — modules, data flow, design decisions
- `docs/USER_GUIDE.md` — task-oriented usage (CLI + API)
- `docs/GUI_USER_GUIDE.md` — launch-and-operate guide for the desktop GUI
- `docs/VALIDATION.md` — what is and is not validated (validation matrix)
- `CHANGELOG.md` — release history
- `AGENTBRAIN.md` — engineering journal / agent continuity

## License

Proprietary — see project governance. No telemetry, no network I/O;
all processing is offline.
