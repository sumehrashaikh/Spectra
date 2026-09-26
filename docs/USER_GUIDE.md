# Spectra User Guide

Spectra is an offline signal-analysis prototype: load an RF capture
(WAV audio or raw IQ), and it detects the signals inside, classifies
their modulation, estimates symbol timing and carrier offset,
demodulates bits, and — when you supply the transmitted bits —
measures bit error rate. Every number it reports is traceable to the
stage that produced it; it never invents a result it cannot measure.

## 1. Installation

```bash
# Python 3.10+ required; tested on 3.13
pip install -e .[dev]        # from the repository root
python -m pytest -q          # verify the suite passes
```

No internet connection is required at runtime. The software performs
no telemetry and makes no network requests.

## 2. Analyzing your first capture

### From the GUI

```bash
python main.py            # from the repository root
```

1. **Open Capture** — pick a WAV or raw IQ file (see §3 for raw IQ).
2. **Analyze Signal** — the pipeline runs on a background thread (the
   progress bar shows activity; the UI stays responsive).
3. **Detected Signals** — one row per candidate: frequency, bandwidth,
   peak level. Click a row to select it.
4. **Analyze all candidates** (checkbox) — batch mode: every detected
   candidate is analyzed. Use the "Analyzed candidate" selector to
   switch the detail panels between them.
5. Detail panels update automatically: classification, symbol rate,
   samples/symbol, timing confidence, recovered symbols/bits, BER,
   decision margin, FEC result, and synchronization frequency/phase
   offsets.
6. **Constellation** — after analysis this shows the *recovered*
   symbol lattice (the actual analysis deliverable), not the raw IQ
   smear. A clean grid means the receive chain locked.
7. **Export JSON** — save the full result payload (all stages,
   warnings, provenance). **Provenance** — per-stage timings,
   configuration, software version and git commit of the last run.
8. **FEC selector** — if the transmitter used forward error
   correction, pick the scheme (conv12, hamming74, repetition3)
   *before* analyzing; decoded results appear in the FEC row. FEC is
   explicit configuration: it is never guessed from data.

Mode presets: `quick` (fast triage), `balanced` (default),
`deep` (weak signals), `realtime` (minimal smoothing).

### From the command line

```bash
spectra validate                          # pipeline self-check (synthetic)
spectra analyze my_capture.wav            # full analysis, JSON to stdout
spectra analyze my_capture.wav --json out.json --mode deep
```

The JSON output contains, in order: input info, detections, the
selected candidate, isolation summary, symbol rate, synchronization,
classification, demodulation, BER (when a reference is available),
warnings, and the provenance manifest.

### From Python

```python
from prototype.pipeline import analyze_capture

result = analyze_capture("my_capture.wav", mode="balanced")
print(result.classification["modulation"], result.classification["confidence"])

from prototype.reporting.export import export_html
export_html(result.to_dict(), "report.html")
```

## 3. Loading captures

### WAV files

Stereo WAV: channel 0 → I, channel 1 → Q. Mono WAV: real waveform →
analytic complex signal via the Hilbert transform. Both integer PCM
and IEEE float formats are supported.

### Raw IQ files (`.iq`, `.cfile`, `.c64`, `.cf32`, `.dat`, ...)

Raw files carry no metadata, so Spectra needs a sample rate — and
ideally the sample format — from a sidecar JSON or from you:

- **GUI**: on open, a dialog asks for the sample rate (Hz) and the
  sample format (int16, complex64, float32, ...). If a sidecar
  `<name>.meta.json` sits next to the file, its values pre-fill and
  any provided fields are not asked again.
- **CLI**: pass them as flags:

```bash
spectra analyze capture.iq --sample-rate 8000 --dtype int16
```

Supported dtypes: `complex64`, `complex128`, `float32`, `float64`,
`int8`, `uint8`, `int16`, `int32`. Flags: `--endianness little|big`,
`--iq-order iq|qi` (Q-first), `--max-samples`.

Sidecar convention (`capture.iq` ← `capture.meta.json`):

```json
{
  "sample_rate": 2000000,
  "dtype": "int16",
  "endianness": "little",
  "iq_order": "iq",
  "center_frequency_hz": 433500000
}
```

Separate I/Q files: use `prototype.io.loaders.load_iq_pair` (Python
API). Streaming chunks for files larger than memory:
`prototype.io.loaders.stream_raw_iq`.

## 4. Measuring BER

BER requires the transmitted bits. Store them as a `.npz` next to the
capture — named `<capture-stem>.reference.npz` with array key `bits`
(e.g. `capture.wav` ← `capture.reference.npz`) — or pass them
explicitly:

```bash
spectra demodulate capture.wav --reference capture.reference.npz
```

```python
result = analyze_samples(samples, fs, reference_bits=tx_bits)
print(result.ber)   # {'ber': 0.0, 'ambiguity_resolution': '...', ...}
```

Blind ambiguities are searched and the applied correction is reported:
BPSK polarity (0/180°), QPSK/16-QAM rotation folds (0/90/180/270°),
and the 16-QAM frame origin (±16 symbols) — the payload's
`ambiguity_resolution` field names the correction that won. Without a
reference the pipeline stops after demodulation and reports no BER —
it never invents one.

## 5. Choosing a processing mode

| Mode | Detection | Use when |
|---|---|---|
| `quick` | lighter smoothing, stricter confidence | fast triage of many files |
| `balanced` | default settings | general analysis |
| `deep` | lower thresholds, looser confidence | weak signals, exhaustive search |
| `realtime` | minimal smoothing | streaming/previews |

## 6. Interpreting results (and non-results)

- `classification.modulation: "Unknown"` — evidence was insufficient.
  Check `confidence`, the coarse result, and warnings; try `--mode deep`
  or a cleaner isolation.
- `ber: null` — no reference bits were available.
- `symbol_rate.confidence` below ~50 — the estimate is weak; treat the
  value as a hypothesis and verify with the constellation. (Known
  limitation: short captures occasionally mislock a subharmonic —
  e.g. 100 Hz reported as 24.99 Hz; the rate value plus the
  constellation shape reveal it.)
- `warnings` list every stage that degraded; provenance `steps` shows
  status (`ok`/`skipped`/`failed`) and durations per stage.
- Block FEC schemes (e.g. hamming74) require the demodulated bit count
  to be a multiple of the block size; otherwise the FEC stage reports
  a warning and passes the bits through unchanged.

Estimates carry method tags (`delay_multiply+cyclostationarity`,
`constellation_geometry`, …) so you always know how a number was
produced.

## 7. Reports

```bash
spectra report capture.wav --html report.html --json results.json
```

The HTML report embeds the flattened results; missing values are
omitted, never fabricated. CSV export (`prototype.reporting.export.export_csv`)
gives a two-column parameter/value dump for spreadsheets. The GUI's
**Export JSON** button writes the same complete payload (including
per-stage provenance) to a file of your choice.

## 8. Benchmarks

```bash
spectra benchmark --modulation QPSK --snr-db 30,20,10,0
```

Benchmarks run the full pipeline over synthetic impaired signals and
report detection, classification, symbol-rate error, and BER per SNR.
They characterize the processing chain on simulated channels — they
are not receiver acceptance tests and say nothing about specific
hardware.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "requires a sample rate" | raw IQ without sidecar | enter it in the GUI prompt, or pass `--sample-rate` |
| "not a multiple of the sample size" | wrong dtype for a raw IQ file | pick the correct format in the prompt / `--dtype` |
| "no candidates detected" | threshold above signal | `--mode deep`, or check the file is IQ (not audio) |
| classification flips between runs | borderline confidence | inspect features; increase capture length |
| BER ≈ 0.5 on BPSK/QAM | ambiguity unresolved or sync failed | check `ambiguity_resolution`; verify with the constellation |
| rate shows 24.99 for a 100 Hz signal | symbol-rate subharmonic mislock (short captures) | known limitation; trust the constellation, not the rate alone |
| FEC shows a warning, no decode | bit count not a block multiple | pad/pick a matching scheme (repetition3 needs ×3, hamming74 ×7) |

## 10. Machine learning

Spectra includes an optional CNN stage that scores modulation class
from raw IQ. Design rules (matching the rest of the project):

- **Evidence, never authority.** The CNN's prediction is stored in the
  payload next to the rule-based DSP classification (`result.ml`);
  it never overrides or gates the DSP verdict.
- **No pickle loading.** The shipped `modulation_cnn.pkl` (a Keras 3
  Sequential CNN: input `(512, 2)`, 3× Conv1D/BN/MaxPool, Dense(16)
  softmax head) is converted once by a static opcode-walking tool that
  never executes the pickle. Runtime inference is a NumPy
  implementation of the forward pass — no TensorFlow or PyTorch
  dependency.
- **Honest labels.** The original file embedded no label map, so the
  conversion shipped with neutral `class_00`..`class_15` names and the
  BatchNorm statistics in the file proved the network had **never been
  trained** (optimizer iteration 0, BN parameters at init). The
  project therefore trains its own artifact on synthetic data.

### Using ML

Python API:

```python
from dataclasses import replace
from prototype.core.config import processing_mode_config
from prototype.pipeline import analyze_samples

config = replace(
    processing_mode_config("balanced"),
    ml=replace(config.ml, enabled=True),   # off by default
)
result = analyze_samples(samples, sample_rate, config=config)
print(result.ml["predicted_class"], result.ml["confidence"])
```

GUI: tick `ML assist (CNN)` (see `docs/GUI_USER_GUIDE.md`, Section 15).

### Training your own artifact

The trainer uses the same synthetic-dataset + channel-simulator stack
as the rest of Spectra; labels are exact by construction.

```bash
# from the prototype/ folder; NumPy only, ~10 min on CPU
python -m prototype.ml.train --frames-per-class 60 --epochs 15 \
    --output ml/modulation_cnn_trained.npz
```

- Every run first performs a finite-difference **gradient check** and
  refuses to train if backprop is wrong.
- The dataset generator covers 16 classes (BPSK … GMSK, AM, Noise)
  with randomized SNR (0–18 dB), CFO, phase, symbol rate and rolloff.
- The runtime automatically prefers `modulation_cnn_trained.npz` over
  the as-shipped conversion; per-frame unit-RMS normalization makes
  predictions amplitude-invariant.
- Holdout accuracy of the first in-project run: ~31% across 16 classes
  (chance is 6.25%) — a verified baseline, not a finished model.
  Increase `--frames-per-class`/`--epochs` for better results.

A future `labels.json` next to the artifact remaps the 16 output
indices to real class names without touching weights.

## 11. Protocol / frame analysis

When a transmitter uses a known message framing (sync/preamble word,
data-length field, and optionally a CRC), the frame layer recovers the
structured payload. The pipeline does **not** guess protocols: it only
runs the frame stage when you attach an explicit frame definition.

Configuration
-------------

Frame definitions are plain configuration — add one per protocol with
`prototype.protocol.make_frame_config`:

```python
from prototype.protocol import make_frame_config, FrameConfig

cfg = make_frame_config(
    name="TelemetryFrame",
    sync_word=0xAA55AA55,   # 32-bit preamble, MSB-first
    data_bytes=4,           # payload after the sync word
    crc=(0x07, 0x00),       # (poly, final_xor), optional
)

from prototype.core.config import processing_mode_config
from dataclasses import replace

from prototype.pipeline import analyze_capture

config = replace(
    processing_mode_config("balanced"),
    protocol=cfg,
)
result = analyze_capture("my_capture.wav", config=config)
```

The pipeline applies the frame definition to the *recovered, demodulated
(bits)*, and clearly reports `Unknown` when the sync word is not
detected — it never invents a protocol. Results include the payload
bytes, the sync-confidence, the position where the sync word landed, and
any CRC status.

Command line
------------

```bash
spectra analyze my_capture.wav --sync-word 0xAA55AA55 --data-bytes 4
spectrum_samples report my_capture.wav --sync-word 0xAA55AA55 \
    --data-bytes 4 --html frame_report.html --json frame_payload.json
```

The `--json` output then carries a `protocol` section (see §8 for JSON
export). Without a frame definition the pipeline reports `protocol: null`.

## 12. Feature checklist

| Feature | GUI | CLI | Python API |
|---|---|---|---|
| WAV open (PCM + float, mono/stereo) | ✓ | ✓ | ✓ |
| Raw IQ open (8 dtypes, endianness, I/Q order, sidecar) | ✓ (prompted) | ✓ (flags) | ✓ |
| Detection + isolation of multiple candidates | ✓ | ✓ | ✓ |
| Batch analysis of all candidates + timeline | ✓ (selector) | — | ✓ |
| Classification (BPSK/QPSK/8-PSK/16-QAM/OOK/BFSK) | ✓ | ✓ | ✓ |
| Blind synchronization + demodulation | ✓ | ✓ | ✓ |
| BER with ambiguity search | ✓ | ✓ | ✓ |
| FEC decode (conv12 / hamming74 / repetition3) | ✓ (selector) | config | ✓ |
| Recovered-constellation view | ✓ | — | payload |
| Provenance (stage timings, versions, config) | ✓ (dialog) | ✓ (JSON) | ✓ |
| JSON / HTML / CSV export | ✓ (JSON) | ✓ | ✓ |
| ML (CNN) classification, NumPy runtime | ✓ (toggle) | config | ✓ |
| ML training (synthetic dataset, gradient-checked) | — | — | ✓ |
| Protocol/frame decode (explicit FrameConfig) | ✓ (Protocol row) | ✓ (flags) | ✓ |
