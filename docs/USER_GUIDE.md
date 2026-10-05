# Spectra User Guide

Spectra is an offline signal-analysis prototype: load an RF capture
(WAV audio or raw IQ), and it detects the signals inside, classifies
their modulation, estimates symbol timing and carrier offset,
demodulates bits, de-interleaves and FEC-decodes (when configured),
runs an explicit frame/sync-word layer, and — when you supply the
transmitted bits — measures bit error rate. Every number it reports is
traceable to the stage that produced it; it never invents a result it
cannot measure.

## 1. Installation

```bash
# Python 3.10+ required; tested on 3.13
pip install -e .[dev]        # from the repository root
python -m pytest -q          # verify the suite passes
```

Optional components install separately and are never required for the
core pipeline: `.[gnuradio]` (real GNU Radio runtime), `.[ml-tools]`
(h5py, one-time pickled-model conversion), `.[docs]` (python-docx), and
the CPU-PyTorch `.venv-mltrain` environment used only for ML training.
Runtime ML inference is NumPy-only.

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
   decision margin, FEC/interleaving/protocol rows, and synchronization
   frequency/phase offsets.
6. **Constellation** — after analysis this shows the *recovered*
   symbol lattice (the actual analysis deliverable), not the raw IQ
   smear. A clean grid means the receive chain locked. **View Symbols
   (I/Q)** and **View Bits / BER** open read-only inspectors listing the
   exact symbol/bit arrays the JSON export carries.
7. **Export JSON** — save the full result payload (all stages,
   warnings, provenance). **Provenance** — per-stage timings,
   configuration, software version and git commit of the last run.
8. **FEC / interleaving / frame controls** — if the transmitter used
   forward error correction or an interleaver, select the mode and
   scheme *before* analyzing (see §9); the Frame search checkbox
   configures the sync-word layer (§11). All of these are explicit
   configuration: never guessed from data.
9. **Theme** — the 🌙/☀ button in the title row toggles dark mode;
   light is the original look and toggling back restores it exactly.

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
BPSK polarity (0/180°), QPSK 45°-step quadrature folds, 8-PSK
rotations, and the 16-QAM frame origin (±24 symbols) — the payload's
`ambiguity_resolution` field names the correction that won. Without a
reference the pipeline stops after demodulation and reports no BER;
the GUI then shows a labelled EVM-based estimate (method
`evm_estimate`) instead of a measured BER — an estimate, never a
measurement.

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

## 9. Automatic FEC identification (Phase 2)

The V2 pipeline now runs an honest, deterministic, GUI-visible and
CLI-visible automatic FEC identification pass after demodulation.
Nothing is guessed past what the decoders actually measure.

### 9.1 How it works

Supported automatically evaluated schemes (the full registry):

- `none` - uncoded bitstream; kept as an honest weak hypothesis
- `repetition3` - rate-1/3 majority-vote decoder
- `hamming74` - systematic Hamming(7,4)
- `conv12` - K=7 convolutional code, Viterbi decoding
- `reedsolomon` - shortened RS over GF(256), 32 data + 8 parity bytes
- `ldpc` - compact (3,6)-regular (16,8) hard-decision bit-flip code
- `concatenated` - serial RS (outer) + convolutional (inner)

The identifier is a *candidate evaluator*, not a decoder. It only ever
runs the existing `prototype.fec` decoders to measure how well a
hypothesis fits the data. The evaluation is deterministic, explainable
and conservative; each hypothesis is scored on decoder evidence (conv12
Viterbi path metric, Hamming/repetition corrections, RS residual
syndromes, LDPC zero-flip validity, inner+outer status for the
concatenated scheme).

- clean supported FEC candidate -> score **70-83**, AUTO_DETECTED
- corrupted/ambiguous -> score 40, UNKNOWN
- insufficient / invalid -> score 0, UNKNOWN

`MIN_CONFIDENCE = 60` is the internal heuristic threshold. Confidence is
a heuristic number, never a calibrated probability.

### 9.2 GUI

The main window carries a **FEC MODE** selector: `Auto`, `Manual`, `None`.

- **Auto**: the pipeline runs automatic FEC identification after
  demodulation. On strong evidence (clean code bits) the identified
  scheme is used for decoding.
- **Manual**: the explicit configured FEC scheme is used as-is and is
  authoritative. `AUTO` never silently overwrites an explicit manual
  choice.
- **None**: no automatic identification and no automatic FEC decoding.

The result panel shows:
- auto status (AUTO_DETECTED / UNKNOWN / unresolved)
- detected scheme, confidence, corrected errors, residual estimate,
  validation
- candidate evidence list

The toolbar also carries the interleaving controls (`Auto` / `Manual` /
`None` plus `Family:` and `Depth:` for manual), and a Frame search
checkbox with sync-word/payload fields. When a transmitted reference
accompanies the capture, the reference-validated joint search can
resolve the interleaving family *and* its parameter (block depths,
seeded pseudo-random permutations, convolutional strides, the square
diagonal) by testing which hypothesis makes the code stream decodable
again. Uncoded interleaved captures stay honestly unresolved by
construction (their received stream is the transmitted stream).

### 9.3 CLI

`--fec-mode auto | manual | none` plus `--fec-scheme` and the
interleaving flags on `analyze`/`report`:

```bash
spectra analyze file.wav --fec-mode auto
spectra analyze file.wav --fec-mode manual --fec-scheme hamming74
spectra analyze file.wav --fec-mode none
spectra analyze file.wav --interleaving-mode manual \
    --interleave-family pseudo_random --interleave-depth 3
```

Identification fields (status, best_scheme, confidence, candidates,
evidence, warnings) are included in the JSON and HTML report.

### 9.4 Reporting / provenance

`fec_identification` and block-interleaving identification are recorded as
production provenance steps. CRC16/CRC32 stay error-detection-only.

### 9.5 SIH / requirement matrix (Phase 3)

The authoritative, requirement-by-requirement status — every row with its
evidence and honest limitations — is kept in
[`SIH_REQUIREMENTS.md`](SIH_REQUIREMENTS.md). Summary:

| Requirement | Implementation status | Where |
|---|---|---|
| AUTO / MANUAL / NONE FEC mode | Implemented + tested | `core/config.py` `FECMode`; GUI FEC selector; CLI `--fec-mode` |
| FEC scheme set (repetition3, hamming74, conv12, RS, LDPC, concatenated) | Implemented + tested | `fec/framework.py`, `fec/reed_solomon.py`, `fec/ldpc.py`, `fec/concatenated.py` |
| Automatic FEC identification (evidence-based) | Implemented (7 schemes) | `fec/identification.py` |
| Block interleaving mode + auto-detected depth | Implemented | `fec/identification_interleaving.py`, `pipeline.py`, GUI + CLI selectors |
| De-interleaver families (block / convolutional / diagonal / pseudo-random) | Implemented (manual config; bit-level round trip) | `fec/interleaving.py`, `FECConfig.interleave_family`, CLI `--interleave-family`, GUI family selector |
| Automatic (non-block) interleaving identification | NOT implemented | block-only by design; no fabricated structural evidence |
| Frame / sync-word search | Implemented + tested | `protocol/`; GUI frame-search control; CLI `--sync-word/--data-bytes` |
| Recovered bits / info + BER | Implemented | `demodulation.fec.decoded_bits`; "Recovered bits"/"BER" GUI rows |
| Provenance / JSON export | Implemented | `core/provenance.py`; GUI Export JSON |
| Reference-free alignment | NOT implemented | codeword alignment requires a reference; resolves 16-QAM and coded streams |
| Real-world / hardware validation | NOT done for the core chain | synthetic fixed-seed captures remain the validated set; an optional external-dataset harness (`dataset-eval`) exists for *reporting*, never for tuning, and no SDR hardware is validated |

### 9.6 Limitations

Validated on synthetic fixed-seed packets with known references. Not
real-world RF identification accuracy. Interleaving identification is now
in scope for this phase; the structural identifier runs only on the
demodulated bitstream (no waveform input) and reports AUTO_DETECTED only
on genuine structural evidence.

Blind codeword alignment (the phase/origin fold) is implemented for
16-QAM and for coded streams scored against a transmitted reference; it is
**not** resolved for uncoded QPSK/8-PSK, which are therefore reported
without a BER reference rather than with a misleading ~0.5 BER.
Automatic interleaving identification remains block-only — the other three
families are applied from explicit configuration.

## 10. Machine learning

Spectra includes an optional CNN stage that scores modulation class
from raw IQ. Design rules (matching the rest of the project):

- **Evidence first, authority never.** The CNN's output is stored in
  the payload next to the rule-based DSP classification (`result.ml`);
  it never overrides or gates the DSP verdict.
- **NumPy-only runtime, no pickle loading.** Inference is a NumPy
  implementation of the forward pass — no TensorFlow or PyTorch
  install is needed at runtime, and the shipped
  `modulation_cnn.pkl` was converted purely statically (pickle opcodes
  walked, never executed).
- **Trained in-project and validated honestly.** The promoted artifact
  (`prototype/ml/modulation_cnn_trained.npz`; a global-average-pooled
  temporal CNN over 1024-sample IQ frames, 16 classes, ~115k
  parameters) is trained on synthetic frames from this repository's
  own generator. It declares `best_val_accuracy ≈ 0.83`; the runtime's
  `ML_VALIDATION_FLOOR` is 0.60, so this artifact is **validated** and
  its argmax may be shown as a *prediction* when the capture also
  clears the confidence/agreement floors. Below the floor the row
  shows the DSP result with the CNN's own answer as a labelled second
  opinion (and the raw answer stays in the JSON either way).

### Using ML

Python API:

```python
from dataclasses import replace
from prototype.core.config import processing_mode_config
from prototype.pipeline import analyze_samples

base = processing_mode_config("balanced")
config = replace(base, ml=replace(base.ml, enabled=True))  # off by default
result = analyze_samples(samples, sample_rate, config=config)
print(result.ml["predicted_class"], result.ml["confidence"])
```

GUI: tick `ML assist (CNN)` (see `docs/GUI_USER_GUIDE.md`, Section 15).

### Training and validating your own artifact

The trainers use the same synthetic-dataset + channel-simulator stack
as the rest of Spectra; labels are exact by construction.

```bash
# NumPy trainer (dependency-free, gradient-checked)
spectra ml-train --frames-per-class 60 --epochs 15 \
    --output ml/modulation_cnn_trained.npz

# PyTorch trainer (v3 architecture; needs the optional .venv-mltrain env)
python -m prototype.ml.torch_train --frames-per-class 1500 --epochs 24 \
    --output ml/modulation_cnn_trained.npz

# Independent artifact validation: NumPy-vs-trainer parity, fresh-seed
# hold-out scored through the runtime, per-class and per-SNR accuracy
spectra ml-eval --artifact ml/modulation_cnn_trained.npz
```

- Every NumPy run first performs a finite-difference **gradient check**
  and refuses to train if backprop is wrong.
- The dataset generator covers 16 classes (BPSK … GMSK, AM, Noise)
  with randomized SNR (0–18 dB), CFO, phase, symbol rate and rolloff;
  train/val/test splits are separated by *realization*, not frame.
- The runtime automatically prefers `modulation_cnn_trained.npz` over
  the as-shipped conversion; per-frame unit-RMS normalization makes
  predictions amplitude-invariant.
- `ml/validation.py` (via `spectra ml-eval`) re-scores the shipped
  artifact through the NumPy runtime on a fresh-seed hold-out; the
  calibration report in the artifact records ECE and confidence
  behaviour (softmax scores are model scores, not calibrated
  probabilities).

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
spectra report my_capture.wav --sync-word 0xAA55AA55 \
    --data-bytes 4 --html frame_report.html --json frame_payload.json
```

The `--json` output then carries a `protocol` section (see §8 for JSON
export). Without a frame definition the pipeline reports `protocol: null`.

## 12. GNU Radio: spectrum and waterfall

GNU Radio is optional and never imported into the Spectra process. When
a GNU Radio runtime is available (e.g. a Radioconda environment), the
GUI's GNU Radio tab can compute the PSD and STFT waterfall by running
the headless flowgraph in `gnuradio_integration/` in a subprocess; the
status row names the backend actually used (`gnuradio` or `numpy`). Set
`SPECTRA_GNURADIO_PYTHON` to the interpreter that has GNU Radio and
`SPECTRA_GNURADIO_VIZ_SCRIPT` to override the flowgraph script. Every
failure falls back to the built-in NumPy plots and says so. Demodulation,
carrier recovery and constellation work always stay on the NumPy DSP
path.

## 13. Optional external real-world dataset (validation only)

`prototype/dataset/` documents an optional third-party off-air dataset
(Mendeley; ~833 MB of HDF5, never committed). Install `h5py` and place
the `subset_*.h5` files there (or pass `--dataset-dir`) to use it as an
*external validation input only* — it is never used for training:

```bash
spectra dataset-inspect                 # shapes, labels, balance
spectra dataset-eval --subset test --per-cell 3 --table
```

`dataset-eval` runs Spectra's existing preprocessing + DSP classifier
(and optionally the ML assist) over sampled frames and breaks the results
down by modulation, clean/multipath channel and labelled SNR. Unsupported
classes are reported as unsupported, never scored as errors. Separately,
`prototype/docs/evidence/` contains the frozen public-benchmark evidence
pack (RadioML 2016.10a transfer results, demo matrix, requirement
matrix) used by the SIH deck.

## 14. Feature checklist

| Feature | GUI | CLI | Python API |
|---|---|---|---|
| WAV open (PCM + float, mono/stereo) | ✓ | ✓ | ✓ |
| Raw IQ open (8 dtypes, endianness, I/Q order, sidecar) | ✓ (prompted) | ✓ (flags) | ✓ |
| Detection + isolation of multiple candidates | ✓ | ✓ | ✓ |
| Batch analysis of all candidates + timeline | ✓ (selector) | — | ✓ |
| Classification (BPSK/QPSK/8-PSK/16-QAM/OOK/BFSK) | ✓ | ✓ | ✓ |
| Blind synchronization + demodulation | ✓ | ✓ | ✓ |
| BER with ambiguity search (+ EVM estimate without reference) | ✓ | ✓ | ✓ |
| FEC decode (6 schemes) | ✓ (mode + scheme) | ✓ (flags) | ✓ |
| Automatic FEC identification (AUTO) | ✓ (Auto FEC row) | ✓ (--fec-mode auto) | ✓ |
| De-interleaving (block/conv/diagonal/pseudo-random) | ✓ (family + depth) | ✓ (flags) | ✓ |
| Protocol/frame decode (explicit FrameConfig) | ✓ (Frame search) | ✓ (flags) | ✓ |
| Recovered-constellation view + symbol/bit inspectors | ✓ | payload | payload |
| GNU Radio spectrum/waterfall (optional backend) | ✓ (checkbox) | — | ✓ |
| Light/dark theme | ✓ | — | — |
| Provenance (stage timings, versions, config) | ✓ (dialog) | ✓ (JSON) | ✓ |
| JSON / HTML / CSV export | ✓ (JSON) | ✓ | ✓ |
| ML assist (CNN), NumPy runtime, validated artifact | ✓ (toggle) | ✓ (--ml) | ✓ |
| ML training + independent artifact validation | — | ✓ (ml-train/ml-eval) | ✓ |
| External real-world dataset harness (optional, h5py) | — | ✓ (dataset-*) | ✓ |
