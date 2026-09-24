# Spectra Architecture

## 1. System overview

```
                    ┌──────────────────────────────────────────────┐
                    │                 ENTRY POINTS                 │
                    │  cli.py (spectra ...)   gui/window.py (Qt)   │
                    │  pipeline.py (analyze_samples/capture)       │
                    └───────────────────────┬──────────────────────┘
                                            │
   ┌────────────────────────────────────────▼────────────────────────────────────────┐
   │                             ANALYSIS PIPELINE STAGES                            │
   │                                                                                 │
   │  io/loaders          WAV / raw IQ / I+Q pair / sidecar → Signal                 │
   │        │                                                                        │
   │  core/preprocessor   validate → DC removal → noise floor → SNR → normalize      │
   │        │                                                                        │
   │  detection/detector  FFT → smoothing → peaks → region expansion/merge →         │
   │        │              artifact suppression → [SignalCandidate]                  │
   │        │                                                                        │
   │  core/isolator       frequency shift (NCO) → Butterworth LPF (sosfiltfilt)      │
   │        │              → isolated baseband Signal (provenance kept)              │
   │        │                                                                        │
   │  classification      coarse (waveform features) ──┐                             │
   │  parameters          symbol rate (delay-multiply │ |x|^2 | FSK run-length)        │
   │  core/synchronization│ carrier (M-th power) → timing (sampling-phase search)      │
   │  classification      fine (constellation geometry) ◄──────────────────────────┘  │
   │        │                                                                        │
   │  demodulation        demodulate_signal() dispatch → modulation kernels          │
   │        │                                                                        │
   │  modulation/ber      polarity-aware / rotation-aware BER vs reference           │
   │        │                                                                        │
   │  fec (optional)      explicit scheme: decode_bits(bits, scheme)                 │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
                     ┌──────────────────────▼───────────────────────┐
                     │            OUTPUT / PERSISTENCE              │
                     │  AnalysisResult dataclass → to_dict()        │
                     │  reporting/export: JSON | CSV | HTML         │
                     │  core/provenance: manifest w/ step timings   │
                     └──────────────────────────────────────────────┘
```

## 2. Module responsibilities

| Module | Responsibility | Key types |
|---|---|---|
| `core/signal.py` | Immutable-ish Signal container with validation | `Signal` |
| `core/config.py` | All tunables; quick/balanced/deep/realtime presets | `AnalysisConfig` … |
| `core/exceptions.py` | Typed errors rooted at `SpectraError` | `LoaderError`, `PipelineError`, … |
| `core/provenance.py` | Hashing, git commit, per-step timing/status | `AnalysisProvenance` |
| `io/loaders.py` | Acquisition of every supported format; streaming | `CaptureMetadata` |
| `dsp/spectral.py` | PSD, spectrogram, inverse STFT | — |
| `dsp/filters.py` | FIR/IIR design+apply, polyphase resample | — |
| `dsp/baseband.py` | Analytic signal, IF, mixing, IQ correction, clipping | `IQImbalance` |
| `dsp/correlation.py` | xcorr, normalized xcorr, sync-word search | — |
| `detection/detector.py` | Multi-candidate spectral detection | `SignalCandidate`, `DetectionResult` |
| `core/isolator.py` | DDC + channel filtering | `IsolationResult` |
| `parameters/extractor.py` | RF parameters (freq, BW, power, SNR) | `SignalParameters` |
| `parameters/symbol_rate.py` | Delay-multiply, \|x\|², FSK run-length estimators; RRC utils | `SymbolRateEstimate` |
| `core/carrier_sync.py` | M-th power CFO + phase estimation/correction | `CarrierRecoveryResult` |
| `core/synchronizer.py` | Timing phase search, symbol recovery, rate estimation | `SynchronizationResult` |
| `core/synchronization.py` | Carrier→timing composition | `FullSynchronizationResult` |
| `classification/classifier.py` | Waveform + constellation fusion, confidence | `ClassificationResult` |
| `modulation/classifier.py` | Feature extraction & decision rules | — |
| `demodulation/demodulator.py` | Public demod dispatch (V2) | `DemodulationResult` |
| `modulation/demodulator.py` | Symbol kernels: BPSK/QPSK/16-QAM/BFSK + BER (V1) | — |
| `demodulation/qpsk_sync.py` | 90°-ambiguity resolution vs preamble | — |
| `fec/*` | CRC, Hamming, repetition, convolutional+Viterbi, interleaver | `FECResult` |
| `simulation/channel.py` | Impairment chain with ground truth | `ChannelConfig`, `ChannelOutput` |
| `reporting/export.py` | JSON/CSV/HTML serialization | — |
| `benchmarking/snr_sweep.py` | SNR sweeps with honest failure reporting | — |
| `pipeline.py` | Orchestration + provenance + graceful degradation | `AnalysisResult` |
| `cli.py` | CLI over the same public API | — |

## 3. Key design decisions

**Structured results.** Every stage returns a dataclass with a
`.summary()`/`to_dict()`. The pipeline composes these into
`AnalysisResult`, which serializes to JSON with numpy/complex
conversion. Console text is a presentation concern (CLI/GUI), never
the analysis output format.

**Graceful degradation.** A failing stage records a `failed` step and
a warning instead of aborting: a capture that cannot be classified
still yields detection and isolation results. Stages that would
produce *guesses* (e.g. demodulating an Unknown modulation) are
refused deliberately.

**Honest confidence.** Confidence values are computed from measured
quantities (phase coherence, clustering quality, spectral ratio) and
mean exactly what the formulas say. `Unknown` is a first-class result.
BER is reported after searching blind ambiguities (BPSK polarity,
QPSK 90° rotations) and the applied correction is recorded — never
hidden.

**Provenance.** Every run records software version, git commit,
input file SHA-256, configuration snapshot, and per-stage timings.
Re-running with the same inputs and config reproduces the analysis
(bit-exact given fixed seeds).

**FEC is configured, not inferred.** The FEC registry is explicit;
the pipeline applies FEC only when `AnalysisConfig.fec.scheme` is
set. There is no "auto-detect coding" pretending to know a protocol.

**Validation honesty.** Everything in `tests/` and `benchmarking/`
runs on synthetic signals with ground truth. `docs/VALIDATION.md`
separates validated claims from unvalidated ones; the HTML report
footer and README repeat the limitation.

## 4. Data model conventions

- Samples are `complex128` end-to-end (`Signal.__post_init__` enforces).
- Frequencies in Hz, times in seconds, powers in linear units, dB only
  as explicitly-labeled `*_db` fields. No dBm without calibration.
- Metadata flows forward: each stage copies and extends the previous
  Signal's metadata; nothing is silently dropped or overwritten.
- The raw capture is kept (`raw_signal`) while preprocessing produces
  a derived signal; both remain available.

## 5. Extension points

- **New modulation**: kernel in `modulation/demodulator.py`, dispatch
  arm in `demodulation/demodulator.py::demodulate_signal`, classifier
  rules in `modulation/classifier.py`, pipeline order mapping in
  `pipeline.py::_modulation_order`.
- **New FEC scheme**: implement `encode(bits)` /
  `decode(bits) -> (bits, stats)`, register in `fec/framework.py`.
- **New capture format**: loader in `io/loaders.py` returning
  `Signal` with `CaptureMetadata`; extend `load_signal()` dispatch.
- **SDR hardware**: implement a source producing chunks consumable by
  `analyze_samples` per block; the pipeline is capture-agnostic.
