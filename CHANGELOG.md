# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: semantic.

## [Unreleased]

### Added

- **Machine-learning subsystem** (`prototype/ml/`): the provided
  `modulation_cnn.pkl` was audited statically (opcode walk, never
  unpickled — pickles are untrusted input) and identified as a Keras 3
  Sequential CNN (input `(512, 2)` IQ, 3× Conv1D/BN/MaxPool, Dense(16)
  softmax). Weight inspection proved the shipped network **untrained**
  (optimizer iteration 0, BatchNorm parameters at initialization), so
  the subsystem builds the full training path instead of pretending:
  a static pickle→NumPy converter (`ml/tools/convert_keras_pickle.py`,
  self-checking fused-BN forward pass vs textbook reference, max diff
  1.5e-08), a NumPy-only inference runtime (`ml/cnn.py`, largest-stride
  frame extraction, unit-RMS normalization, honest score reporting),
  a labeled synthetic dataset generator over 16 modulation classes
  (`ml/dataset.py`, built on the repo RRC + channel simulator), and a
  gradient-checked trainer (`ml/train.py`, finite-difference max error
  1.9e-08, Adam, BN moving statistics, holdout validation). The first
  in-project trained artifact reaches ~31% holdout accuracy over 16
  classes (5× chance) as a verified baseline.
- **ML stage in the pipeline + GUI**: `AnalysisConfig.ml` (off by
  default) runs the CNN as a second opinion; `AnalysisResult.ml` and
  the provenance step `ml_classification` carry the verdict, top-3
  scores, artifact name and labels source. The GUI gained an
  `ML assist (CNN)` toolbar toggle and an `ML Prediction` parameter
  row; batch mode receives the flag through the shared config. The
  trained artifact is preferred automatically over the as-shipped
  conversion; a `labels.json` beside the artifact remaps the 16 output
  indices without touching weights.
- **GUI user guide** (`docs/GUI_USER_GUIDE.md`): a launch-and-operate
  walkthrough for GUI-only users — installation, the three launch
  paths, every panel and button explained, first-analysis walkthrough,
  raw-IQ prompts and sidecars, batch mode, BER reference convention,
  FEC selector, export/provenance, troubleshooting and limitations.
- **Friendly GUI launch paths**: `spectra gui` (new CLI subcommand
  with a graceful JSON error when the GUI stack is unavailable) and
  `python -m prototype.main` (the README's promised package entry,
  which was missing). `python main.py` from `prototype/` still works.
- **Raw IQ capture support in the GUI**: the open dialog now accepts
  raw interleaved IQ files (`.iq`, `.cfile`, `.c64`, `.cf32`, `.dat`,
  ...) alongside WAV, prompting for sample rate and sample format
  (int16/complex64/float32/...) unless a `<name>.meta.json` sidecar
  provides them — matching the CLI loader semantics (`--sample-rate`,
  `--dtype`, `--endianness`, `--iq-order`). Verified end-to-end: an
  int16 `.iq` capture of an impaired 16-QAM signal analyzes to BER 0
  through both the GUI open path and the CLI.
- **User guide rewritten** (`docs/USER_GUIDE.md`): GUI walkthrough,
  raw-IQ section (GUI prompts, CLI flags, sidecar convention),
  ambiguity-resolution and FEC notes, feature-checklist matrix.
- **GUI exposes the full V2 feature set** — FEC scheme selector
  (none/conv12/hamming74/repetition3, explicit configuration as the FEC
  framework requires) with corrected-error reporting; recovered-symbol
  constellation panel (the synchronized lattice, not the raw-IQ smear;
  `analyze_samples` gained `capture_symbol_samples`); synchronization
  frequency/phase offsets, decision margin, and pipeline BER labels;
  batch candidate selector (detail panels follow the chosen candidate);
  `candidate_timeline` attached to batch payloads; indeterminate progress
  bar during analysis; Export-JSON and Provenance (per-stage timings,
  versions, git commit) actions.
- **16-QAM end-to-end blind recovery** — a QAM-specific
  synchronization chain (`core/synchronization.py::synchronize_qam_signal`):
  sub-bin residual carrier-frequency correction (4th-power tone with
  zero-padded parabolic interpolation), a matched RRC at the receiver
  (single-RRC ISI is fatal for 16-QAM's narrow decision boundaries),
  joint (phase × fractional step) timing search on a lattice-fit
  criterion, and static decision-directed phase correction. The BER
  stage resolves the blind frame-origin ambiguity with a symbol-aligned
  ±16-symbol search alongside the existing 90° fold search. New full-
  pipeline regression `tests/test_end_to_end_qam16_full.py` asserts
  bit-exact recovery from an impaired capture.
- **GUI runs the V2 pipeline on a background thread**
  (`gui/worker.py`, `gui/window.py`): new `AnalysisWorker(QThread)`
  executes `pipeline.analyze_samples` or the multi-candidate batch
  (`pipeline_batch.analyze_all_candidates`, "Analyze all candidates"
  checkbox) off the UI thread; results are delivered as JSON-safe
  dicts through Qt signals and mapped onto the existing panels.
  Failures surface as structured error dialogs, never raised into
  the event loop. The summary dialog reports the V2 summary
  (modulation, symbol rate, samples/symbol, demodulated counts)
  plus per-stage warnings.

### Changed

- **Detection spectrum** (`detection/detector.py::compute_spectrum`):
  replaced the full-length rectangular FFT with Welch PSD averaging
  (Hann window, nperseg capped at capture length). Long captures no
  longer dilute narrowband energy across tens of thousands of bins,
  giving a stable noise floor and a peak-to-floor ratio that
  reflects real channel occupancy. Side effect (documented, tested):
  estimated candidate bandwidths are slightly wider than with the
  old estimator.

### Fixed

- **GUI "Isolate Selected" always failed** (`gui/window.py`): the slot
  still called the V1 ndarray API (`isolate_signal(samples, sample_rate,
  ...)`) while the V2 isolator expects a `Signal` and returns an
  `IsolationResult`, so every click raised "isolate_signal() requires a
  Signal object." The slot now wraps the capture in a `Signal` and
  unwraps the result into the ndarray + filter-info shape the rest of
  the window consumes (a stale `filter_info` reference in the log call
  removed with it).
- **Analyze Selected reported ~0.5 BER on 16-QAM**: the legacy
  per-modulation demod in the selected-signal path lacked the QAM
  synchronization chain (matched RRC, residual-CFO derotation,
  lattice-fit timing) and the symbol-aligned origin search, so a
  perfectly recoverable capture showed a misleading 50% BER. The
  16-QAM branch now runs the same proven chain as the main pipeline
  (`synchronize_qam_signal` → `demodulate_signal(synchronized=True)`
  → fold × symbol-origin BER search); verified BER 0 through the
  GUI isolate → analyze-selected flow.
- **GUI "Open WAV" crashed on first use** (`gui/window.py`):
  `open_wav` passed an undefined `stats` to `update_basic_information`,
  raising `NameError` the moment a file was opened — the file info
  panel never filled and the Analyze button stayed disabled. (The
  offscreen smoke test sets window state directly, which is why it
  never hit the dialog slot.) The panel now computes its own
  `basic_stats`, and the Format label reports `Raw IQ` instead of a
  hardcoded `WAV` for raw captures.
- **Symbol-rate fields never populated in the main pipeline path**:
  the SIGNAL PARAMETERS panel's Samples/Symbol, Symbol Rate and Timing
  Confidence labels are now updated from the V2 result (previously
  only the legacy selected-signal path set them).
- **16-QAM demodulation decimated twice** (`pipeline.py`): the
  synchronized ~1-sample-per-symbol stream was demodulated with the raw
  samples-per-symbol estimate (e.g. 80), collapsing 512 symbols to 7
  decisions. Synchronized QAM now demodulates at sps = 1.
- `tests/test_end_to_end_qpsk_full.py` normalized-preamble-error
  gate recalibrated 0.25 → 0.30 (with in-test rationale): the
  Welch-PSD spectrum above estimates slightly wider candidate
  bandwidths (~211 Hz vs ~170 Hz on the e2e capture),
  widening the isolation filter; measured error moved 0.1689 →
  0.2576. The binding end-to-end property — BER == 0 through the
  real demodulator with the resolved 270° phase fold — is unchanged
  and verified on both detector paths.
- GUI smoke/probe scripts (`gui_smoke.py`, `probe_win.py`): the
  pipeline summary `QMessageBox` blocked offscreen event loops
  (spurious TIMEOUT); dialogs are now stubbed and result inspection
  reads the dict payload (`symbol_rate.symbol_rate_hz`).

### Verified (synthetic)

- Full suite: **181 passed**.
- Offscreen GUI verification: worker probe (1.5 s balanced),
  full-window probe (QPSK 99.4%, 100.0 sym/s, 4030 bits), deep
  batch headless (no warnings).

## [2.2.0] - 2026-09-22

### Added

- **CI** (`.github/workflows/ci.yml`): pytest matrix across
  ubuntu-latest/windows-latest × Python 3.10/3.12.
- **Packaging**: repo-root `pyproject.toml` (console script `spectra`,
  package discovery); verified wheel build + clean-venv install + CLI.
- **Multi-signal batch analysis** (`pipeline_batch.py`):
  `analyze_all_candidates()` runs the full pipeline per candidate with
  a ranked candidate timeline; `tests/test_pipeline_batch.py`.
- **8-PSK and OOK/ASK support** (`modulation/digital.py`): Gray-coded
  8-PSK decisions, 8th-power blind phase estimator, k-means OOK
  thresholding; wired into `demodulate_signal()` dispatch, the pipeline
  demodulation stage, `modulation_order` (8th-power carrier recovery
  for 8-PSK, timing-only sync for OOK), and both classifier stages.
- **Negative/property test suite** (`tests/test_negative_property.py`,
  27 tests): loader failure modes (missing/empty/truncated/bad-offset,
  WAV empty/NaN/corrupt header), Signal validation (NaN/Inf/empty/rate),
  DSP invariances (resample identity/ratio, Welch non-negativity + tone
  location, exact STFT inverse, DC estimate, IQ-correction energy,
  channel identity + SNR, Hamming/CRC roundtrip properties), and
  detector behavior on tone/noise-only captures.

### Fixed

- **8-PSK phase estimator**: the previous `(angle(mean8) − π)/8` form
  was systematically biased (could never return a positive phase) and
  the 8-candidate tie-break loop was provably π/4-invariant (a bug also
  subtracted complex ideal points from angles). Replaced with the
  unbiased `angle(−mean(s⁸))/8`; the residual π/4 blind ambiguity is
  documented (offsets |φ| < π/8 recover exactly).
- **OOK classification**: waveform stage now uses envelope structure
  (amplitude CV, p10/p90 ratio, bimodality) instead of fragile
  phase-coherence gates; constellation stage detects one-sided line
  projections. DC-centered OOK ≡ BPSK after DC removal is documented.
- **8-PSK constellation classification**: new magnitude-CV gate
  (PSK ≈ 0.05 vs 16-QAM ≈ 0.33) plus circular phase-cluster counting
  mod 90° with median-normalized thresholding and wraparound merging
  (QPSK = 1 cluster, 8-PSK = 2) — verified on phase-locked symbols
  through the real synchronization chain; confidence scoring added.
- **BFSK demodulation** with fractional samples-per-symbol: resamples
  to an integer-sps grid before symbol slicing.

### Verified (synthetic)

- CLI end-to-end on a stereo 8-PSK WAV (1200 baud, 37 Hz CFO, 20°
  phase offset): classified 8-PSK at 80.4% (fine stage), symbol rate
  1200.0 Hz (true 1200), 1501 symbols demodulated.
- Full test suite: **181 passed** (136 → 181).

## [2.1.0] - 2026-09-22

### Added

- **Acquisition layer** (`io/loaders.py`): raw IQ loading for
  complex64/128, float32/64, int8/uint8/int16/int32 with configurable
  endianness and I/Q order; separate I/Q files; JSON metadata sidecars;
  chunked streaming (`stream_raw_iq`); IQ saving with round-trip
  guarantees; WAV float/PCM loading via `load_wav_signal`; unified
  `load_signal()` dispatcher with sidecar fallback.
- **DSP engine** (`dsp/`): Welch PSD (two-sided for complex input),
  STFT/spectrogram + inverse (COLA-exact), FIR designs (Kaiser
  low/high/band-pass/band-stop), IIR SOS (Butterworth/Chebyshev/
  elliptic), polyphase resampling, baseband ops (analytic signal,
  envelope, instantaneous phase/frequency, mixing, blind IQ-imbalance
  correction with covariance whitening, clipping detection),
  correlation utilities (xcorr, amplitude-invariant normalized xcorr,
  autocorrelation, sync-word finder with non-max suppression).
- **Channel simulator** (`simulation/channel.py`): twelve documented
  impairments (multipath, integer+fractional timing, Doppler, CFO,
  phase noise, static phase, IQ imbalance, DC, interferer, impulsive
  noise, AWGN, clipping, quantization) applied in a fixed order and
  returning exact ground-truth parameters.
- **FEC framework** (`fec/`): CRC-16/CCITT-FALSE and CRC-32 append/
  check on bit streams; systematic Hamming(7,4) encode + syndrome
  decode (verified for every single-bit error position); rate-1/3
  repetition with majority-vote decode; K=7 rate-1/2 convolutional
  encoder (poly 171/133, zero tail) + hard-decision Viterbi decoder
  storing predecessor states (verified: perfect decode with 32
  scattered errors in 806 code bits); row-column block interleaver;
  explicit scheme registry — FEC is configured, never guessed.
- **Symbol-rate estimation upgrades** (`parameters/symbol_rate.py`):
  delay-and-multiply estimator (works for constant-envelope signals
  where |x|^2 fails) as primary with |x|^2 cyclostationary
  cross-check, agreement raising combined confidence; FSK run-length
  estimator with single-run refinement (100.07 baud measured vs 100
  true on the V1 BFSK profile).
- **End-to-end pipeline** (`pipeline.py`):
  `analyze_capture()`/`analyze_samples()` orchestrating
  preprocess -> detect -> isolate -> classify -> symbol-rate -> sync
  -> fine-classify -> demodulate -> BER -> optional FEC decode with
  per-stage timing, graceful degradation into warnings, honest BER
  ambiguity handling (BPSK polarity search, QPSK 90-degree rotation
  search with the applied correction reported), and a full provenance
  manifest (software version, git commit, input hash, config, step
  timings/status).
- **Configuration** (`core/config.py`): frozen dataclasses for every
  pipeline stage plus quick/balanced/deep/realtime presets and
  validation errors for out-of-range values.
- **Exceptions** (`core/exceptions.py`): typed hierarchy rooted at
  `SpectraError`.
- **Provenance** (`core/provenance.py`): SHA-256 file hashing,
  in-memory sample hashing, timed step recording (ok/skipped/failed).
- **Reporting** (`reporting/export.py`): JSON export (numpy/complex
  safe), flat two-column CSV, self-contained HTML report with honest
  "engineering prototype" disclaimer and no fabricated values.
- **CLI** (`cli.py`): analyze/detect/classify/parameters/demodulate/
  report/benchmark/validate/version subcommands with loader overrides
  (sample rate, dtype, endianness, IQ order), JSON output, processing
  modes, and JSON-encoded errors.
- **Benchmark harness** (`benchmarking/snr_sweep.py`): SNR sweeps over
  synthetic AWGN (optionally CFO/phase/timing impaired) signals with
  honest None reporting where stages fail; explicitly labeled as
  processing-chain characterization, not real-world validation.
- **Classifier hardening** (`modulation/classifier.py`): instantaneous-
  frequency bimodality feature distinguishes FSK (long dwell near two
  tones) from pulse-shaped PSK/QAM (brief transition spikes), fixing
  RRC-QPSK being mislabeled BFSK.
- **Docs**: README, `docs/ARCHITECTURE.md`, `docs/USER_GUIDE.md`,
  `docs/VALIDATION.md`, this changelog, `AGENTBRAIN.md` journal.
- **Tests**: 121 new tests (io, dsp, channel, fec, pipeline,
  config/provenance/reporting) — 136 total.

### Fixed

- `tests/test_synchronizer.py` regression: `core/synchronizer.py`
  `synchronize_signal()` now estimates the symbol rate when not
  supplied and reports `estimated_symbol_rate`.
- FEC `fec/hamming.py` generator matrix was transposed (codewords did
  not satisfy H·cᵀ=0); rewritten as the correct 4x7 systematic matrix.
- `fec/convolutional.py` Viterbi traceback reconstructed bits from
  decision-bit history (non-invertible); rewritten to store
  predecessor states per step.
- `dsp/spectral.py::inverse_stft` applied a spurious ifftshift and
  missed scipy's `input_onesided=False` for complex input; complex
  STFT inversion is now exact.
- `dsp/baseband.py::correct_iq_imbalance` reported phase imbalance
  via arccos (range wrong for >45°); corrected to arcsin.
- `parameters/symbol_rate.py` masked input-validation ValueErrors as
  estimation RuntimeError; error classes now preserved.
- `pyproject.toml` console script pointed at nonexistent
  `spectra.cli`; fixed to `prototype.cli:main` with explicit package
  list; version aligned to 2.1.0.
- Import-convention violations in all new modules (top-level
  `core.*`/`io.*` shadowing stdlib `io` under direct execution);
  converted to the repo-standard `prototype.*` absolute imports.

### Validated (synthetic signals only)

- Full-suite regression: 136 passed.
- QPSK end-to-end through the real `demodulate_signal()` path:
  RRC-shaped 512 symbols @500 Hz carrier, AWGN: classified QPSK 99.4%,
  symbol rate 1000.24 Hz (true 1000), BER 0.0.
- BFSK end-to-end: V1 500/700 Hz profile classified BFSK, >=300 bits
  recovered, BER < 0.05.
- CLI `validate` self-check passes on a fresh synthetic QPSK capture
  loaded from a WAV file, demonstrating file->pipeline->JSON flow.
- QPSK SNR sweep (AWGN): correct detection+classification+BER 0 at
  30/20 dB; degraded results honestly reported at 10 dB; Unknown at
  0 dB.

## [2.0.0] - 2026-09-19 (baseline)

- V2 modular architecture established from V1: Signal model,
  preprocessing, FFT detection with artifact suppression, isolation/DDC,
  parameter extraction, cyclostationary symbol-rate estimation,
  M-th-power carrier recovery, timing synchronization, waveform +
  constellation classification, BPSK/QPSK/16-QAM/BFSK demodulation,
  BER validation, matplotlib visualization, PySide6 GUI.
- Validated QPSK regression: 64-symbol preamble, +37 Hz CFO, +30 deg
  phase, 13-sample timing offset, 1500 Hz interferer, noise -> BER 0.
