# AGENTBRAIN — SPECTRA V2 Engineering Journal

> Living document. Every agent working on this repo MUST read this file first and
> update it at the end of their session (what changed, what was verified, what remains).
> Honesty rule: never record validation that was not actually executed.

## 1. Project Overview

SPECTRA is an RF/signal-analysis platform: load IQ/WAV captures → detect signals →
isolate candidates → extract parameters → estimate symbol rate → synchronize carrier
and timing → classify modulation → demodulate → measure BER → report.

- Package name: `prototype` (repo root is `SpectraV2/prototype`; conftest.py puts the
  parent dir on `sys.path` so `from prototype.core.signal import Signal` works).
- Python: 3.13 (Windows dev box). Deps: numpy, scipy, matplotlib, PySide6, scikit-learn (optional), python-docx (optional).
- Status: **engineering prototype, synthetic-data validated only.** No hardware, no
  real-world captures, no certification. Do not claim otherwise in docs or output.

## 2. Architecture Map (updated 2026-09-22, session 2)

| Module | File(s) | Status | Notes |
|---|---|---|---|
| Signal model | `core/signal.py` | WORKING | `Signal` dataclass, complex128, validates NaN/Inf, `metadata` dict |
| Loader (V1) | `core/loader.py` | WORKING | WAV mono→Hilbert, stereo→IQ. Uses `print()` (legacy) |
| Loader (V2) | `io/loaders.py` | NEW | raw IQ dtypes/endianness/order, sidecars, streaming, save_iq |
| Preprocessing | `core/preprocessor.py` | WORKING | DC removal, RMS normalize, spectral noise floor, SNR |
| Detection | `detection/detector.py` | WORKING | FFT-based candidates, artifact suppression, `SignalCandidate` |
| Legacy analyzer | `core/analyzer.py` | WORKING | V1 dict-based analysis used by GUI |
| Isolation/DDC | `core/isolator.py` | WORKING | freq shift + Butterworth sosfiltfilt |
| Parameters | `parameters/extractor.py` | WORKING | peak/center freq, 99% BW, noise, SNR |
| Symbol rate | `parameters/symbol_rate.py` | WORKING | cyclostationary |x|^2 method + RRC/BPSK generators; FSK run-length estimator added |
| Carrier sync | `core/carrier_sync.py` | WORKING | M-th power freq+phase, confidence |
| Timing sync | `core/synchronizer.py` | WORKING | sampling-phase search over magnitude variation; now also estimates symbol rate when not given |
| Timing (alt) | `core/timing.py` | WORKING | folded-envelope clock recovery (`recover_symbol_timing`) used by GUI |
| Full sync | `core/synchronization.py` | WORKING | carrier→timing chain, preserves metadata |
| Classification | `classification/classifier.py` | WORKING | wraps `modulation/classifier.py` (waveform features + constellation geometry). BPSK/QPSK/8-PSK/16-QAM/BFSK/OOK/Unknown |
| Advanced demod | `modulation/digital.py` | DONE+TESTED | 8-PSK (8th-power phase, unbiased `angle(-mean8)/8` estimator; pi/4 fold documented) + OOK/ASK (k-means threshold) |
| Batch analysis | `pipeline_batch.py` | DONE+TESTED | analyze all candidates, per-candidate results + timeline |
| QPSK ambiguity | `demodulation/qpsk_sync.py` | WORKING | preamble-based 90° resolution, Gray mapping |
| BPSK decisions | `core/bpsk.py` | WORKING | 2nd-moment phase align + hard decisions |
| BER | `core/ber.py`, `modulation/demodulator.py::calculate_ber` | WORKING | polarity-aware BPSK validation vs `.reference.npz` |
| Visualization | `visualization/plots.py` | WORKING | matplotlib figures (time/spectrum/waterfall/constellation) |
| GUI | `gui/window.py` | WORKING | PySide6, V1-derived, monolithic (~1600 lines) — refactor candidate |
| DSP engine | `dsp/` | DONE+TESTED | Welch PSD (2-sided complex), STFT+inverse (exact for complex), FIR Kaiser, IIR SOS, polyphase resampling, baseband (analytic/IF/mixing/blind IQ correction/clipping), correlation+sync-word finder |
| Channel sim | `simulation/channel.py` | DONE+TESTED | 12 impairments in fixed order, returns exact ground truth |
| FEC | `fec/` | DONE+TESTED | CRC-16/32 append+check, Hamming(7,4) (all single-error positions verified), repetition3, K=7 r=1/2 conv (171,133)+Viterbi with predecessor-state traceback (perfect at 32 scattered errors/806 bits), block interleaver, explicit registry |
| Pipeline | `pipeline.py` | DONE+TESTED | analyze_samples/analyze_capture: preprocess→detect→isolate→classify→rate→sync→fine-classify→demodulate→BER(ambiguity-searched)→optional FEC; per-stage provenance; graceful warnings |
| Reporting | `reporting/export.py` | DONE+TESTED | JSON (numpy/complex-safe), flat CSV, self-contained HTML with disclaimer |
| CLI | `cli.py` | DONE+TESTED | analyze/detect/classify/parameters/demodulate/report/benchmark/validate/version; loader overrides; JSON out |
| Benchmark | `benchmarking/snr_sweep.py` | DONE+TESTED | honest SNR sweep (QPSK: OK ≥20 dB, degraded 10 dB, Unknown 0 dB) |
| Config | `core/config.py` | DONE+TESTED | frozen dataclasses + quick/balanced/deep/realtime presets |
| Provenance | `core/provenance.py` | DONE+TESTED | sha256 file+samples, git commit, step ctx mgr (ok/skipped/failed) |
| Docs | README, CHANGELOG, docs/{ARCHITECTURE,USER_GUIDE,VALIDATION}.md | DONE | honest scope throughout |
| Empty dirs | `ml/`, `correlation/` | STUB | ml intentionally not started (no fabricated models); correlation code lives in `dsp/correlation.py`; interleaving in `fec/interleaving.py` |

## 3. Conventions

- Imports: absolute `prototype.<pkg>...` (conftest.py handles sys.path for tests; GUI/main run from repo root). detection/ uses relative `..core.` imports — keep as is.
- Structured results = dataclasses with `.summary()` or `asdict`; no bare dicts in NEW code.
- Logging: `from prototype.core.logging_config import logger` (new code). `print()` only in tests/legacy.
- New tunables go in `core/config.py`, not hardcoded.
- Never fabricate: Unknown is a valid classification result; confidence < threshold → Unknown.
- Units: Hz, seconds, dB (power). Never dBm without calibration metadata.
- Test files: `tests/test_*.py`, run via `python -m pytest -q` from `prototype/`.

## 4. Validated Regressions (preserve these!)

- `tests/test_end_to_end_bpsk_full.py`: RRC BPSK @500Hz, +37Hz CFO, +30° phase, 13-sample
  timing offset, 1500Hz interferer, noise → detect/isolate/params/symbol-rate/sync/classify/
  demodulate → BER == 0. Uses the real `demodulate_signal()` path.
- `tests/test_end_to_end_qpsk_full.py`: same impairment set + 64-symbol preamble → 0 bit errors.
- `tests/test_classification_pipeline.py`: BPSK/QPSK/16-QAM/BFSK classify correctly; pure tone → Unknown.
- `tests/test_advanced_modulation.py`: 8-PSK clean+noise roundtrip, OOK roundtrip,
  classifier OOK detection, dispatch through `demodulate_signal`.
- `tests/test_negative_property.py`: loader failure modes, Signal validation,
  DSP invariances, FEC roundtrip property, CRC single-bit detection.
- QPSK phase-ambiguity resolution and symbol-rate estimation have dedicated tests.
- Full suite must stay green after every change: `python -m pytest -q` (181 passed
  as of session 3).

## 5. Known Issues / Debt

- `gui/window.py` monolithic; still uses `print()` and V1 analyzer; has duplicated
  BFSK reset lines. Needs modularization + background worker thread (QThread).
- `main.py` entry works only from repo root (imports `prototype.gui.window`).
- `modulation/psk.py`, `qam.py`, `fsk.py` are empty — demod kernels live in
  `modulation/demodulator.py` + `modulation/digital.py`. Either fill or delete.
- `core/logging_config.py` configures on import (side effect).
- BFSK demodulator resamples to integer sps when fractional (fix in session 3),
  but sub-0.02 drift warning still emitted for edge cases.
- 8-PSK blind phase recovery folds modulo 45° (fundamental constellation
  symmetry); offsets |phi| > pi/8 need preamble/differential coding.
- DC-centered OOK classifies as BPSK (mathematically identical after DC removal);
  only pre-correction envelope structure distinguishes them — documented.
- Classifier reliably works ≥ ~15–20 dB SNR for 8-sps RRC signals; low-SNR
  robustness (10 dB) needs longer captures/retuned thresholds.
- ML subsystem not started (deliberate: deterministic DSP first, no fabricated models).
- scipy 1.17 `istft(boundary=None)` crashes on even-sized input — our
  `inverse_stft` uses default boundary and is verified exact; avoid boundary=None.

## 6. Work Log

### 2026-09-24 — Session 7: full feature verification + raw .IQ in the GUI + user guide — VERIFIED

- **Feature check (all green)**: 182 pytest; GUI smoke (WAV→QPSK 98.4%);
  window probe (16-QAM, BER 0, repetition3 corrected 490 errors); batch
  probe (3 candidates, frequency-ordered timeline, selector switching);
  screenshots re-captured.
- **Raw .IQ support in the GUI** (system side — `io/loaders.py` 8 dtypes,
  endianness, I/Q order, `.meta.json` sidecars, streaming — and the CLI
  flags already existed; the GUI was the only WAV-only piece):
  `open_wav` is now an open-capture dialog (WAV + common raw-IQ
  extensions); `_open_raw_iq` prompts for sample rate and sample format
  (only for fields a sidecar doesn't provide) and routes through
  `load_signal` — identical semantics to the CLI. Verified end-to-end:
  the same 16-QAM capture exported as interleaved int16 `.iq` opens,
  classifies 16-QAM (93.1%), symbol rate 99.97, **BER 0** through the
  GUI open path; CLI `analyze test_capture.iq --sample-rate 8000
  --dtype int16` agrees (BER 0.0). One bug found and fixed during
  verification: prompted values were dropped instead of passed as
  overrides.
- **`docs/USER_GUIDE.md` rewritten**: GUI walkthrough (batch selector,
  FEC, export/provenance, recovered constellation), raw-IQ section
  covering GUI prompts + CLI flags + sidecar convention, updated
  ambiguity-resolution notes (16-QAM origin search), FEC block-size
  warning, subharmonic-mislock caveat, refreshed troubleshooting table,
  and a feature-checklist matrix (GUI/CLI/API). Removed the stale
  "GUI worker is a known gap" note (the worker has existed since
  session 4).
- Probe files: `probe_iq.py` added (raw-IQ window probe);
  `test_capture.iq` + `test_capture.reference.npz` kept as the raw-IQ
  fixture pair for probes.
- Full suite: **182 passed**.

### 2026-09-24 — Session 6: GUI/pipeline feature alignment + GUI upgrades — VERIFIED

- **Audit** (pipeline payload vs GUI): the GUI ignored or under-exposed
  FEC decoding, provenance, the synchronized constellation, sync
  frequency/phase detail, BER results, decision margin, batch candidate
  detail (hardcoded to candidate 0), and `candidate_timeline`. All now
  wired.
- **FEC selector** (toolbar: none/conv12/hamming74/repetition3): FEC is
  explicit configuration, so the selector feeds `AnalysisConfig.fec` in
  `AnalysisWorker`; `demodulation.fec` (scheme, corrected errors,
  uncorrectable blocks) now shows in the parameters panel and summary
  dialog. Note: block schemes (hamming74) honestly refuse bit counts
  that aren't block multiples — that surfaces as a stage warning.
- **Recovered constellation**: `analyze_samples(capture_symbol_samples=)`
  attaches up to 4096 synchronized symbols (JSON-safe pairs) to the
  demod payload; the constellation panel prefers it over the raw-IQ
  scatter (which is a smear for pulse-shaped signals). New
  `create_symbol_constellation_figure` in `visualization/plots.py`.
  Verified visually: a clean 4×4 16-QAM lattice from the e2e capture.
- **Sync detail + BER + margin**: frequency/phase offset (Hz/deg),
  decision margin, and pipeline BER (with error/compared counts) now
  fill dedicated parameter-panel labels.
- **Batch candidate selector**: batch mode populates a combo
  ("#1: QPSK …"); switching re-targets the detail panels at that
  candidate. `candidate_timeline` is attached to batch payloads in the
  worker (supplementary, failure-tolerant) and lands in exports.
- **Progress + export + provenance**: indeterminate progress bar while
  the worker runs; "Export JSON" saves the full payload (stages,
  warnings, provenance); "Provenance" dialog shows per-stage timings,
  versions, git commit.
- **Verification (offscreen)**: window probe on a generated impaired
  16-QAM WAV + `.reference.npz` sidecar — BER 0 (0/2040), FEC repetition3
  corrected 490 errors, freq −1.086 Hz / phase 7.35° shown, constellation
  510 points captured, fec_decode+ber stages in provenance. Batch probe:
  3 candidates, timeline rows frequency-ordered, selector switches detail.
  Probe captures regenerated: `gui_qam16.wav` (was `gui_smoke.wav`),
  `probe_win.py` updated, `probe_worker.py` removed (covered by
  `probe_win.py`), `probe_batch.py` added; screenshots re-captured.
- Full suite: **182 passed**.

Known gaps (deliberate): batch mode still reports BER per candidate
against the same reference (near-0.5 for wrong candidates — honest);
realtime mode's GUI entry is unchanged; no drag-and-drop file open.

### 2026-09-23 — Session 5: 16-QAM end-to-end blind recovery — VERIFIED

- **Goal** (journal item 6 + recommended #1): make 16-QAM recover bit-exact
  through the V2 pipeline. It previously reported ~0.45–0.5 BER on captures
  that QPSK handled perfectly.
- **Demod threading bug fixed** (`pipeline.py::_demodulate`): the 16-QAM
  branch passed the raw sps estimate (e.g. 80) into demodulation of the
  already-synchronized ~1-sps stream, decimating it a second time (512
  symbols → 7). Synchronized QAM now demodulates with sps=1.0.
- **Root cause of the residual BER — three stacked defects**, each invisible
  on PSK:
  1. **No matched filter.** TX shapes with a single RRC; zero-ISI only
     holds after a *second* identical RRC at the receiver. Single-RRC ISI
     is h(T)/h(0) ≈ 0.077 for β=0.35 — harmless for QPSK's wide quadrant
     decisions, fatal for 16-QAM's ±0.316 inner boundaries. Verified by
     slicing the *raw TX-side* shaped waveform: 50% BER with no channel
     at all.
  2. **Residual CFO.** The isolation DDC's frequency grid is coarse;
     ~1 Hz of residual rotates the constellation ~19 rad over a 5 s
     capture. PSK stages don't see this (preamble/fold math is rotation
     tolerant); QAM geometry is destroyed.
  3. **Origin search granularity.** The BER-stage frame-origin search
     stepped by single *bits* over ±16 bits; the true offset was 8
     *symbols* (RRC group delay 640 = 8·80) — unreachable. Classic
     units bug; found by validating the harness against raw TX symbols
     (BER 0.0) before trusting any negative result.
- **New `synchronize_qam_signal`** (`core/synchronization.py`): QAM-specific
  chain — (1a) 4th-power residual-CFO estimate with zero-padded parabolic
  interpolation (~0.01 Hz effective resolution; 0.02 Hz deadband — bin-
  quantized estimates applied verbatim *introduce* rotation) — (1b) matched
  RRC — (2) joint (integer phase × fractional step) grid search on the
  lattice-fit criterion (mean distance to the ideal grid; sharp at true
  timing, unlike the magnitude-variation metric which has *no* peak on
  amplitude-modulated carriers) — (3) static decision-directed phase
  correction. Dead ends documented in-code: per-segment drift-slope timing
  (fit curve too shallow, argmax picks fabricate slopes) and a DD phase
  *tracking* loop (random walk when there is no real drift; static DD is
  idempotent and safe).
- **BER stage** (`pipeline.py::_evaluate_ber`): 16-QAM now searches the
  unobservable 90° folds × symbol-aligned frame origin ±16 symbols (both
  directions); blind origin ambiguity is inherent, resolving it against a
  reference is honest.
- **New regression** `tests/test_end_to_end_qam16_full.py`: full-pipeline
  bit-exact assertion on an impaired 16-QAM capture (13-sample timing,
  37 Hz CFO, 30° phase, noise) — exercises all three fixes.
- **Seed sweep** (20 seeds, same impairment model): **14/20 bit-exact**;
  every failure is the *pre-existing* symbol-rate estimator locking a
  subharmonic (24.99 ≈ 100/4, 33.39) — it mislocks QPSK captures the same
  way (seeds 6/7/20). When the rate estimate is correct, 16-QAM is
  bit-exact 14/14. **Next work: rate-estimator subharmonic robustness.**
- Full suite: **182 passed**.

### 2026-09-23 — Session 4: GUI modernization onto the V2 pipeline (QThread) — VERIFIED

- **Background worker** (`gui/worker.py`, new): `AnalysisWorker(QThread)` runs
  `pipeline.analyze_samples` (single candidate) or `pipeline_batch.analyze_all_candidates`
  (batch checkbox) off the UI thread; results via `finished_with_result(payload_dict)`,
  failures via `failed(str)` — exceptions never cross into the Qt event loop.
- **Window rewired** (`gui/window.py`, +561/−133): Analyze button → worker start;
  `_on_pipeline_finished` → `_apply_pipeline_result` maps the V2 payload onto the
  existing V1-shaped panels (`_analysis_from_pipeline` converts detections to the
  legacy table shape; `_apply_candidate_detail` fills modulation/params views;
  batch mode shows candidate 1 detail + multi-candidate table). "Analyze all
  candidates" checkbox; mode combo drives pipeline mode. Summary dialog now
  reports the V2 summary text (modulation, rate, sps, demod counts) + warnings.
- **Verified (offscreen, `QT_QPA_PLATFORM=offscreen`)**: worker isolation probe
  (1.5 s balanced run, full payload), full-window probe (QPSK 99.4%, rate
  100.0 sym/s, 4030 bits, summary dialog correct), deep batch headless (1
  candidate, 1.0 s, no warnings). Note: probes must run with cwd=prototype/
  on sys.path (PYTHONPATH=..) — same convention as conftest.
- **Smoke-script fixes**: `gui_smoke.py` previously TIMED OUT because the
  pipeline summary `QMessageBox` blocked the event loop — dialogs are now
  stubbed offscreen (as the screenshot script already did); result reads
  moved to the dict payload keys (`classification.modulation`,
  `symbol_rate.symbol_rate_hz`). `probe_win.py` likewise reads
  `symbol_rate_hz` now. BER is None in these probes by design: no
  `.reference.npz` sidecar accompanies `gui_smoke.wav`.
- **Test recalibration** (`tests/test_end_to_end_qpsk_full.py`): the
  normalized-preamble-error gate moved 0.25 → 0.30 with an in-test comment.
  Root cause (probed empirically, scratch scripts, since removed): session-4
  `detection/detector.py::compute_spectrum` moved from a full-length
  rectangular FFT to Welch PSD averaging (a deliberate improvement: stable
  noise floor). Estimated candidate bandwidth widened (~170 → ~211 Hz on the
  e2e capture), widening the isolation filter and raising the measured
  preamble error 0.1689 → 0.2576. The binding end-to-end property (BER == 0
  through the real `demodulate_signal` path, correct 270° phase fold, 87%
  confidence) is unchanged and still asserted; verified both detector paths
  produce BER 0.000000 (errors 0/2000).
- Full suite: **181 passed**.

### 2026-09-22 — Session 3: CI, packaging, batch, advanced mods, negative tests (COMPLETE)

- **CI**: `.github/workflows/ci.yml` — pytest matrix (ubuntu/windows × py3.10/3.12).
- **Packaging**: repo-root `pyproject.toml` (package moved; prototype/pyproject removed);
  verified wheel build → clean-venv install → `spectra` CLI on PATH → 136 tests
  green against installed package; `.packaging-venv` cleaned up afterwards.
- **Multi-signal batch**: `pipeline_batch.py` — analyze every candidate + timeline;
  `tests/test_pipeline_batch.py`.
- **8-PSK**: `modulation/digital.py` kernels (psk8/ook decisions + demodulators);
  wired into `demodulation/demodulator.py` dispatch, pipeline `_demodulate`,
  `_modulation_order` (order 8 → 8th-power carrier recovery), OOK timing-only sync.
- **8-PSK phase estimator fixed**: old `(angle(mean8)-pi)/8` was biased (could
  never return positive phase) and the candidate-loop tie-break was provably
  pi/4-invariant (all 8 folds scored identically — line 120 subtracted complex
  ideal points from angles instead of ideal angles). Replaced with unbiased
  `angle(-mean(s**8))/8`; residual pi/4 fold documented as fundamental.
- **OOK classifier fixed**: envelope rule (cv≥0.5, p10/p90≤0.25, bimodality≥1.2)
  before BPSK; constellation-stage OOK via one-sided line projections; documented
  DC-centered-OOK≡BPSK identity.
- **8-PSK constellation rule**: magnitude_cv < 0.18 separates PSK from 16-QAM;
  circular phase-cluster count mod 90° (median-normalized threshold, dilation,
  wraparound) gives QPSK=1 vs 8-PSK=2 clusters. Verified on phase-locked
  symbols through the real sync chain; confidence scoring added.
- **BFSK fractional-sps fix**: demodulator resamples to integer sps grid.
- **Negative/property tests** (`tests/test_negative_property.py`, 27 tests):
  missing/empty/truncated/offset-error files, WAV zero/NaN/header failures,
  Signal NaN/Inf/rate validation, resample identity/ratio, Welch nonneg + tone
  location, STFT inverse unit error, DC estimate, IQ-correction energy,
  channel identity + SNR, Hamming/CRC properties, tone detection.
- **CLI e2e verified** with 8-PSK stereo WAV: 8-PSK 80.4% (fine stage),
  rate 1200.0 Hz (true 1200), 1501 symbols demodulated.
- Full suite: **181 passed**.

### 2026-09-22 — Session 2: pipeline, FEC, channel, CLI, reporting, docs (COMPLETE)

Built and verified end-to-end:
- Fixed `test_synchronizer` regression (synchronizer now estimates symbol rate).
- Symbol-rate: added delay-multiply estimator (primary, handles constant-envelope)
  with |x|^2 cross-check; FSK run-length estimator w/ single-run refinement
  (measured 100.07 vs true 100 baud).
- Classifier: added instantaneous-frequency bimodality feature → RRC-QPSK no
  longer mislabeled BFSK; existing classification tests still pass.
- FEC: CRC16/32, Hamming(7,4) (fixed transposed G matrix), repetition3,
  K=7 conv+Viterbi (fixed traceback: predecessor states), interleaver, registry.
- Channel simulator with ground truth (12 impairments).
- io/loaders.py acquisition layer (all raw dtypes/endianness/order, sidecars,
  streaming, save) + io/__init__.
- dsp/ engine (spectral, filters, baseband, correlation). Fixed inverse_stft
  (scipy input_onesided=False for complex, no ifftshift) and IQ phase-imbalance
  estimator (arcsin).
- pipeline.py with provenance + graceful degradation; BER with BPSK polarity and
  QPSK rotation search (applied correction reported).
- reporting/export.py (JSON/CSV/HTML), cli.py (9 subcommands), benchmarking/snr_sweep.py.
- core/config.py (presets), core/exceptions.py, core/provenance.py, __version__=2.1.0.
- Fixed import convention: all new modules use prototype.* absolute imports
  (top-level `io` import shadowed stdlib when run from repo root).
- 121 new tests; full suite 136 passed. CLI validate/report/benchmark verified
  against real WAV files. pyproject fixed (console script + package list).
- Docs: README, CHANGELOG, docs/ARCHITECTURE.md, docs/USER_GUIDE.md, docs/VALIDATION.md.

Verified numbers (synthetic, honest): QPSK e2e 99.4% conf, rate 1000.24 Hz
(true 1000), BER 0.0; BFSK e2e BER 0.005; QPSK SNR sweep OK ≥20 dB.

### 2026-09-22 — Milestone 0/1: audit + foundations (agent session 1)

Audited all 69 Python files. Baseline: 14 passed / 1 failed
(`tests/test_synchronizer.py` used an older API).

Implemented:
- `core/synchronizer.py`: `synchronize_signal(signal, symbol_rate=None, ...)` now estimates
  symbol rate (cyclostationary) when not supplied; added `estimated_symbol_rate` field.
- `core/exceptions.py`, `core/config.py` (quick/balanced/deep presets), `core/provenance.py`.
- `io/loaders.py`: WAV (via scipy.io.wavfile incl. float32), raw IQ (complex64/128,
  int8/16/32, float32/64, either endianness, I/Q order), separate I/Q files, JSON sidecars,
  `stream_raw_iq()` chunked reader, `save_iq()` round-trip.
- `dsp/`: `spectral.py` (Welch PSD two-sided, STFT/spectrogram + inverse), `filters.py`
  (FIR kaiser designs, IIR SOS, polyphase resampling, RRC delegate), `baseband.py`
  (analytic signal, envelope, inst. phase/freq, mixing, blind IQ-imbalance correction,
  clipping detection), `correlation.py` (xcorr, normalized correlation, sync-word finder).
- `simulation/channel.py`: full impairment chain with ground truth.
- `fec/`: CRC-16/CCITT + CRC-32, Hamming(7,4), repetition, K=7 r=1/2 convolutional
  encoder + hard-decision Viterbi, block interleaver, registry.
- `parameters/symbol_rate.py`: added `estimate_symbol_rate_fsk()` (tone separation +
  run-length method).
- `pipeline.py`: `analyze_capture()`/`analyze_samples()` — full chain with graceful
  per-stage warnings, BPSK polarity-aware and QPSK 4-rotation BER, provenance manifest,
  `to_json()`.
- `reporting/export.py`: JSON/CSV/HTML report export.
- `cli.py`: `spectra` subcommands; `pyproject.toml` fixed (console script, packaging where=..).
- `benchmarking/snr_sweep.py`: SNR sweep harness.
- New tests: `tests/test_io_loaders.py, test_dsp.py, test_channel.py, test_fec.py,
  test_pipeline.py, test_reporting.py, test_cli.py, test_benchmark.py`.
- Docs: README.md, CHANGELOG.md, docs/ARCHITECTURE.md.

Verified: full pytest suite green (see CHANGELOG for the count at last run).

### Session 10 — ML subsystem (2026-09-24)

The user supplied `modulation_cnn.pkl` (15.9 MB, Desktop). Findings
and decisions, all verified:

- **Static audit, never unpickled**: pickletools opcode walk showed a
  Keras 3 `Sequential` (`keras.src.models.sequential.Sequential` +
  embedded `.keras` ZIP = config.json + model.weights.h5). Architecture:
  input (512, 2) IQ → Conv1D(64,k7)+BN+MaxPool → ×(128,k5) → ×(256,k3)
  → Dense(256) → Dropout → Dense(16) softmax.
- **The shipped network is UNTRAINED**: optimizer/vars/0 (iteration
  counter) = 0; every BN γ≡1, β≡0, mean≡0, var≡1; biases exactly 0;
  kernel ranges equal Glorot bounds. Hence near-uniform softmax
  (~0.075 vs 1/16=0.0625) on any input — no preprocessing choice could
  fix that. Decision: keep the architecture, train in-project on
  synthetic data (repo ethos) rather than ship fake intelligence.
- **NumPy-only runtime** (`ml/cnn.py`): converter folds BN into conv
  (self-check vs textbook forward: 1.5e-08); Keras MaxPool1D(2) floor
  semantics (drop odd tail) was caught by a shape check — [::2] would
  have been wrong (15616 vs 15872 flatten width).
- **Gradient-checked trainer** (`ml/train.py`): analytic backprop
  (conv via strided windows + tensordot, argmax-masked maxpool,
  frozen-stats BN) verified by finite differences to 1.9e-08 before
  every run. Dataset (`ml/dataset.py`): 16 classes (BPSK…AM, Noise)
  synthesized with repo RRC + `simulation.channel`, randomized
  SNR/CFO/phase/rate/rolloff. First artifact: 640 frames × 10 epochs →
  **31.2% holdout accuracy, 5× chance**, loss 8.0→1.44 (still
  improving — more data/epochs is the knob).
- **Wiring**: `AnalysisConfig.ml` (default off) → provenance step
  `ml_classification` → `AnalysisResult.ml` payload; GUI toggle
  `ML assist (CNN)` + `ML Prediction` row + summary-dialog line;
  batch gets it via shared config; runtime prefers
  `modulation_cnn_trained.npz` over the shipped conversion.
- Open items: real label map (neutral class_00..15 until someone
  provides the training classes), confidence calibration, longer
  training runs, CLI `spectra ml-train` wrapper.

### Next steps (recommended order)

1. DONE (session 7): GUI modernized — QThread worker, V2 pipeline,
   batch, FEC, export/provenance. Session 8 added `docs/GUI_USER_GUIDE.md`
   and fixed the Open-WAV NameError (undefined `stats`), the missing
   `prototype/main.py` entry point, and the new `spectra gui` subcommand.
   Session 9 fixed Isolate Selected (V1 ndarray call vs V2 `Signal` API)
   and routed the selected-signal 16-QAM path through the V2 QAM chain
   (BER 0 verified through the GUI isolate → analyze-selected flow).
2. GUI modernization follow-ups: modularize `gui/window.py`, wire it to
   `pipeline.analyze_samples` + `pipeline_batch.analyze_all_candidates`
   (V1 legacy per-modulation path remains in the Analyze-Selected flow).
3. Tracking loops (Gardner TED, Costas) for low-SNR and fractional-timing
   robustness.
3. Protocol/frame layer on top of `dsp.correlation.find_sync_word` (config-driven
   preamble/sync-word/CRC definitions) + FEC wiring into the pipeline.
4. MSK/GMSK, 4-FSK, 64-QAM kernels + classifier rules.
5. ML subsystem only with synthetic-dataset pipeline + leakage controls.
6. 16-QAM fine-stage e2e through `demodulate_signal(synchronized=True)`
   currently requires `samples_per_symbol` — thread it from symbol_rate
   summary (one-line fix candidate).