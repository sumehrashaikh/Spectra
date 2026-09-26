# Spectra Validation Report

**Date:** 2026-09-22
**Software version:** 2.1.0
**Environment:** Windows, Python 3.13.1, numpy/scipy (pip-installed), CPU only
**Test command:** `python -m pytest -q` (136 passed)

## 1. What "validated" means here

All validation in this document was executed against **synthetic
signals** generated with fixed seeds, where the transmitted bits and
channel parameters are known exactly. This validates the *DSP
processing chain*: algorithms, estimators, decoders, and their
confidence reporting.

It does **not** validate: performance on real RF captures, behavior
with real receiver front-ends (AGC, real ADC quantization, real clock
drift), any specific SDR hardware, regulatory/spectral-measurement
accuracy, or suitability for any safety-, mission-, or
 revenue-critical use.

## 2. Validation matrix (executed)

| # | Area | Test | Expected | Actual | Status |
|---|---|---|---|---|---|
| 1 | Regression | `tests/test_end_to_end_qpsk_full.py` | BER 0 through preamble-based flow | BER 0 | PASS |
| 2 | Regression | `tests/test_end_to_end_bpsk_full.py` | BER 0 via `demodulate_signal()` | BER 0 | PASS |
| 3 | Regression | `tests/test_classification_pipeline.py` | BPSK/QPSK/16-QAM/BFSK correct; tone → Unknown | as expected | PASS |
| 4 | Pipeline | `tests/test_pipeline.py::test_qpsk_end_to_end` | QPSK, rate ≈1000 Hz, BER < 0.01 | 99.4% conf, 1000.24 Hz, BER 0 | PASS |
| 5 | Pipeline | `test_bpsk_end_to_end` | BPSK, BER < 0.01 | BPSK, BER 0 | PASS |
| 6 | Pipeline | `test_bfsk_end_to_end` | BFSK, BER < 0.05 | BFSK, BER 0.005 | PASS |
| 7 | Pipeline | `test_no_candidates_detected` | graceful empty result + warning | as expected | PASS |
| 8 | Pipeline | `test_second_candidate_analysis` | candidate 1 (weaker tone) selected | center 2500 Hz | PASS |
| 9 | IO | `tests/test_io_loaders.py` (17 tests) | round-trips ±quantization bound; errors raised | as expected | PASS |
| 10 | DSP | `tests/test_dsp.py` (23 tests) | PSD peaks, filter rejection < −40 dB, exact IQ correction, sync-word find | as expected | PASS |
| 11 | Channel | `tests/test_channel.py` (16 tests) | each impairment matches truth; AWGN SNR ±0.5 dB | as expected | PASS |
| 12 | FEC | `tests/test_fec.py` (25 tests) | CRC detect, Hamming corrects every single-bit error, conv. corrects 32 scattered errors, interleave round-trip | as expected | PASS |
| 13 | Symbol rate | `test_rectangular_bpsk_constant_envelope` etc. | rate ±1% for RRC-BPSK, rect-BPSK, QPSK, BFSK | max err 0.07% | PASS |
| 14 | Config/Provenance/Reporting | `tests/test_core_modules.py` (20 tests) | serialization, step statuses, hashes | as expected | PASS |
| 15 | CLI | `spectra validate` (run manually) | QPSK + BER < 0.05 on synthetic WAV | passed: true | PASS |
| 16 | CLI | `spectra report --html --json` | files produced | 8.0 KB HTML, 4.7 KB JSON | PASS |
| 17 | Benchmark | `spectra benchmark --modulation QPSK --snr-db 30,20,10,0` | documented behavior curve | correct ≥20 dB; degraded 10 dB; Unknown 0 dB (honest) | PASS (documented) |
| 18 | Protocol / frame layer | `tests/test_protocol.py` (6) + `tests/test_protocol_pipeline.py` (4) | sync word found + payload recovered; no-sync → Unknown (never guessed); JSON/CSV export serializes the field | as expected | PASS |

## 3. Benchmarks (synthetic AWGN, 256 symbols, RRC α=0.35, fs=8 kHz)

| SNR | Detected | Classification | Symbol-rate error | BER |
|---|---|---|---|---|
| 30 dB | yes | QPSK (correct) | 0.49 Hz | 0.0 |
| 20 dB | yes | QPSK (correct) | 0.49 Hz | 0.0 |
| 10 dB | yes | QPSK (correct) | 874.9 Hz (failed) | 0.34 |
| 0 dB | yes | Unknown | n/a | n/a |

Interpretation: the chain is reliable at high SNR and fails honestly
below ~10 dB for 8 sps RRC-QPSK at this capture length. Failure modes
are recorded, not masked. Longer captures and lower rolloff improve
the low-SNR floor (not yet systematically tuned — see limitations).

## 4. Known limitations

1. **Synthetic-only validation.** No real-world captures have been
   analyzed. Real front-ends introduce AGC transients, spurious
   tones, real clock drift, and non-Gaussian noise not simulated here.
2. **Low-SNR classification.** Reliable classification currently
   requires roughly ≥ 15–20 dB SNR for 8 sps pulse-shaped signals;
   the deep mode lowers thresholds but does not add new estimators.
3. **BFSK demodulation drift.** `demodulate_bfsk` slices symbols at
   integer samples-per-symbol; with a fractional rate estimate,
   boundary drift accumulates on long captures (a warning is emitted).
4. **Timing recovery granularity.** Timing offset is estimated at
   integer sample resolution; fractional timing offsets are not
   tracked by a loop (no Gardner/Costas tracking yet — snapshot
   estimators only).
5. **Classifier scope.** BPSK/QPSK/16-QAM/BFSK/Unknown only.
   8-PSK, MSK/GMSK, OQPSK, 64/256-QAM, AM/FM are not implemented.
6. **No frame/protocol layer.** Preamble correlation exists
   (`dsp/correlation.find_sync_word`), but no protocol parser.
7. **No ML subsystem.** Deterministic DSP only, by design.
8. **GUI scalability.** The GUI runs analysis on the UI thread for
   some operations; large captures should use the CLI.
9. **Windows-first.** CI is not configured; Linux/macOS are expected
   to work (pure Python + numpy/scipy) but are untested.

## 5. What would be required to extend validation

- Real captures with known truth (loopback recordings, calibrated
  signal generators) for end-to-end validation.
- Hardware-in-the-loop tests with common SDRs (RTL-SDR, HackRF) to
  validate the acquisition path and real quantization behavior.
- Statistical benchmarking: multiple seeds per point, confidence
  intervals, ROC curves for detection versus SNR.
- Interoperability checks against reference tools (e.g., GNU Radio
  generated captures) for format and DSP cross-validation.

## 6. Explicit non-claims

Spectra is **not**: certified by any authority, flight-qualified,
mission-approved, a replacement for calibrated spectrum-analyzer
instrumentation, or validated against real-world satellite/aerospace
signals. Do not use it as evidence of regulatory compliance.
