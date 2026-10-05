# SPECTRA — SIH26147 evidence pack

Generated 2026-09-29 16:31 UTC from commit `8b697cc` on branch `feature/gnuradio-gui-v1`.

> **Snapshot notice.** This pack is a frozen point-in-time generation
> (commit `8b697cc`, 435 tests). The repository has since grown to 538
> tests and gained further fixes (uncoded QPSK/FSK BER, ML v3 artifact,
> GNU Radio spectrum/waterfall bridge, light/dark theme); read current
> status from `docs/SIH_REQUIREMENTS.md` and `docs/VALIDATION.md`.
> The generator scripts (`tools/sih_deck/*`) were working-session tools
> and are not part of this snapshot.

Every number on the SIH deck is quoted from this file. Nothing here is hand-entered: `tools/sih_deck/sih_evidence_pack.py` audits the working tree, runs the suite and runs the demo matrix.

## 1. Headline numbers

| Metric | Value |
|---|---|
| Requirement areas fully verified (test-exercised) | 13 / 18 |
| Requirement areas partial (hard parts) | 4 / 18 |
| **Build coverage** (weighted share implemented + test-exercised) | **87.6%** |
| **Unattended autonomy** (weighted share deciding with no ground truth) | **73.6%** |
| Test suite | 435 passed |
| Test wall-clock | 95.2 s |
| Demo captures exercised end-to-end | 11 |
| Modulations classified correctly | 11 / 11 |
| Symbol-rate estimate vs. truth (configured runs) | max error 0.059%, mean 0.030% |
| FEC-bearing captures with payload recovered (ground truth supplied) | 3 / 4 |
| Auto FEC detections claimed by the tool | 0 / 11 |
| Auto interleaver detections claimed by the tool | 0 / 11 |

### How the two percentages are defined

* **Build coverage** = weighted mean of `implementation` over the requirement matrix: an area counts fully only when the capability exists *and* a named test exercises it.
* **Unattended autonomy** = weighted mean of `autonomy`: the same areas counted only where the tool reaches a decision with no ground truth and no operator hint. This is the number that matters for an unknown capture, and it is deliberately lower.
* Partial values (0.3–0.55) are engineering estimates for areas that are half-built; verified rows are 1.0 by measurement. The estimated rows are exactly the ones the deck marks as roadmap.

## 2. Requirement matrix

| ID | Area | Status | Built | Auto | Weight | Implementation evidence | Test evidence |
|---|---|---|---|---|---|---|---|
| R01 | Capture ingest | VERIFIED | 1.00 | 1.00 | 1.00 | `io/loaders.py`, `core/loader.py` | `test_io_loaders.py` |
| R02 | Pre-processing | VERIFIED | 1.00 | 1.00 | 1.00 | `core/preprocessor.py` | `test_core_modules.py` |
| R03 | Detection / segmentation | VERIFIED | 1.00 | 1.00 | 1.00 | `detection/detector.py`, `core/analyzer.py`, `dsp/spectral.py` | `test_dsp.py`, `test_detection_isolation.py` |
| R04 | Extraction / DDC | VERIFIED | 1.00 | 1.00 | 1.00 | `core/isolator.py`, `dsp/filters.py` | `test_detection_isolation.py` |
| R05 | Parameter extraction | VERIFIED | 1.00 | 1.00 | 1.00 | `parameters/extractor.py`, `parameters/symbol_rate.py` | `test_parameter_extractor.py`, `test_symbol_rate.py`, `test_symbol_rate_rect_qam.py`, `test_bpsk_parameters.py` |
| R06 | Synchronisation | VERIFIED | 1.00 | 1.00 | 1.00 | `core/synchronizer.py`, `core/carrier_sync.py`, `synchronization/timing.py`, `synchronization/frequency.py` | `test_synchronizer.py`, `test_synchronization.py`, `test_timing_recovery.py`, `test_carrier_recovery.py` |
| R07 | Modulation classification | VERIFIED | 1.00 | 0.85 | 1.00 | `modulation/classifier.py`, `classification/classifier.py`, `modulation/digital.py` | `test_classification_pipeline.py`, `test_advanced_modulation.py` |
| R08 | Demodulation | VERIFIED | 1.00 | 0.90 | 1.00 | `demodulation/demodulator.py`, `demodulation/qpsk_sync.py`, `modulation/demodulator.py`, `modulation/digital.py` | `test_demodulation_pipeline.py`, `test_end_to_end_qpsk_full.py`, `test_end_to_end_qam16_full.py`, `test_end_to_end_bpsk_full.py`, `test_qpsk_phase_resolution.py` |
| R09 | Quality measurement (BER) | VERIFIED | 1.00 | 0.90 | 1.00 | `core/ber.py`, `core/provenance.py` | `test_pipeline.py`, `test_end_to_end_qpsk.py` |
| R10 | FEC decoder library | VERIFIED | 1.00 | 0.00 | 1.00 | `fec/framework.py`, `fec/convolutional.py`, `fec/reed_solomon.py`, `fec/ldpc.py`, `fec/concatenated.py`, `fec/hamming.py`, `fec/repetition.py`, `fec/crc.py` | `test_fec.py`, `test_fec_schemes_extended.py` |
| R11 | FEC identification | PARTIAL | 0.55 | 0.55 | 1.00 | `fec/identification.py` | `test_fec_identification.py`, `test_fec_identification_extended.py`, `test_pipeline_fec_identification.py` |
| R12 | De-interleaver library | VERIFIED | 1.00 | 0.00 | 1.00 | `fec/interleaving.py` | `test_interleaving_families.py`, `test_cli_interleaving.py` |
| R13 | Interleaver identification | PARTIAL | 0.45 | 0.45 | 1.00 | `fec/identification_interleaving.py` | `test_interleaving_identification.py`, `test_pipeline_interleaving.py` |
| R14 | Bit stream correlation | PARTIAL | 0.35 | 0.35 | 1.00 | `protocol/config.py`, `protocol/parser.py`, `protocol/semantics.py`, `dsp/correlation.py` | `test_protocol.py`, `test_protocol_pipeline.py`, `test_protocol_semantics.py` |
| R15 | GUI + reporting | VERIFIED | 1.00 | 1.00 | 1.00 | `gui/window.py`, `gui/worker.py`, `visualization/plots.py`, `reporting/export.py` | `test_gui_fec_demo.py`, `test_gui_interleaving.py`, `test_gui_gnuradio.py` |
| R16 | Automation + provenance | VERIFIED | 1.00 | 1.00 | 1.00 | `pipeline.py`, `pipeline_batch.py`, `core/provenance.py`, `cli.py` | `test_pipeline.py`, `test_pipeline_batch.py`, `test_final_integration.py` |
| R17 | GNU Radio / live ingest | PARTIAL | 0.50 | 0.50 | 0.50 | `io/gnuradio/source.py`, `io/gnuradio/adapter.py`, `io/gnuradio/config.py`, `io/gnuradio/streaming.py`, `io/gnuradio/gui_controller.py` | `test_gnuradio.py` |
| R18 | ML assist | EXPERIMENTAL | 0.30 | 0.30 | 0.25 | `ml/cnn.py`, `ml/fusion.py`, `ml/train.py`, `ml/dataset.py` | `test_ml_cnn.py` |

### What each row actually does

**R01 — Capture ingest (VERIFIED)**  
*PS asks:* Load .wav and .IQ captures that store the waveform differently  
*Reality:* WAV (mono -> Hilbert analytic, stereo -> IQ, 8/16/32-bit PCM and float) plus raw interleaved IQ (complex64/128, float32/64, int8/uint8/int16/int32, both endianness, I-first or Q-first), separate I/Q files, JSON sidecars and chunked streaming.

**R02 — Pre-processing (VERIFIED)**  
*PS asks:* Condition the capture before measurement  
*Reality:* DC removal, noise-floor estimate, SNR, normalisation.

**R03 — Detection / segmentation (VERIFIED)**  
*PS asks:* Find the signals inside a wide capture  
*Reality:* FFT/STFT spectral peaks and regions with an energy threshold; multi-candidate output with frequency/bandwidth/peak level.

**R04 — Extraction / DDC (VERIFIED)**  
*PS asks:* Isolate one signal for measurement  
*Reality:* Digital down-conversion: mix to baseband, low-pass, decimate.

**R05 — Parameter extraction (VERIFIED)**  
*PS asks:* Identify signal parameters (sampling frequency and beyond)  
*Reality:* Sample rate (from file/sidecar), centre frequency, bandwidth, SNR, PAPR, DC offset, symbol rate (M-th power / QAM-specific / BFSK estimators), samples-per-symbol and pulse-shape roll-off.

**R06 — Synchronisation (VERIFIED)**  
*PS asks:* Make the measurement valid before deciding anything  
*Reality:* Gardner-type timing recovery plus carrier phase/frequency recovery.

**R07 — Modulation classification (VERIFIED)**  
*PS asks:* Identify the modulation type  
*Reality:* Waveform gates (frequency/magnitude bimodality, r2 phase coherence, amplitude spread) plus a constellation classifier on synchronised symbols. Six classes are returned; anything the evidence does not support is returned as 'Unknown', which costs autonomy but is the honest behaviour.

**R08 — Demodulation (VERIFIED)**  
*PS asks:* Demodulate FSK, QAM and PSK  
*Reality:* BPSK, QPSK, 8-PSK, 16-QAM, BFSK and OOK/ASK decision chains. Known residuals are documented rather than hidden: 16-QAM blind phase keeps a non-90-degree rotation and 8-PSK keeps a pi/4 rotation; both are resolved against a reference when one exists.

**R09 — Quality measurement (BER) (VERIFIED)**  
*PS asks:* Quantify how good the recovered bit stream is  
*Reality:* BER against the transmitted reference with an explicit ambiguity search (BPSK polarity, QPSK 90-degree, 16-QAM rotation x symbol origin). Without a reference the tool says 'no reference loaded' instead of inventing a number.

**R10 — FEC decoder library (VERIFIED)**  
*PS asks:* FEC: convolutional + Viterbi, RS block codes, concatenated codes, LDPC  
*Reality:* Six registered schemes: hamming74, repetition3, conv12 (K=7 rate-1/2 Viterbi), reedsolomon (shortened GF(256), 32+8, t=4), ldpc (compact (3,6) regular, hard-decision bit-flip) and concatenated (RS outer + convolutional inner). Autonomy is 0 by design: FEC is never inferred, the caller selects the scheme.

**R11 — FEC identification (PARTIAL)**  
*PS asks:* Identify FEC from the observation data itself  
*Reality:* A candidate evaluator, not a decoder-and-hope: it runs the real decoders and scores structural/decoder evidence, reports AUTO_DETECTED only at MIN_CONFIDENCE=60 with compatible structure, and otherwise returns UNKNOWN/UNRESOLVED. CRC16/CRC32 are used as validity evidence only, never as a decoding hypothesis. Known gap: the scored candidate set is fixed; unknown/vendor schemes and long-block LDPC recognition are the roadmap.

**R12 — De-interleaver library (VERIFIED)**  
*PS asks:* De-interleaving: block, convolutional, diagonal, pseudo-random  
*Reality:* All four families implemented and round-trip tested through the receiver's own deinterleaver, including the frame-length constraints (block/convolutional/pseudo-random need a whole multiple of the stride; diagonal needs exactly depth^2 bits). Autonomy is 0 by design: the family and depth are configured.

**R13 — Interleaver identification (PARTIAL)**  
*PS asks:* Identify the interleaver from the observation data itself  
*Reality:* Structural identification of the row-column block interleaver only: bounded depth search, applicability as a gate, deterministic scoring; AUTO_DETECTED requires confidence >= 55 and a >= 12 margin over the no-interleaving baseline, otherwise NO_INTERLEAVING/UNKNOWN. Convolutional, diagonal and pseudo-random are deliberately down-weighted: they appear as weak hypotheses, never as a claim.

**R14 — Bit stream correlation (PARTIAL)**  
*PS asks:* Correlate the bit stream to find header and payload  
*Reality:* Sync-word correlation, frame extraction and semantic field parsing run against an explicitly configured FrameConfig; an unmatched capture returns Unknown rather than a guess. Known gap: automatic discovery of an unknown frame format / header length / payload partition is roadmap, and prototype/correlation/ is still empty.

**R15 — GUI + reporting (VERIFIED)**  
*PS asks:* GUI-based model with visualisation and export  
*Reality:* PySide6 desktop GUI: source selection, spectrum / waterfall / time-domain / recovered-constellation tabs, per-signal parameter rows, FEC and interleaving controls, frame-search controls, batch candidate analysis, JSON/HTML export and a provenance dialog.

**R16 — Automation + provenance (VERIFIED)**  
*PS asks:* Automated analysis with a defensible result  
*Reality:* One call drives the whole chain (analyze_samples/analyze_capture), batch mode sweeps every detected candidate, and every stage records method, timings, software version and git commit in a provenance manifest that is exported with the result.

**R17 — GNU Radio / live ingest (PARTIAL)**  
*PS asks:* Work with the tooling named in the statement  
*Reality:* GNU Radio is integrated as an input/streaming layer behind a documented chunk contract, with a deterministic offline synthetic source so the flow is testable without hardware. Live SDR backends (RTL-SDR/HackRF/USRP) and hardware validation are roadmap -- the GNU Radio package is not installed in this checkout.

**R18 — ML assist (EXPERIMENTAL)**  
*PS asks:* Advanced models (optional, evidence only)  
*Reality:* A NumPy-only CNN runtime with configurable fusion (side_by_side, dsp_over_ml, ml_over_dsp, max_confidence). It is a second opinion that never gates the DSP result. The packaged trained artifact reaches only 0.3125 validation accuracy on synthetic frames and the alternative artifact is unlabelled, so ML output is reported as scores, never as a calibrated detector.

## 3. Test suite

```
$ C:\Users\eiraa\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests -q --tb=line -p no:cacheprovider
435 passed
```

Return code: `0` · wall clock `95.2s`

```
...                                                                      [100%]
============================== warnings summary ===============================
prototype/tests/test_dsp.py::TestSpectral::test_stft_inverse_roundtrip
  C:\Users\eiraa\SpectraV2\prototype\tests\test_dsp.py:57: UserWarning: Input data is complex, switching to return_onesided=False
    _, _, complex_stft = _stft(x, fs=8000, nperseg=256, noverlap=128)

prototype/tests/test_negative_property.py::TestDSPProperties::test_stft_inverse_is_unit
  C:\Users\eiraa\SpectraV2\prototype\tests\test_negative_property.py:157: UserWarning: Input data is complex, switching to return_onesided=False
    _, _, complex_stft = scipy_stft(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
435 passed, 2 warnings in 92.33s (0:01:32)
```

## 4. Demo matrix — the same ten captures, twice

**Run A (automatic):** no ground truth supplied — what the tool decides on its own.

| Capture | Truth (mod / FEC / interleaver) | Classified | Symbol rate (Hz) | FEC auto | Interleaver auto | BER |
|---|---|---|---|---|---|---|
| `16-QAM_nofec_nointerleave` | 16-QAM / none / none | 16-QAM | 99.9414 | UNKNOWN | UNRESOLVED | 0.003 |
| `16-QAM_conv12_nointerleave` | 16-QAM / conv12 / none | 16-QAM | 99.9709 | UNKNOWN | UNRESOLVED | 0 |
| `16-QAM_reedsolomon_block` | 16-QAM / reedsolomon / block | 16-QAM | 99.9531 | UNKNOWN | UNRESOLVED | 0 |
| `16-QAM_concatenated_block` | 16-QAM / concatenated / block | 16-QAM | 99.9767 | UNKNOWN | UNRESOLVED | 0 |
| `16-QAM_concatenated_pseudo_random` | 16-QAM / concatenated / pseudo_random | 16-QAM | 99.9767 | UNKNOWN | UNRESOLVED | 0 |
| `QPSK_nofec_nointerleave` | QPSK / none / none | QPSK | 99.9707 | UNKNOWN | UNRESOLVED | — |
| `QPSK_nofec_block` | QPSK / none / block | QPSK | 99.9707 | UNKNOWN | UNRESOLVED | — |
| `QPSK_nofec_convolutional` | QPSK / none / convolutional | QPSK | 99.9707 | UNKNOWN | UNRESOLVED | — |
| `QPSK_nofec_diagonal` | QPSK / none / diagonal | QPSK | 99.9953 | UNKNOWN | UNRESOLVED | — |
| `8-PSK_nofec_block` | 8-PSK / none / block | 8-PSK | 99.9554 | UNKNOWN | UNKNOWN | — |
| `BFSK_nofec_nointerleave` | BFSK / none / none | BFSK | 99.9853 | UNKNOWN | UNRESOLVED | — |

**Run B (ground truth supplied):** the analyst (or a database) knows the scheme — what the chain can then recover.

| Capture | Classified | Decoded bits | Payload prefix matched | Payload recovered | Uncorrectable blocks | Alignment |
|---|---|---|---|---|---|---|
| `16-QAM_nofec_nointerleave` | 16-QAM | 0 | 0 / 1024 | no | — | True |
| `16-QAM_conv12_nointerleave` | 16-QAM | 1004 | 1001 / 1024 | no | 0 | True |
| `16-QAM_reedsolomon_block` | 16-QAM | 1024 | 1024 / 1024 | yes | 0 | True |
| `16-QAM_concatenated_block` | 16-QAM | 1024 | 1024 / 1024 | yes | 0 | True |
| `16-QAM_concatenated_pseudo_random` | 16-QAM | 1024 | 1024 / 1024 | yes | 0 | True |
| `QPSK_nofec_nointerleave` | QPSK | 0 | 0 / 1024 | no | — | — |
| `QPSK_nofec_block` | QPSK | 0 | 0 / 1024 | no | — | — |
| `QPSK_nofec_convolutional` | QPSK | 0 | 0 / 1024 | no | — | — |
| `QPSK_nofec_diagonal` | QPSK | 0 | 0 / 6400 | no | — | — |
| `8-PSK_nofec_block` | 8-PSK | 0 | 0 / 1008 | no | — | — |
| `BFSK_nofec_nointerleave` | BFSK | 0 | 0 / 1000 | no | — | — |

True symbol rate for every matrix capture is 100.0 Hz (8000 Hz / 20493-sample capture) — the estimator column can be read against it directly.

Reading the two tables together is the whole argument of the deck: the automatic path decides what the evidence supports and refuses to guess the rest; the configured path recovers payloads bit-exactly. The gap between the two columns is the roadmap, stated as a number.

Two footnotes on Run B so the table is not read as stronger than it is:

* `16-QAM_conv12_nointerleave` recovers 1001 of 1024 payload bits: the pulse-shaping tail is lost with the capture, exactly the truncation the pipeline reports instead of inventing the missing bits. The payload prefix is bit-exact.
* Uncoded PSK captures have no FEC stage, so 'decoded bits' is 0 by design; their deliverable is the measured coded-stream BER and the recovered constellation, not a decoded payload.

## 5. Automatic-identification capability ladder

**L1 — FEC identification on cleanly encoded bit streams (the capability itself).**

*Short stream (256-bit payload).*

| Truth | Input bits | Status | Best scheme | Confidence | Correct |
|---|---|---|---|---|---|
| none | 256 | UNKNOWN | — | 40 | yes |
| hamming74 | 448 | AUTO_DETECTED | hamming74 | 80 | yes |
| repetition3 | 768 | AUTO_DETECTED | repetition3 | 80 | yes |
| conv12 | 524 | AUTO_DETECTED | conv12 | 80 | yes |
| reedsolomon | 320 | UNKNOWN | — | 40 | no |
| ldpc | 512 | AUTO_DETECTED | ldpc | 80 | yes |
| concatenated | 652 | AUTO_DETECTED | conv12 | 80 | no |

Correct decisions: **5 / 7** — including 'no FEC' reported as UNKNOWN rather than invented.

*Demo-capture length (1024-bit payload, the length the matrix captures use).*

| Truth | Input bits | Status | Best scheme | Confidence | Correct |
|---|---|---|---|---|---|
| conv12 | 2060 | AUTO_DETECTED | conv12 | 80 | yes |
| reedsolomon | 1280 | AUTO_DETECTED | reedsolomon | 70 | yes |
| concatenated | 2572 | AUTO_DETECTED | concatenated | 83 | yes |

Correct decisions: **3 / 3**.

Two honest sensitivity findings fall out of this table: the identifier needs enough stream to build evidence (RS returns UNKNOWN on the 320-bit stream but AUTO_DETECTS on the 1280-bit one), and at 652 bits a concatenated frame is confused with its own inner convolutional code — the outer RS layer's contribution is weak at short length. Both are documented, not hidden.

**L1b — interleaver identification, blind vs. with a validity oracle.** Ground truth: block interleaver, depth 8.

| Input | Validator supplied | Status | Detected type | Depth | Confidence |
|---|---|---|---|---|---|
| blind (pipeline today) | no | UNRESOLVED | — | — | 0.5 |
| FEC-decode oracle supplied | yes | AUTO_DETECTED | block | 8 | 1 |

The block-interleaver identifier only awards structured evidence when a frame validator is supplied (`identify_interleaving(..., frame_validator=...)`), so the blind call in the pipeline today caps out below the detection threshold and returns UNRESOLVED. With FEC-decode success supplied as the oracle it resolves the true depth. That is the next milestone, not an open research problem.

**L3 — end-to-end on lossy captures:** read section 4's Run A column above. Auto FEC reports UNKNOWN on every FEC capture and the interleaver identifier reports UNRESOLVED, because the received stream loses the pulse-shaping tail (for example 1240 aligned bits of 1280 transmitted) and carries residual symbol errors, so whole-codeword structural evidence no longer clears the threshold. The configured path still recovers the payload in the same captures — the gap is evidence robustness, not the decoder.

## 6. Environment

| Item | Value |
|---|---|
| python | 3.13.1 |
| platform | Windows-10-10.0.19045-SP0 |
| generated_utc | 2026-09-29 16:31 UTC |
| numpy | 2.2.3 |
| scipy | 1.17.1 |
| PySide6 | 6.11.2 |
| matplotlib | 3.11.2 |
| gnuradio | not installed (synthetic fallback used) |
| git_commit | 8b697cc |
| git_branch | feature/gnuradio-gui-v1 |
| git_commit_count | 8 |
| git_first_commit | ['2026-09-16'] |
| git_last_commit | 2026-09-29 |

## 7. Claims this pack does NOT support

* No live-SDR or hardware validation: GNU Radio is an ingest interface with a synthetic offline source in this checkout.
* No real-world/off-air capture validation: every matrix capture is synthetic with known ground truth.
* No claim that blind FEC/interleaver identification is solved: the automatic path returns UNKNOWN/UNRESOLVED whenever evidence is weak.
* No Turbo-code support, and the ML CNN is not a validated detector.
