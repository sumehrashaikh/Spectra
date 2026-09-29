# Spectra — SIH-147 Requirement Matrix

**Date:** 2026-09-29
**Branch:** `feature/gnuradio-gui-v1`
**Software version:** 2.1.0
**Test command:** `QT_QPA_PLATFORM=offscreen python -m pytest -q` → **433 passed**

Legend: ✅ working and tested · 🟡 implemented but limited / manual only · ❌ not implemented

This matrix audits each SIH-147 requirement against the code that actually
exists. It deliberately does **not** claim reference-free alignment,
non-block automatic interleaving identification, or any real-world capture
validation — none of those are implemented.

## 1. Signal chain

| # | Requirement | Status | Evidence / notes |
|---|---|---|---|
| 1 | Spectral detection / candidate isolation | ✅ | `tests/test_detection_isolation.py`, `tests/test_pipeline.py` |
| 2 | Modulation classification — PSK | ✅ | BPSK/QPSK/8-PSK; `tests/test_classification_pipeline.py`, `test_advanced_modulation.py` |
| 3 | Modulation classification — QAM | ✅ | 16-QAM; `tests/test_end_to_end_qam16_full.py` |
| 4 | Modulation classification — FSK | ✅ detect · 🟡 demod | BFSK is classified; `demodulate_bfsk` needs the two tone frequencies (auto-found by the pipeline, but only for BFSK). No 4FSK/GMSK. |
| 5 | Symbol-rate estimation | ✅ | `tests/test_symbol_rate.py` |
| 6 | Carrier + timing synchronization | ✅ | `tests/test_synchronization.py`, `test_carrier_recovery.py` |
| 7 | Demodulation + recovered bitstream | ✅ | `demodulation.received_bits` preserved in every result |
| 8 | BER against a reference | ✅ · 🟡 | Clean for 16-QAM and all coded streams. 🟡: the blind phase/origin fold is **not** resolved for uncoded QPSK/8-PSK, so those captures honestly report "no reference" rather than a misleading ~0.5 BER. |

## 2. FEC + interleaving

| # | Requirement | Status | Evidence / notes |
|---|---|---|---|
| 9 | Convolutional FEC | ✅ | K=7 rate-1/2 (171,133); `tests/test_fec.py`, `test_end_to_end_fec_demo.py` |
| 10 | Reed-Solomon FEC | ✅ | `t=4`, 32-byte blocks; clean end-to-end decode |
| 11 | Concatenated FEC | ✅ | inner conv12 + outer RS; clean decode |
| 12 | Additional schemes | ✅ | repetition3, Hamming(7,4), LDPC(8,16), CRC — implemented + tested |
| 13 | Partial-codeword handling | ✅ | whole-codeword trim with reported `trimmed_tail_bits` |
| 14 | Automatic FEC identification | ✅ | Evidence-based (`fec/identification.py`): clean codeword → itself, random bits → UNKNOWN. Limited to the 7 supported schemes. |
| 15 | Manual FEC scheme selection (GUI + CLI) | ✅ | GUI FEC combo (`auto/manual/none` + scheme); CLI `--fec-mode/--fec-scheme` |
| 16 | Block interleaver | ✅ | round-trip + full-chain clean decode (`*.block`) |
| 17 | Convolutional interleaver | ✅ bit-level | round-trip verified; demonstrable in the GUI on uncoded QPSK |
| 18 | Diagonal (square) interleaver | ✅ bit-level | needs exactly `depth²` bits; demo uses depth 80 (6400 bits) |
| 19 | Pseudo-random interleaver | ✅ | round-trip + full-chain clean decode (concatenated + seed 3) |
| 20 | Manual interleaving family/depth (GUI + CLI) | ✅ | GUI interleaving combo + family + depth; CLI `--interleaving-mode/--interleave-family/--interleave-depth` |
| 21 | Automatic interleaving identification | 🟡 **block only** | `fec/identification_interleaving.py` identifies block interleaving from structural evidence. Convolutional / diagonal / pseudo-random have no reliable structural signature here and require manual configuration. |

## 3. Frame, provenance, recovery

| # | Requirement | Status | Evidence / notes |
|---|---|---|---|
| 22 | Sync-word / frame search | ✅ | `protocol/`; GUI "Frame search" control; CLI `--sync-word/--data-bytes` |
| 23 | No-sync → Unknown (never guessed) | ✅ | `tests/test_protocol_semantics.py` |
| 24 | Codeword alignment (blind phase/origin fold) | ✅ 16-QAM · ❌ QPSK/8-PSK | Applied only against a supplied reference; resolves 16-QAM and coded streams, not uncoded QPSK/8-PSK |
| 25 | Recovered bits / information display | ✅ | GUI "Recovered bits" row + `demodulation.fec.decoded_bits` in JSON |
| 26 | Provenance manifest | ✅ | `provenance` in single + batch payloads; GUI Provenance button |
| 27 | Export JSON | ✅ | GUI Export JSON; CLI `--json` |

## 4. GUI

| # | Requirement | Status | Evidence / notes |
|---|---|---|---|
| 28 | WAV / raw-IQ load | ✅ | GUI `open_wav`; raw IQ prompts rate/dtype, honours `*.meta.json` sidecar |
| 29 | Six visualisation tabs render | ✅ | Time Domain · Spectrum · Waterfall/STFT · Constellation · Detection/Results · GNU Radio |
| 30 | Analyze (single + batch) | ✅ | `AnalysisWorker` on a QThread |
| 31 | Batch shows candidate-specific results | ✅ | candidate combo + per-candidate detail (regression fixed) |
| 32 | Manual FEC / interleaving / frame controls reach the backend | ✅ | `tests/test_gui_fec_demo.py`, `test_gui_interleaving.py` |
| 33 | GNU Radio tab + source switching | ✅ synthetic · 🟡 real | Real backend when `gnuradio` is installed (not installed in this environment); falls back to the built-in synthetic source otherwise |
| 34 | ML CNN stage + fusion | ✅ | Fixed this pass: the pipeline called `predict_modulation` without importing it |

## 5. CLI

| # | Requirement | Status | Evidence / notes |
|---|---|---|---|
| 35 | `analyze / detect / classify / parameters / demodulate / report / benchmark / validate / version` | ✅ | all subcommands parse and run |
| 36 | FEC / interleaving flags | ✅ | `--fec-mode/--fec-scheme/--interleaving-mode/--interleave-family/--interleave-depth` |
| 37 | `--sync-word/--data-bytes` | ✅ | Fixed this pass: `analyze` built the `FrameConfig` and then dropped it |
| 38 | `--ml/--ml-fusion/--labels` | ✅ | Fixed this pass: the flags were parsed but never reached the config |
| 39 | `--source gnuradio` | ✅ | Fixed this pass: the branch referenced names it never imported |
| 40 | JSON payload completeness | ✅ | `received_bits`, `fec`, `codeword_alignment`, `interleaving_result`, `protocol`, `provenance` |

## 6. Demonstration + validation honesty

| # | Requirement | Status | Evidence / notes |
|---|---|---|---|
| 41 | Deterministic demo set (PSK/QAM/FSK + FEC + interleavers) | ✅ | `tests/demo_captures.py` (11 cases) + reference sidecars; every case classifies correctly and coded cases decode cleanly |
| 42 | Reference-free alignment | ❌ | Not implemented and not claimed |
| 43 | Non-block automatic interleaving identification | ❌ | See #21 |
| 44 | Real-world capture validation | ❌ | Synthetic only |
| 45 | Hardware / SDR validation | ❌ | Not performed |

## 7. Bugs fixed in this pass (2026-09-29)

1. `pipeline.py` — the ML stage called `predict_modulation` without importing
   it, so enabling ML only ever appended a warning. The import is added.
2. `cli.py` — `analyze` built a `FrameConfig` from `--sync-word` and never
   passed it to the pipeline.
3. `cli.py` — `analyze --ml/--ml-fusion/--labels` were parsed but never wired
   into the `AnalysisConfig`; `analyze_capture` now accepts them.
4. `cli.py` — the `--source gnuradio` branch referenced `analyze_samples` /
   `processing_mode_config` without importing them and passed a
   `reference_bits_path` argument `analyze_samples` does not accept.
5. `cli.py` — `demodulate` computed the protocol result and then omitted it
   from its JSON payload.
6. Dead duplicate `set_defaults(func=cmd_train)` calls removed.
7. Demo set + reference sidecars added so the GUI demonstrates the clean
   chain (alignment + BER + decoded payload) for the cases that support it.

## 8. Known limitations (concise)

- Blind phase/origin alignment is 16-QAM-only; uncoded QPSK/8-PSK cannot be
  BER-scored against a reference without it.
- Automatic interleaving identification is block-only.
- FSK: BFSK classification + coherent demodulation with known/estimated
  tones; no blind FSK demodulation, no 4FSK/GMSK.
- All validation is on synthetic, fixed-seed captures. No real RF, no SDR
  hardware, no regulatory claims.
