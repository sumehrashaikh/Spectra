# Spectra GUI User Guide

This guide is for people who use **the desktop application only** — no
terminal work is required after the one-time installation.

The GUI loads a radio capture (a WAV audio file or a raw IQ file),
detects the signals inside it, classifies their modulation, synchronizes
carrier and timing, demodulates the data bits, and — when a reference of
the transmitted bits is provided — measures the bit error rate. It never
guesses a result it cannot measure: unsupported signals are reported as
"Unknown" and demodulation is skipped rather than fabricated.

> Spectra is an engineering prototype validated on synthetic signals.
> Results on real-world hardware captures are not certified.

---

## 1. Installing (one time)

You need Python 3.10 or newer. From the repository root:

```bash
pip install -e .
```

This installs everything the GUI needs (numpy, scipy, matplotlib,
PySide6) and the `spectra` launcher command.

---

## 2. Starting the app

Any of these opens the same window (titled **"SIH Signal Analyzer"**):

| How | What to type / click |
|---|---|
| Easiest, after `pip install -e .` | run `spectra gui` from any terminal |
| From the repository root | `python -m prototype.main` |
| From the `prototype/` folder | `python main.py` |

A window opens with the toolbar at the top and empty plot panels below.
If the window fails to appear with an error about Qt or "offscreen",
you are on a headless machine — the GUI needs a desktop session.

---

## 3. The window at a glance

From top to bottom:

1. **Toolbar** — `Open WAV`, `Analyze Signal`, `Analyze Selected`,
   `Isolate Selected`, `Mode:`, `Analyze all candidates`,
   `ML assist (CNN)`, `FEC:`, `Clear`.
2. **Progress bar** — a thin bar that animates while an analysis runs on
   a background thread (the window stays responsive).
3. **Export bar** — `Export JSON` and `Provenance` buttons (enabled
   after an analysis).
4. **File information** — File, Format, Sample Rate, Samples, Duration.
5. **TIME DOMAIN** — amplitude of the whole capture.
6. **SPECTRUM** — power spectral density; spikes are your signals.
7. **WATERFALL** — spectrum over time; a steady vertical stripe means a
   continuous carrier.
8. **DETECTED SIGNALS** — the table of detected signals, plus the
   **"Analyzed candidate:"** selector in batch mode.
9. **SELECTED SIGNAL** — which table row is currently selected.
10. **CONSTELLATION** — the recovered symbol constellation after an
    analysis (or the raw IQ scatter otherwise).
11. **SIGNAL PARAMETERS** — the full results glossary (Section 8).

All panels scroll; nothing is hidden behind menus.

---

## 4. Your first analysis (step by step)

1. Click **Open WAV**. The file dialog filters on
   `*.wav *.iq *.cfile *.c64 *.dat *.raw …` — WAV and raw IQ are both
   accepted here.
2. The File information panel fills in (Format shows `WAV` or
   `Raw IQ`), and the Time / Spectrum / Waterfall panels render
   immediately.
3. Click **Analyze Signal**. The button changes to "Analyzing…" and the
   progress bar animates. Large captures take longer; the window stays
   responsive and a second click is politely refused.
4. A summary dialog appears, for example:

   ```
   Pipeline: V2 (balanced)
   Modulation: QPSK
   Symbol rate: 100.00 symbols/s
   Samples/symbol: 80.00
   Demodulated: 512 symbols / 1024 bits
   BER: no reference loaded
   ```

5. Dismiss the dialog and read the results in the panels (Sections 7
   and 8). If the analysis completed with warnings, they are listed in
   the same dialog — read them, they usually mean "this stage could not
   produce evidence" rather than "something broke".

To start over with a different file, just click **Open WAV** again;
**Clear** resets everything.

---

## 5. Opening a raw IQ file

Raw IQ files (`.iq`, `.cfile`, `.c64`, `.cf32`, `.dat`, `.raw`, …)
contain no metadata, so the app asks for what it needs — but only what
it actually needs:

- **Sample rate in Hz** — always prompted if unknown. If you don't know
  it, frequency and symbol-rate results will be scaled incorrectly, so
  this one is worth getting right.
- **Sample format** — prompted if unknown; `int16` (interleaved I/Q) is
  the most common SDR export. The full list: `int16`, `complex64`,
  `complex128`, `float32`, `float64`, `int8`, `uint8`, `int32`.

You can avoid the prompts by placing a small JSON sidecar next to the
file, named `<yourfile>.meta.json`, for example:

```json
{
  "sample_rate": 8000,
  "dtype": "int16",
  "endianness": "little",
  "iq_order": "iq",
  "center_frequency_hz": 433900000
}
```

If the sidecar provides a field, the matching prompt is skipped.

---

## 6. Detected signals: select, isolate, analyze

After an analysis the **DETECTED SIGNALS** table lists every signal
found, with its frequency, bandwidth and peak level.

- Click a row to **select** it. The SELECTED SIGNAL panel updates.
- **Isolate Selected** runs a digital down-conversion + filter that
  extracts that one signal into its own stream. A dialog confirms the
  center frequency.
- **Analyze Selected** re-runs the deep per-signal analysis on the
  isolated stream. This is the path that also handles 8-PSK, OOK/ASK
  and the per-modulation constellation views.

For a single-signal capture the main **Analyze Signal** pass is all you
need; the Isolate/Analyze-Selected pair is for digging into one signal
out of many.

---

## 7. Batch mode: analyzing every signal at once

Tick **Analyze all candidates** before clicking **Analyze Signal**, and
the pipeline runs on *every* detected signal instead of only the
strongest. A new selector appears under the table:

```
Analyzed candidate:  #1: QPSK ▾
```

Switching the selector re-targets every detail panel — parameters,
constellation, BER, FEC — at that candidate. The table itself always
shows all detections.

---

## 8. Reading the SIGNAL PARAMETERS panel

| Field | Meaning |
|---|---|
| Sample Rate / Duration / Peak / RMS | basic statistics of the loaded file |
| Signal Detected | whether anything rose above the noise |
| Noise Floor / Detection Threshold | the level evidence was judged against |
| Dominant Frequency / Detected Band / Bandwidth | where the signal lives |
| SNR | estimate when reliable; "Not reliably estimable" otherwise |
| **Modulation** | the classification result (`QPSK`, `16-QAM`, `BPSK`, `BFSK`, or `Unknown`) |
| Samples/Symbol / Symbol Rate | timing estimate; the rate estimate carries a confidence |
| Timing Confidence | how sure the estimator is — below ~50% treat the rate as provisional |
| Recovered Symbols/Bits | how much data was demodulated |
| **BER** | bit error rate against a reference (Section 10); "No reference loaded" otherwise |
| Decision Margin | average distance from each symbol to its nearest wrong decision point — bigger is cleaner |
| **FEC** | error-correction summary: scheme, corrected errors, uncorrectable blocks |
| Freq Offset / Phase Offset | residual carrier frequency (Hz) and phase (degrees) measured by the synchronizer |
| **ML Prediction** | the optional CNN's verdict with its score and the top-3 ranking (Section 15); `off` until you enable `ML assist (CNN)` |
| Selected Signal | which table row you selected |

The **CONSTELLATION** panel after an analysis shows the *recovered*
symbol lattice (e.g. a tight 4×4 grid for 16-QAM) — a smeared blob
there means timing/noise problems, a clean grid means good demodulation.

---

## 9. FEC decoding (forward error correction)

If the transmitter encoded its data with an error-correcting code, pick
the scheme in the **FEC:** selector before analyzing:

- `none` — plain demodulation (default)
- `conv12` — rate-1/2 convolutional K=7 + Viterbi
- `hamming74` — Hamming(7,4) block code
- `repetition3` — 3× repetition

FEC is **never guessed**: if you select a scheme, the demodulated bits
are run through it and the parameters panel reports how many bit errors
were corrected (and whether any blocks were beyond repair). Note the
block schemes need the bit count to divide evenly (7 bits per Hamming
block, 3 per repetition block) — otherwise you get an honest
"no FEC stage output" note instead of a wrong answer.

---

## 10. BER validation with a reference file

Bit error rate is only meaningful against the bits that were actually
transmitted. To have the GUI measure it, place a reference file next to
the capture:

- For `mycapture.wav` → `mycapture.reference.npz` (same folder, same
  base name) containing a NumPy array named `bits` of 0/1 uint8:

  ```python
  import numpy as np
  np.savez("mycapture.reference.npz", bits=transmitted_bits)
  ```

- The same convention works for raw IQ captures.

When the reference is found, the parameters panel and the summary
dialog report the BER, the number of bit errors, and how many bits were
compared. For BPSK the receiver also searches both polarities; QPSK
searches the four 90° phase ambiguities; 16-QAM searches the symbol
origin. Without a reference the GUI still demodulates — it just
reports "No reference loaded" instead of inventing a BER.

---

## 11. Processing modes

The **Mode:** selector trades speed for thoroughness:

| Mode | Use it when |
|---|---|
| `quick` | exploring: fastest pass, basic checks |
| `balanced` | default: full DSP chain (recommended) |
| `deep` | hard or marginal captures: extra cross-checks |
| `realtime` | long streaming-style captures |

---

## 12. Saving your work

- **Export JSON** — writes the complete machine-readable result of the
  last analysis: every stage's estimates with confidences and methods,
  warnings, and provenance. The default filename is
  `<capture>_analysis.json`.
- **Provenance** — shows the audit trail of the last run: which
  software version and git commit produced it, when it started, and
  how many milliseconds each pipeline stage took. If a number ever
  looks odd, this tells you exactly what produced it.

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Dialog says "Unable to open file" | raw IQ file without a known format — answer the prompts, or check the dtype (int16 interleaved is the usual default) |
| `Modulation: Unknown` | the signal is real but does not match a supported scheme; the tool refuses to guess — check the constellation for a pattern |
| Symbol rate looks like an exact fraction of the expected one (e.g. 25 instead of 100) | known estimator weakness on short captures (subharmonic mislock); try `deep` mode or a longer capture |
| BER ≈ 0.5 with a good-looking constellation | the reference file probably belongs to a different capture, or the receiver picked a different symbol origin — check the reference name matches the capture |
| "Analysis in progress" on Analyze | the previous run is still on the background thread; wait for the progress bar to finish |
| FEC reports "no FEC stage output" | bit count is not divisible by the block size (7 for hamming74, 3 for repetition3) |
| Frequency numbers are all scaled by the same factor | the raw-IQ sample rate was entered wrong; re-open the file with the correct rate |

---

## 14. Known limitations

- Validated on synthetic signals only; not certified for operational use.
- Supported modulations: BPSK, QPSK, 16-QAM, BFSK end-to-end; 8-PSK,
  OOK/ASK demodulation via the isolate-and-analyze path. Anything else
  is reported honestly as `Unknown`.
- The symbol-rate estimator can lock onto subharmonics on short
  captures (affects QPSK captures equally — it is not QAM-specific).
- FEC must be configured explicitly; it is never auto-detected.

---

## 15. ML-assisted classification (CNN)

Tick **ML assist (CNN)** before analyzing and the pipeline also runs a
convolutional neural network over the selected signal, reported in the
**ML Prediction** parameter row and in the completion dialog, e.g.

```
ML Prediction: QPSK (41%, 8 frames)  QPSK 0.41 8FSK 0.23 GMSK 0.17
```

The verdict, its score, and the top-3 ranking are *supplementary
evidence*: the CNN never overrides or replaces the rule-based DSP
classification, and the two are reported side by side so you can
compare them.

Under the hood:

- The network runs on a NumPy-only engine — no TensorFlow/PyTorch
  install is needed, and nothing leaves your machine.
- The shipped model was **trained inside this project on synthetic
  signals** (the same generator and channel impairments the rest of
  Spectra uses) and is validated on synthetic holdouts only. Treat its
  scores as a weak second opinion, not a guarantee.
- The original `modulation_cnn.pkl` shipped with an untrained network
  (verified from its weights); the trained artifact in
  `prototype/ml/` supersedes it. You can retrain any time — see
  `docs/USER_GUIDE.md`, section "Machine learning".

## 16. Where to go next

- `docs/USER_GUIDE.md` — the task-oriented guide including CLI usage
  and the machine-learning (training) workflow
- `docs/ARCHITECTURE.md` — how the pipeline works inside
- `docs/VALIDATION.md` — what is and is not validated
- `README.md` — quick start and supported formats
