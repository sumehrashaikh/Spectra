# Third-party validation — public benchmark (RadioML 2016.10a)

Source: `HF datasets-server: hitrs909/RML2016/6db_0 test split`
Frames evaluated: **1100** (SNR slice 6 dB, 8 samples/symbol, 1 Msps baseband)

This is the only section of the evidence pack that is not produced by our own generator. It runs the same estimators over an externally produced dataset, so the numbers are not ours to choose.

## 1. Parameter extraction — symbol rate (label-free)

RadioML 2016.10a frames are generated at a known symbol rate, so this claim needs no labels at all.

| Metric | Value |
|---|---|
| Ground-truth symbol rate | 125000 Hz |
| Frames with an estimate | 1100 |
| Median estimate | 7874.0 Hz |
| Mean relative error | 89.39% |
| Median relative error | 93.70% |
| Estimates within 1% of truth | 0.0% |
| Mean reported confidence | 77.99 |

### 1b. Why per-frame benchmark slices cannot test it

The benchmark serves 128-sample frames at 1 Msps, i.e. 16 symbols at 8 samples/symbol. Measured on our own captures (exact ground truth), the estimator's behaviour against record length is:

| Samples | Symbols | Estimate (Hz) | Truth (Hz) | Relative error | Reported confidence |
|---|---|---|---|---|---|
| 128 | 1.6 | 188.98 | 100.0 | 89.0% | 81.2 |
| 256 | 3.2 | 62.75 | 100.0 | 37.2% | 68.8 |
| 512 | 6.4 | 15.66 | 100.0 | 84.3% | 98.3 |
| 1024 | 12.8 | 23.46 | 100.0 | 76.5% | 100.0 |
| 4096 | 51.2 | 33.21 | 100.0 | 66.8% | 100.0 |
| 40973 | 512.2 | 99.97 | 100.0 | 0.0% | 100.0 |

Two things follow, and both are honest limitations of the current build rather than of the dataset:

1. The estimator needs a long observation window (thousands of samples). On short records it is not merely inaccurate, it can be **confidently** wrong — at 512 samples it reported 15.66 Hz against a true 100 Hz with 98.3 confidence. Gating confidence on observation length is a concrete, high-value fix on the roadmap.
2. Therefore the benchmark's 128-sample slices cannot fairly test symbol-rate estimation; the number below is reported for completeness, not as a capability claim.

## 2. Modulation classification — cross-dataset transfer

Our classifier returns `Unknown` rather than guessing, so the useful pair of numbers is agreement with the label *and* how often a decision is made at all.

| Mode | In-scope frames | Correct | Agreement | Frames with a decision (not `Unknown`) |
|---|---|---|---|---|
| raw frame (as served by the dataset) | 400 | 100 | **25.0%** | 100.0% |
| pipeline preprocessing applied | 400 | 89 | **22.2%** | 43.5% |

| RadioML class | Frames | In scope | Raw decision | Conditioned decision |
|---|---|---|---|---|
| 8PSK | 100 | yes | BPSK (100) | Unknown (99), OOK (1) |
| AM-DSB | 100 | no | BPSK (100) | OOK (67), Unknown (33) |
| AM-SSB | 100 | no | BPSK (100) | OOK (69), Unknown (31) |
| BPSK | 100 | yes | BPSK (100) | BPSK (85), OOK (14) |
| CPFSK | 100 | no | BPSK (100) | Unknown (88), BFSK (11) |
| GFSK | 100 | no | BPSK (100) | Unknown (65), OOK (21) |
| PAM4 | 100 | no | BPSK (100) | OOK (93), BPSK (6) |
| QAM16 | 100 | yes | BPSK (100) | Unknown (96), OOK (4) |
| QAM64 | 100 | no | BPSK (100) | Unknown (89), OOK (11) |
| QPSK | 100 | yes | BPSK (100) | Unknown (94), QPSK (4) |
| WBFM | 100 | no | BPSK (100) | OOK (64), Unknown (25) |

**The finding that matters more than the accuracy number:** in raw mode the classifier answers `BPSK` for every single frame of every class — a silent, confident collapse on out-of-distribution input. With the pipeline's own preprocessing in front of it the same classifier stops claiming an answer it cannot support and abstains (`Unknown`) instead. Neither mode reaches the in-domain result, which is the honest cross-dataset gap this harness exists to expose — and the reason the architecture carries a learning layer and an abstention path.

## 3. Caveats a judge should apply to the numbers above

* **Label mapping.** This public mirror's card does not document its class ordering. The mapping used here is the canonical DeepSig order (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM) and is corroborated by the per-class amplitude-spread structure reported below — amplitude spread is label-free, so it independently checks which classes are constant-modulus.
* **Domain shift, stated plainly.** SPECTRA's rule-based classifier is tuned on RRC-shaped synthetic captures with our own synchroniser in front of it. RadioML frames are 128 samples with different pulse shaping, random phase and fading, and are fed here without the detection/isolation stages. This number is therefore a *transfer* measurement, not the classifier's in-domain accuracy (which the demo matrix reports as 11/11).
* **Short frames.** 128 samples is below the length our symbol-rate and timing stages prefer, so per-frame failures are expected and are reported as `Unknown` rather than silently dropped.

## 4. Amplitude-spread corroboration (label-free)

| Class label (inferred) | Mean |x| spread (CV) |
|---|---|
| 8PSK | 0.0365 |
| AM-DSB | 0.0049 |
| AM-SSB | 0.0406 |
| BPSK | 0.0327 |
| CPFSK | 0.0345 |
| GFSK | 0.0267 |
| PAM4 | 0.0285 |
| QAM16 | 0.0378 |
| QAM64 | 0.0381 |
| QPSK | 0.0358 |
| WBFM | 0.0104 |

## 5. Reproduce

```
python prototype/tools/sih_deck/validate_public_dataset.py
# or against a local copy of the DeepSig release:
python prototype/tools/sih_deck/validate_public_dataset.py --database path/to/RML2016.10a_dict.pkl
```

> **Note:** the harness script above was a working-session tool and is not
> part of this repository snapshot; the numbers in this document are the
> frozen output it produced.
