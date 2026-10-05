# SIH 2026 — PS SIH26147 / PS 147 — Deck Pack

**Problem statement:** *Automated model for analysis of .IQ and .wav files along with signal parameter extraction*
**Organisation:** NTRO · **Theme:** Miscellaneous · **PS category:** Software
**Project name:** **SPECTRA** — evidence-scored signal analysis workbench

---

## 0. How to use this document

1. **Fastest path:** open `docs/SIH26147_SPECTRA_IDEA.pptx` — a copy of the official template
   already filled with this content (6 slides, the instruction slide removed, all figures
   placed, pointer lines preserved). Replace `<<TEAM ID>>`, `<<TEAM NAME>>`, `<<COLLEGE>>`,
   `<<GITHUB LINK>>` and `<<DEMO VIDEO LINK>>`, trim bullets that overflow, then export to PDF.
   Regenerate it any time with `python tools/sih_deck/make_pptx.py`.
   *(Manual path: open the official template and paste the* ON-SLIDE TEXT *blocks below into
   the matching slide — Slide 1 → title page, Slide 2 → "Idea title / Proposed solution", etc.)*
2. Drop the figures from `prototype/docs/assets/` (plus `radioml_confusion.png` from
   `prototype/docs/evidence/`) into the slide where marked. They were generated from the
   running code — no mock-ups.
3. **Delete the template's last "Important instructions" slide** before exporting.
4. Export to **PDF** and upload. Six slides total, including the title.
5. Every number on every slide traces to a file in `prototype/docs/evidence/`. The trail is
   in the Claim Ledger (§9). Before submitting, re-run
   `python tools/sih_deck/sih_evidence_pack.py` and then
   `python tools/sih_deck/verify_deck_claims.py` — the second script fails loudly if any
   slide number has drifted from the evidence.

**Placeholders to fill before you submit:** `<<TEAM ID>>`, `<<TEAM NAME>>`,
`<<COLLEGE>>`, `<<MENTOR>>`, `<<GITHUB LINK>>`, `<<DEMO VIDEO LINK>>`.
Confirm the theme string for PS 147 on the portal (this pack uses *Miscellaneous*, which
matches the published problem-statement listing).

**Voice rule for the whole deck:** every claim is either *measured* (a number from the
evidence pack) or *labelled as roadmap*. Nothing is asserted. That is the entire
differentiator against the other teams answering this statement.

**Snapshot note:** the generator scripts (`tools/sih_deck/*`) and the figure assets
(`prototype/docs/assets/fig_*.png`) were working-session tools and are **not**
shipped in this repository snapshot. What is shipped: this deck pack, the filled
`docs/SIH26147_SPECTRA_IDEA.pptx`, and the frozen evidence pack in
`prototype/docs/evidence/` — every number still traces through the Claim Ledger
below to a file in that directory.

---

## SLIDE 1 — TITLE PAGE

### ON-SLIDE TEXT

> **SMART INDIA HACKATHON 2026**
>
> **Problem Statement ID — SIH26147 (PS 147)**
> **Problem Statement —** Automated model for analysis of .IQ and .wav files along with
> signal parameter extraction
> **Theme —** Miscellaneous · **PS Category —** Software
> **Organisation —** National Technical Research Organisation (NTRO)
>
> **Team ID —** <<TEAM ID>> · **Team Name —** <<TEAM NAME>>
>
> **SPECTRA** — from a raw capture to an auditable answer
> *blind parameter extraction · demodulation · FEC & de-interleaving · bit-stream
> correlation, in one offline workbench*

### LAYOUT & VISUALS

- Three thumbnails in a single row, each captioned, taken straight from
  `docs/assets/`: `fig_spectrum.png`, `fig_waterfall.png`, `fig_constellation.png`.
- Caption under the strip: *"Real output from the working prototype — 16-QAM capture,
  carrier and timing recovered by the pipeline."*

### SPEAKER NOTES (15 s)

"We picked PS 147 from NTRO. The ask is to take a raw off-air recording — .IQ or .wav, in
any of the formats they come in — and work out what it is: sampling rate, modulation, FEC,
interleaving, then demodulate it and correlate the recovered bit stream. We built SPECTRA
to do that, and — this is the part we want you to judge us on — to say *how sure it is*
at every step."

**Narrator do-not-say:** do not say "we replace a signals-intelligence analyst". Say
"we compress the manual part of the workflow and keep the analyst's decision auditable".

---

## SLIDE 2 — IDEA TITLE / PROPOSED SOLUTION

*Template pointers to cover: detailed explanation of the solution · how it addresses the
problem · innovation and uniqueness.*

### ON-SLIDE TEXT

**Idea title — SPECTRA: a confidence-aware, evidence-scored workbench that turns a raw
IQ/WAV capture into signal parameters, demodulated bits and an auditable result.**

**The problem, precisely**
- The same waveform is stored two different ways (`.wav` vs `.IQ`), across kHz–GHz bands,
  from different sensors and locations — so parameters vary and the two formats need
  different processing paths.
- Manual analysis is slow, operator-dependent and not reproducible; the detail that
  downstream sensors actually need — sampling rate, modulation, FEC, interleaving — is
  exactly the detail that is missing.
- A *wrong* identification is worse than an honest "unknown", because it propagates into
  the sensor configuration downstream.

**What SPECTRA does** — one pipeline, one workbench:
`load → pre-process → detect → isolate → extract parameters → synchronise → classify →
demodulate → de-interleave → FEC decode → bit-stream correlate → report`
Every stage returns **`{value, method, confidence, evidence}`** and **`Unknown` is a
first-class result**, never a guess.

**How it addresses the problem statement**

| PS requirement | What SPECTRA does today | Status |
|---|---|---|
| Sampling rate, bandwidth, centre freq, SNR, symbol rate | Extracted automatically with confidence; symbol-rate error ≤ 0.06% on the demo matrix | Built |
| Modulation type | 6 classes + explicit `Unknown` (BPSK, QPSK, 8-PSK, 16-QAM, BFSK, OOK/ASK) | Built |
| Demodulation (FSK, QAM, PSK) | BPSK/QPSK/8-PSK/16-QAM/BFSK/OOK-ASK decision chains, BER against a reference | Built |
| De-interleaving (block, convolutional, diagonal, pseudo-random) | All 4 families, round-trip proven through the receiver's own deinterleaver | Built; automatic ID is block-only and validator-gated |
| FEC (convolutional+Viterbi, RS, concatenated, LDPC) | 6 registered schemes incl. K=7 Viterbi, shortened GF(256) RS, (3,6) LDPC, RS+conv concatenated | Built; blind auto-ID is evidence-scored |
| Bit-stream correlation (header/payload) | Sync-word correlation, frame extraction and field parsing | Partial — explicit frame config; automatic discovery is roadmap |
| GUI: spectrum, constellation, waterfall, demodulation | 4 plot tabs + per-signal parameter rows + FEC/interleaver controls + JSON/HTML export | Built |

**Innovation and uniqueness**
1. **Evidence fusion, not a single-shot guess** — DSP structure, decoder outcome and
   optional ML scores combine into one explainable confidence.
2. **Honest-`Unknown` as a design principle** — the engine refuses to answer when the
   evidence is weak, and records what the evidence was.
3. **Provenance on every stage** — method, timings, software version and git commit are
   exported with the result, so a finding can be reproduced and audited.
4. **Closed-loop escalation** — low confidence routes to a deeper re-analysis pass;
   analyst-confirmed labels accumulate into a signal-fingerprint library.
5. **Fully offline** — no cloud, no external service, no RF transmission needed.

**Status badge:** *"13 of 18 requirement areas built and test-verified in 13 days;
387 test functions (435 tests passing); the 5 partial areas are the hard ones and are
listed as measured roadmap in Slide 4."*

### LAYOUT & VISUALS

- The mapping table is the centre of gravity — keep it, shrink the prose.
- Right-hand column: the **status badge** as a callout box, plus one small figure:
  `docs/evidence/radioml_confusion.png` is *not* for this slide (it is slide-4 material).
  Use `fig_constellation.png` here if you want a second visual.

### SPEAKER NOTES (60 s)

"Three things make this different. First, we don't answer in one shot: every stage
produces a value *plus* the method, a confidence and the evidence behind it. Second, when
the evidence is weak the tool says `Unknown` — we treat that as a feature, because in this
domain a confident wrong answer is worse than no answer. Third, every stage is
provenance-tracked, so any result can be reproduced and audited. That's why the mapping
table can be this specific: for each point in the statement we can show either a measured
capability or a labelled roadmap item."

---

## SLIDE 3 — TECHNICAL APPROACH

*Template pointers to cover: technologies · methodology and process, with flowchart /
working prototype.*

### ON-SLIDE TEXT

**Technology stack (as built)**
- **Python 3.13** — NumPy 2.2.3 / SciPy 1.17.1 DSP core (estimation, filtering,
  synchronisation, coding)
- **PySide6 6.11.2 + matplotlib** — desktop GUI and figures
- **GNU Radio** — integrated as the *ingest/streaming layer* behind a documented chunk
  contract (hardware backends are roadmap)
- **pytest** — 387 test functions across the DSP, FEC, interleaving, GUI and end-to-end
  chain; **435 tests passing**
- **Optional NumPy-only CNN runtime** — a second opinion that never gates the DSP result
- **C++ kernels for compute-critical loops** — planned, not yet used (stated honestly)
- **SigMF-compatible metadata sidecars** — capture metadata travels with the data

**Methodology (the algorithms actually in the repository)**
- Format-agnostic ingest: WAV (mono → Hilbert analytic, stereo → IQ, 8/16/32-bit PCM and
  float) · raw interleaved IQ (complex64/128, float32/64, int8/uint8/int16/int32, both
  endianness, I-first or Q-first) · separate I/Q files · JSON/SigMF sidecars · chunked
  streaming
- Pre-processing → spectral peak/region detection → DDC isolation (mix, low-pass, decimate)
- Parameter extraction: SNR, symbol rate (M-th power / QAM-specific / BFSK estimators),
  samples-per-symbol, roll-off
- Synchronisation: Gardner-type timing recovery + carrier phase/frequency recovery
- Classification: waveform gates (r2/r4 phase coherence, amplitude spread, frequency
  bimodality) + constellation clustering on synchronised symbols
- Demodulation: Gray-coded QPSK/16-QAM, 8-PSK (π/4 residual documented), coherent BFSK,
  OOK/ASK
- Quality: BER against a transmitted reference with an explicit ambiguity search
  (BPSK polarity · QPSK 90° · 16-QAM rotation × symbol origin)
- FEC identification: runs the real decoders, scores structural evidence, claims only at
  confidence ≥ 60, otherwise `UNKNOWN`; interleaver identification is structural and
  requires a supplied validator

**Working prototype — measured, not mocked**

| Result | Value |
|---|---|
| Test suite | **435 passed**, 0 failed |
| Demo matrix (deterministic captures with ground truth) | 11 captures · **11/11 classified correctly** |
| Symbol-rate estimate vs. truth | **max error 0.059%**, mean 0.030% |
| Payload recovery with the scheme supplied | **3 of 4** FEC-bearing captures bit-exact (1024/1024: RS+block, concatenated+block, concatenated+pseudo-random); convolutional 1001/1024 with the pulse-shaping tail honestly reported |
| Flagship capture (16-QAM + concatenated + block) | symbol rate 99.98 Hz (true 100) · SNR 52.1 dB · **BER 0.0** · 1024 bits decoded |
| Automatic identification ladder | FEC **3/3** at capture length on clean bits (conv 80, RS 70, concatenated 83); interleaver AUTO_DETECTED at the true depth 8 with confidence 1.0 **when a FEC-decode validator is supplied**; end-to-end on lossy captures it returns `UNKNOWN`/`UNRESOLVED` (0 of 11 claimed) — root cause on Slide 4 |

**Prototype status:** requirement coverage **87.6%** (13/18 areas built and
test-exercised) · unattended autonomy **73.6%** · residual work is concentrated in the
five partial areas below the bar on the chart.

**Roadmap, stated as milestones:** M1 wire the FEC decoder as the interleaver validator
(engine already proven) · M2 tail-loss-tolerant identification evidence · M3
learning layer trained on public benchmarks · M4 confidence gating on observation length ·
M5 live SDR backend · M6 C++ kernels.

### LAYOUT & VISUALS

- **Left/top: `fig_flowchart.png`** — our own flowchart, colour-coded
  solid = built, dashed = partial, dotted = roadmap. Do **not** reuse the old workflow
  image: it advertises Turbo codes, LSTM/Transformers, content recovery and C++ that
  are not in the repository, and an NTRO judge can break that in one question. The
  generated flowchart says the same thing about the architecture *and* survives scrutiny.
- **Right/bottom block:** the measured-results table, then
  `fig_gui_annotated.png` (the real GUI after analysis) — this is the "working prototype"
  evidence the template asks for.
- **Small strip:** `fig_spectrum.png` · `fig_waterfall.png` · `fig_constellation.png`.
- **Bottom bar:** `fig_requirement_status.png` (coverage vs autonomy per requirement) and
  `fig_auto_id_ladder.png` (the capability ladder).

### SPEAKER NOTES (75 s)

"The stack is deliberately boring and open: Python, NumPy/SciPy, PySide6, GNU Radio as an
ingest layer, pytest for verification. The interesting part is the method — the pipeline
is staged, and every stage is test-exercised: 387 test functions, 435 tests passing.

On the demo matrix — eleven captures with known ground truth — we classify eleven out of
eleven correctly and the symbol-rate error stays under 0.06%. With the coding scheme
supplied we recover payloads bit-exactly; where the capture loses its pulse-shaping tail
we report the truncation instead of inventing bits.

The last row is the one we want to draw your attention to, because it is where this
problem is actually hard. FEC identification works on clean coded streams — convolutional,
Reed-Solomon and concatenated all detect correctly. End-to-end, on a lossy capture, it
returns `UNKNOWN`, and the interleaver identifier only resolves when a validity check is
supplied. We know the root cause of both and it is on the next slide. We measured it
rather than claiming it."

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

*Template pointers to cover: feasibility analysis · potential challenges and risks ·
strategies for overcoming them.*

### ON-SLIDE TEXT

**Feasibility**
- **Technical:** pure software on a commodity laptop. 100% open-source stack, fully
  offline. No RF transmission is required, so evaluation is safe and reproducible.
- **Architectural:** band-agnostic by construction — all analysis happens on baseband
  captures, so the same chain applies to HF, VHF, UHF and GHz recordings.
- **Demonstrated:** 13 days of work produced 8 commits, 387 test functions, a working GUI
  and a full pipeline; 13 of 18 requirement areas are built and test-verified.
- **Data:** a deterministic synthetic testbed with exact ground truth (11 captures,
  FEC × interleaver × modulation matrix) **plus** an offline harness against the public
  RadioML 2016.10a benchmark.

**Challenges → strategies (each row is measured, not hypothetical)**

| # | Challenge | Measured evidence | Strategy / milestone |
|---|---|---|---|
| 1 | Blind FEC identification on unknown signals | On clean coded bit streams: conv 80, RS 70, concatenated 83 confidence, correct scheme 3/3. End-to-end on the 4 FEC captures: `UNKNOWN` 4/4 | Root cause is the lost pulse-shaping tail (1240 of 1280 aligned bits) + residual symbol errors, which breaks whole-codeword evidence. **M2:** partial-codeword and soft-decision evidence; decoder-success as a validator |
| 2 | Blind interleaver identification | Blind: `UNRESOLVED`, no depth claimed. With a FEC-decode validator supplied: **AUTO_DETECTED, block, depth 8, confidence 1.0** | The engine is validator-gated by design. **M1:** wire the existing FEC decoder as that validator — an integration step, not research |
| 3 | Cross-dataset transfer | 1100 public RadioML frames: 22–25% in-scope agreement (chance = 25%), and in raw mode the classifier answered `BPSK` for every frame — a silent collapse | Frames are 128 samples with different pulse shaping and no front-end. **M3:** run the full front-end on public *captures*, train the learning layer on public data, and gate abstention on evidence — the `Unknown` path already exists |
| 4 | Short observation records | Our own capture: symbol-rate error 89% at 128 samples and 84% at 512 samples — **with 98.3 reported confidence** | The estimator is confidently wrong below its evidence length. **M4 (highest value, lowest cost):** gate confidence on observation length; aggregate multi-frame windows |
| 5 | No live SDR / hardware validation | GNU Radio package not installed in this checkout; a deterministic synthetic source keeps the flow testable | GNU Radio is an *interface* today. **M5:** RTL-SDR/HackRF backend. Evaluation never depends on hardware, so the deliverable is never blocked by procurement |
| 6 | Evidentiary integrity | Per-stage provenance manifest (method, timing, version, git commit) exported with every result; SigMF-style metadata sidecars | Built and demonstrated: every slide number in this deck is reproducible from one command |

**Viability**
- **Who uses it:** monitoring/SIGINT stations, SDR test benches, spectrum regulators
  running interference investigations, RF research and teaching labs.
- **Deployment model:** analyst workstation today; headless CLI (`spectra analyze …`)
  for integration into existing collection pipelines tomorrow.
- **Cost model:** zero licensing, commodity hardware — import-substituting capability
  instead of foreign closed tools.
- **Timeline:** phases 1–3 delivered (foundations, RF/DSP, modulation & demodulation);
  the remaining milestones are integration and validation work, not greenfield research.
- **Honesty statement (keep this line on the slide):** *validated on synthetic captures
  with known ground truth and on a public benchmark slice; not validated on classified or
  operational recordings.*

### LAYOUT & VISUALS

- Challenges table takes two-thirds of the slide; the feasibility and viability blocks are
  short bullets.
- Place `docs/evidence/radioml_confusion.png` next to row 3 — showing the failing
  confusion matrix *deliberately*. Teams that hide this get caught in Q&A; showing it with
  the root cause converts it into credibility.

### SPEAKER NOTES (75 s)

"Feasibility first: this is software, entirely open-source, runs offline on a laptop, and
needs no RF emission — so it is safe and reproducible to evaluate. Because everything
happens on baseband captures, the same chain works from HF to GHz.

Now the hard part, and we'd rather you hear it from us. Four of our challenges have
numbers attached. Blind FEC identification works on clean coded streams — 3 out of 3 with
confidence 70 to 83 — but end-to-end on a lossy capture it says `UNKNOWN`, because the
pulse-shaping tail is lost and whole-codeword evidence breaks. The interleaver identifier
resolves the true depth perfectly *when a validity check is supplied*; wiring our own FEC
decoder in as that validator is the next milestone. On 1100 public RadioML frames our
rule-based classifier agrees with the labels about a quarter of the time — and in raw mode
it collapsed to one answer for every class, which we found and fixed by adding the
preprocessing front-end. And on short records the symbol-rate estimator can be
*confidently* wrong.

Every one of those has a named root cause and a milestone attached. This is why the
architecture has a confidence engine, an abstention path and a learning layer."

---

## SLIDE 5 — IMPACT AND BENEFITS

*Template pointers to cover: potential impact on target audience · benefits (social,
economic, environmental, etc.).*

### ON-SLIDE TEXT

**Direct impact on the target users**
- **Signal analysts (NTRO and equivalent):** hours of manual parameter hunting become
  minutes of automated extraction — with the method and confidence recorded, so the
  result can be defended.
- **Downstream sensor engineering:** the parameters the statement asks for (sampling rate,
  modulation, FEC, interleaving) are precisely the inputs needed to configure sensors, so
  measurement accuracy propagates directly. An honest `Unknown` prevents a bad
  configuration; a confident wrong answer causes one.
- **Monitoring stations:** consistent, reproducible extraction instead of operator-specific
  notes; batch mode sweeps every detected candidate in a wide capture.
- **Regulators, responders and labs:** fast identification of unknown or legacy emitters
  during interference investigations, disaster communications and RF teaching.

**Benefits**
- **Operational:** faster analysis, repeatable results, quantified confidence, and a
  complete audit trail per stage.
- **Strategic / national:** indigenous, self-hosted, offline spectrum-intelligence tooling —
  no cloud dependency and no foreign closed-source seat licences; strengthens
  *Atmanirbhar Bharat* capability in a sensitive domain.
- **Economic:** zero licensing cost, commodity hardware, an open Python/C++ skill base, and
  a codebase convertible into either a workstation product or an API for existing
  collection chains.
- **Social / educational:** a documented, reproducible DSP testbed for training the next
  cohort of RF analysts — including the negative results, which is what makes it usable
  for teaching.
- **Environmental:** software-only; no additional hardware footprint and no RF emission
  during analysis.

**Long-term**
The fingerprint/template library grows with every analyst-confirmed identification, turning
isolated captures into a queryable national signal-knowledge base. Because the engine is
baseband-agnostic, new bands and modulations extend the same pipeline rather than
requiring a redesign.

### LAYOUT & VISUALS

- Four impact cards (analyst · sensor engineering · monitoring · regulators) across the
  top, benefits as a two-column list below, long-term vision as a single closing strip.
- Optional small figure: `fig_requirement_status.png` if the deck wants an impact anchor,
  otherwise keep this slide text-only for contrast.

### SPEAKER NOTES (45 s)

"The measurable impact is analyst throughput: the manual parameter hunt collapses into one
automated pass, and every number comes with its method and confidence. The second-order
benefit matters more — the parameters this tool extracts are exactly what sensor
engineering needs, so an honest `Unknown` prevents a mis-configured sensor while a
confident wrong answer causes one.

Nationally, this is indigenous, offline, self-hosted spectrum tooling: no cloud, no foreign
licences. And because we ship the negative results too, the same codebase doubles as a
teaching testbed for the next set of analysts."

---

## SLIDE 6 — RESEARCH AND REFERENCES

*Template pointer: details / links of the reference and research work.*

### ON-SLIDE TEXT

**Automatic modulation classification — the benchmark**
- O'Shea, Corgan & Clancy, **RadioML 2016.10a / 2018.01A** (DeepSig) — 11–24 classes,
  −20…+18 dB, the de-facto public AMC benchmark. *Used directly:* our offline harness
  evaluates 1100 public frames and reports the cross-dataset gap (slide 4, row 3).

**Blind channel-code and interleaver recognition — the state of the art we are aligning to**
- **ConvLSTM-TFN** (*Sensors* 2025): convolutional-code identification above 90% accuracy
  for SNR > 0 dB — the accuracy target our staged FEC-ID roadmap aims at.
- **CRANet** (2025) and dual-branch CNN (*Sci. Rep.* 2026): deep-learning blind code
  recognition under heterogeneous signals.
- **Ahamed et al., *IEEE Access* 2024** — *Blind Interleaver Recognition Using Deep
  Learning*: endorses the structural-then-statistical approach we implement and the DL
  stage we plan.
- **Wee, Choi & Jeong** — interleaver-parameter estimation from the Hamming-weight
  distribution of linear codes; the algebraic route for depth estimation.

**Standards that define our FEC/interleaver template library**
- **MIL-STD-188-110C** and **STANAG 4285 / 4539** — HF serial-tone waveforms
  (convolutional coding + block interleaving), the terrestrial bands in the statement.
- **CCSDS 131.0-B** — Reed-Solomon + convolutional, turbo and LDPC options (space links).
- **DVB-S2** — LDPC + BCH with block interleaving.
- **3GPP channel coding** — turbo/LDPC families. *(These define what our library must
  eventually recognise, not what we have already decoded.)*

**Tooling and metadata standards**
- **GNU Radio** — building blocks and the acquisition interface.
- **SigMF**, Hilburn, *GRCon 2018* — the signal-metadata format our sidecars align with.
- **PySDR**, **inspectrum**, **Universal Radio Hacker** — feature and UX reference points.
- Classical links in the chain: Gardner timing recovery · Costas loop · cyclostationary
  symbol-rate estimation · M-th-power carrier-phase estimation.

**Links**
<<GITHUB LINK>> · <<DEMO VIDEO LINK>>
DeepSig RadioML: `deepsig.ai/datasets` (registration) · public mirror used here:
`zenodo.org/records/18397070` · SigMF: `sigmf.org` · GNU Radio: `wiki.gnuradio.org`
Papers: `doi.org/10.3390/s25041000` (ConvLSTM-TFN) · `ieeexplore.ieee.org` (Ahamed 2024)

### LAYOUT & VISUALS

- Three columns: *Benchmark* · *Blind recognition* · *Standards & tooling*. One line each,
  with a link per column.
- Do not list references you have not read: every item above maps to a design decision in
  slide 2, 3 or 4.

### SPEAKER NOTES (40 s)

"Our references separate into three groups. First the AMC benchmark everyone uses —
RadioML — which we actually run, not just cite. Second the blind code and interleaver
recognition literature, which tells us where the accuracy bar is — around 90% above 0 dB
SNR for convolutional codes — and confirms our structural-then-statistical approach.
Third the standards — MIL-STD-188-110C, STANAG, CCSDS, DVB-S2 — which define which FEC and
interleaver families a real tool has to recognise. And SigMF, which we align our metadata
to."

---

## 7. Q&A DEFENCE — the ten questions that decide this round

The independent analysis of PS 147 flags exactly two traps: claiming solved blind
FEC/interleaver identification, and claiming blind demodulation of an arbitrary unknown
signal. Both answers below turn the trap into evidence.

**Q1. "Show me this working on a signal you have never seen."**
Take any of the demo captures they name, or a RadioML frame, and run it live:
`spectra analyze <file> --json out.json`. Say: *"classification, parameters and BER are
live; FEC/interleaver identification reports what the evidence supports and `Unknown`
otherwise — you can see the thresholds in the output."* Never promise a decoded payload on
an arbitrary capture; promise a *measured* result and an evidence trail.

**Q2. "How do you identify the FEC without decoding it first?"**
We don't guess: the identifier runs the real decoders as hypotheses and scores structural
plus decoder evidence, claiming a scheme only at confidence ≥ 60, otherwise `UNKNOWN`.
CRC16/CRC32 are treated as validity evidence, never as a decoding hypothesis. And we tell
you the measured limit: 3/3 correct on clean coded streams, `UNKNOWN` end-to-end on lossy
captures because the tail is lost.

**Q3. "So it doesn't work end-to-end?"**
It identifies on clean streams and refuses on damaged ones — which for intelligence use is
the correct direction of failure. The fix is scoped: partial-codeword and soft-decision
evidence, milestone M2.

**Q4. "Why should we believe 87.6% coverage?"**
Because the number is computed, not asserted: `tools/sih_deck/sih_evidence_pack.py` walks
the requirement matrix, verifies every named module and test file exists, runs the suite
and the demo matrix, and prints the two percentages. Re-run it in front of them. It also
prints a deliberately *lower* second number — unattended autonomy, 73.6% — and explains
the difference.

**Q5. "What happens at 2 GHz, or with a 100 MHz capture?"**
All analysis is on baseband captures, so band is irrelevant to the chain; what matters is
sample rate and compute. Isolation decimates early, and ingestion streams in chunks, so a
wideband capture is processed in bounded memory. C++ kernels for the hot loops are
roadmap, and we say so.

**Q6. "Interleavers: you claim all four families?"**
The *deinterleavers* for all four families are implemented and round-trip tested —
block, convolutional, diagonal and pseudo-random, including their frame-length
constraints. The *automatic identification* is deliberately conservative: structural
detection of the block family, validator-gated, and it returns `UNRESOLVED` otherwise.
The other three are reported as weak hypotheses, never as a claim.

**Q7. "Can you correlate the bit stream and find headers?"**
Sync-word correlation, frame extraction and field parsing run against an explicitly
configured frame format, and an unmatched capture returns `Unknown`. Automatic discovery
of an unknown frame format is roadmap — `prototype/correlation/` is still empty and we
say so rather than pretending otherwise.

**Q8. "Your ML — is it a detector?"**
No, and we never present it as one. It is a second opinion with configurable fusion, it
never gates the DSP decision, and the packaged trained artifact reaches only 0.31
validation accuracy on synthetic frames. It is in the deck as roadmap, with the public
benchmark as the training set for milestone M3.

**Q9. "Would this survive a real capture from our collection?"**
"On real off-air captures with unknown pulse shaping and noise you should expect the
classification stage to abstain more often, and the symbol-rate estimator to need a longer
observation window — we measured both effects on public data and documented the fixes.
What we guarantee is that it will not silently invent an answer."

**Q10. "It's an offline file tool — where's the automation?"**
The automation is the pipeline: one command runs detect → isolate → extract → synchronise →
classify → demodulate → decode → correlate → report, batch mode sweeps every candidate,
and every stage is provenance-tracked. Live acquisition is an interface (GNU Radio) with
the hardware backend on the roadmap; the analysis half — which is what the statement
asks for — is complete.

---

## 8. DEMO & EVIDENCE RUNBOOK

Run these before recording or presenting; every slide number reappears here.

```bash
# from the repository root (the package folder is prototype/)
cd prototype

# 1. full test suite — the number on slide 3
python -m pytest tests -q                       # -> 435 passed

# 2. the evidence pack: requirement matrix, demo matrix, auto-ID ladder
python tools/sih_deck/sih_evidence_pack.py      # -> docs/evidence/EVIDENCE.md

# 3. third-party benchmark (needs network, ~20 s)
python tools/sih_deck/validate_public_dataset.py  # -> docs/evidence/PUBLIC_DATASET.md

# 4. regenerate every slide figure
python tools/sih_deck/make_assets.py            # -> docs/assets/*.png

# 5. live demo of the flagship case (16-QAM + concatenated + block interleaving)
python -c "from tests import demo_captures as dc; c=dc.build_capture(modulation='16-QAM', fec_scheme='concatenated', interleave_family='block', interleave_param=8); dc.write_capture('demo_flagship.wav', c)"
spectra analyze demo_flagship.wav --fec-mode manual --fec-scheme concatenated \
        --interleaving-mode manual --interleave-family block --interleave-depth 8 \
        --json flagship.json
spectra gui     # then open demo_flagship.wav in the GUI for the screenshots
```

**Slide → artifact map**

| Slide element | Source file |
|---|---|
| Title strip (spectrum/waterfall/constellation) | `docs/assets/fig_spectrum.png`, `fig_waterfall.png`, `fig_constellation.png` |
| Pipeline flowchart | `docs/assets/fig_flowchart.png` |
| Working-prototype screenshot | `docs/assets/fig_gui_annotated.png` (from the committed `gui_full_analyzed.png`) |
| Coverage vs autonomy chart | `docs/assets/fig_requirement_status.png` |
| Auto-ID capability ladder | `docs/assets/fig_auto_id_ladder.png` |
| Cross-dataset confusion matrix (slide 4, row 3) | `docs/evidence/radioml_confusion.png` |
| Every number on slides 2–5 | `docs/evidence/EVIDENCE.md`, `evidence.json`, `PUBLIC_DATASET.md` |

---

## 9. CLAIM LEDGER — number → source

| Claim used on a slide | Value | Source |
|---|---|---|
| Requirement areas built and test-verified | 13 / 18 | `evidence.json → requirement_matrix` (each row names its modules and proving test) |
| Requirement coverage · unattended autonomy | 87.6% · 73.6% | `evidence.json → build_coverage_pct / autonomy_pct` |
| Test functions · tests passing | 387 · 435 passed (0 failed) | `evidence.json → tests` (re-counted each run) |
| Demo matrix classification | 11 / 11 | `evidence.json → demo_summary` |
| Symbol-rate error | max 0.059%, mean 0.030% | `evidence.json → demo_summary.symbol_rate_max_rel_error` |
| Payload recovery (scheme supplied) | 1024/1024 bit-exact (RS, concatenated); 1001/1024 (convolutional, tail reported) | `evidence.json → demo_matrix[].configured` |
| Flagship capture | 99.98 Hz symbol rate · 52.1 dB SNR · BER 0.0 · 1024 bits decoded | `docs/assets/assets_manifest.json` |
| FEC auto-ID on clean coded bits | conv 80 · RS 70 · concatenated 83 · 3/3 correct | `evidence.json → auto_id_ladder.fec_bit_level_long` |
| Interleaver ID with validator | AUTO_DETECTED, block, depth 8, confidence 1.0 | `evidence.json → auto_id_ladder.interleaver_with_oracle` |
| Interleaver ID blind / end-to-end | UNRESOLVED / **0 of 11** | `auto_id_ladder.interleaver_blind`, `demo_summary.auto_interleaver_claims` |
| Cross-dataset agreement (public benchmark) | 25.0% raw · 22.2% preprocessed, 43.5% decisive (n=1100) | `PUBLIC_DATASET.md §2` |
| Symbol-rate vs record length | 89% error @128 samples, 84% @512 with 98.3 confidence, 0.03% at capture length | `PUBLIC_DATASET.md §1b` |
| FEC schemes · interleaver families · CLI commands · commits | 6 · 4 · 13 · 8 | repository audit in `evidence.json → requirement_matrix` |

---

## 10. WHAT WE DELIBERATELY DO NOT CLAIM

Keep this list to yourself; it is the discipline that makes the rest credible.

- No live-SDR or hardware validation. GNU Radio is an ingest interface with a synthetic
  offline source in this checkout.
- No real off-air or operational-capture validation. Synthetic ground truth + a public
  benchmark slice only.
- No solved blind FEC or interleaver identification; no claim that all four interleaver
  families are auto-detected.
- No Turbo-code support (the family is roadmap; the statement does not require it).
- No automatic header/payload discovery — frame parsing needs an explicit configuration.
- No ML detector: the CNN is evidence only, and the packaged artifact is weak.
- No decompressed/fabricated numbers: if a value is not in `docs/evidence/`, it does not
  go on a slide.
