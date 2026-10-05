# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: semantic.

## [Unreleased]

### Changed (final cleanup, docs and repository hygiene — 2026-10-05)

- **Dead code removed.** Deleted the empty `modulation/psk.py`,
  `modulation/qam.py` and `modulation/fsk.py` stubs (demod kernels live
  in `modulation/demodulator.py` + `modulation/digital.py`), the
  superseded `ml/dataset_v1.py`, and three uncollected probe scripts
  (`tests/constellation_test.py`, `tests/constellation_snr_test.py`,
  `tests/snr_benchmark.py`) whose functionality is covered by the real
  suite.
- **Repository hygiene.** Training scratch candidates
  (`ml/_*.npz`/`_*.config.json`), generated analysis exports
  (`/*_analysis.json`) and agent/client local state (`.freebuff/`) are
  now ignored; the evidence-figure PNG is explicitly *not* ignored so
  the documentation pack ships.
- **Version alignment.** `pyproject.toml` now declares `2.2.0`,
  matching `prototype.__version__` (it had been left at 2.1.0).
- **Documentation brought to the final state.** README rewritten for
  the current feature set (six-scheme FEC, four interleaver families,
  frame layer, GNU Radio bridge, ML v3, external dataset, theme,
  inspectors); `docs/USER_GUIDE.md` gained GNU Radio and external-
  dataset sections, the full FEC/interleaving and ML-v3 material, and
  fixed CLI examples (`--fec-scheme`, `spectra report`); the GUI guide
  covers the theme toggle, symbol/bit inspectors, FEC/interleaving/
  frame controls, GNU Radio tab and the honest ML row;
  `docs/VALIDATION.md` records the 538-test run and the external-input
  policy; `docs/ARCHITECTURE.md`'s data-flow diagram was repaired and
  its module table extended; `docs/SIH_REQUIREMENTS.md` refreshed.
- **Evidence pack annotated as a snapshot.** `docs/evidence/EVIDENCE.md`
  is explicitly labelled as generated at commit `8b697cc` (435 tests)
  with a pointer to the current status, its R08 evidence list no longer
  cites the deleted empty modules, and the deck pack/PUBLIC_DATASET
  notes state that the `tools/sih_deck/*` generators are not shipped.

### Added (in this changeset, first-time tracked files)

- `gnuradio_integration/` — standalone GNU Radio flowgraphs + headless
  scripts used by the spectrum/waterfall bridge.
- `prototype/external_validation/` + `prototype/cli_dataset.py` — the
  optional real-world HDF5 dataset harness behind `spectra
  dataset-inspect` / `spectra dataset-eval` (`cli.py` imports it, so
  this was required for a fresh checkout to run at all).
- `prototype/gui/theme.py` (light/dark theme),
  `prototype/io/gnuradio/viz.py` (subprocess visualization bridge),
  `prototype/ml/torch_train.py` (ML v3 PyTorch trainer) and
  `prototype/ml/validation.py` (artifact validation via `spectra
  ml-eval`).
- `prototype/dataset/README.md` + config (the `.h5` data itself stays
  out of git), the SIH deck pack (`docs/SIH26147_SPECTRA_DECK.md`,
  `.pptx`) and the generated evidence pack (`docs/evidence/`).
- Eight new test modules (external validation, GNU Radio viz, theme/GUI,
  ML evidence/v3/validation, reference-free read-out,
  reference-validated FEC).

### Fixed

- **Uncoded QPSK/BFSK demos could not measure a BER (now they do).**
  The demo transmitter's hand-written QPSK level table placed two
  quadrants on the wrong bit words (the `-I` half was Grey-reversed
  relative to `modulation.demodulator.qpsk_decision`, which the table's
  own comment claimed to follow), so the *receiver* was bit-exact only
  up to a per-quadrant permutation and the measured BER plateaued near
  0.25 — no synchronization or ambiguity search could push past it. The
  QPSK and 8-PSK tables are now *derived* from the receiver's decision
  kernel (evaluate the kernel on the ideal constellation, read off the
  word→symbol map it implements), so the fixture cannot drift from the
  receiver again, and a regression test round-trips every word through
  the kernel. With the mapping correct: an uncoded QPSK capture measures
  **BER 0.00 (0/1004 bits)**, the FSK demo **BER 0.00 (0/1000)**, and
  `reference_is_usable` now writes reference sidecars for QPSK and BFSK
  (8-PSK stays excluded: those captures are classified as 16-QAM, so the
  decision device applied to them is the wrong one and a reference there
  would only report a misleading ~0.45).
- **QPSK/8-PSK BER and codeword alignment were searched on the wrong
  lattice.** The BER and alignment searches only tried 90-degree folds
  for QPSK; its axis-aligned twin (0/90/180/270) is an equally valid
  quadrature lattice and a blind M-th-power phase estimate can lock onto
  either, so a resolvable stream could be reported at ~0.24 BER. Both
  searches now try 45-degree steps, and 8-PSK — previously compared
  *directly* (no fold, no origin search, BER capped near 0.45) — got the
  same rotation × origin treatment. Max symbol-origin search widened
  from ±16 to ±24 symbols.
- **The synchronizer was chosen from the coarse label and never
  revisited.** A QPSK capture is labelled 16-QAM by the coarse waveform
  stage, so the first synchronization optimizes the wrong lattice and the
  recovered cloud is smeared (measured: mean lattice distance 0.41 vs
  0.03 after the fix) — the constellation tab, the EVM estimate and the
  decision chain all inherited that. `_classify_and_sync` now runs one
  corrective re-sync when the fine (constellation-domain) classifier
  corrects the coarse label, evaluating the matched-filter + lattice-fit
  synchronizer against the corrected modulation's own ideal lattice and
  keeping it only when it is *materially* better (≥5% tighter, scored by
  the same lattice-fit criterion the synchronizer optimizes; a marginal
  gain keeps the primary chain so the recovered symbol count stays
  stable). The run is recorded as `synchronization.refinement` and the
  classification stage as `fine_refined`. The lattice-fit synchronizer
  itself (`core/synchronization.py::synchronize_qam_signal`) gained an
  optional `lattice`/`lattice_label`, and its timing/phase stages now
  score against that lattice.
- **Wrong ML predictions (train/inference preprocessing mismatch).**
  `ml/train.py` trained on the raw, amplitude-varying frames returned by
  `build_dataset`, while the runtime (`ml/cnn.py`) normalises every frame
  to unit RMS — the same convention the artifact declares
  (`normalization="unit_rms"`). Measured on a fresh 160-frame holdout the
  shipped weights scored **37.5% on raw frames but only 21.2% on the
  unit-RMS frames the pipeline actually feeds them**; the trainer now
  normalises the dataset so training and inference share one
  representation, and the training summary records it. The shipped
  artifact was replaced with the warm-started, normalization-correct
  retrain (declared holdout 0.344): **accuracy on the representation the
  runtime actually feeds the network rose from 0.229 to 0.422** on a
  fresh 192-frame holdout. The CNN is still below the 0.60 validation
  floor, so its output is still reported as evidence - but it is no
  longer handicapped by the mismatch, and the artifact now declares the
  normalization it was trained under.
- **ML labels were silently discarded by fusion.** The CNN vocabulary
  spells constellations `16QAM`/`8PSK` while the deterministic classifier
  uses `16-QAM`/`8-PSK`, so `fuse_classification` rewrote every QAM/PSK
  ML prediction to `Unknown` and the ML could never corroborate or
  contradict the DSP result. A shared canonicalisation
  (`ml.fusion.canonical_modulation`) now normalises both sides, and a
  label outside the DSP vocabulary (AM-DSB, PAM4, ...) is recorded as
  evidence instead of being reported as a rival answer.
- **ML output could no longer masquerade as a classification.**
  `predict_modulation` now reports `validated` (from the artifact's own
  holdout accuracy), `presented_as` (`prediction` only when the model is
  validated *and* the capture clears the confidence/agreement floors) and
  the canonical top class. The GUI row reads `ML Prediction: ...` for a
  supported label and `ML (CNN): evidence only - top class X (Y%), model
  holdout accuracy 31%` otherwise, so a confident-but-wrong argmax is no
  longer shown as a result. Top-3 scores are shown in the summary dialog.
- **Joint interleaver/FEC search was block-only.**
  `_reference_validated_fec_search` probed `none` plus block depths, so
  `16-QAM concatenated pseudo_random` stayed `UNRESOLVED` in AUTO mode
  even with a reference. The hypothesis set now covers every implemented
  family (block depths, seeded pseudo-random permutations, convolutional
  strides, the square diagonal) and the accepted hypothesis records its
  family/parameter (`interleaving_result.family`, `.best_param`,
  `evidence.hypothesis`).
- **GUI: analysis with the ML toggle on failed with `NameError`.**
  `MainWindow._pipeline_summary_text` passed a local `modulation` that does
  not exist in that method (it holds the DSP result in
  `self.modulation_result`), so every completed analysis with
  `ML assist (CNN)` checked — single or batch — crashed in
  `_on_pipeline_finished` with `name 'modulation' is not defined` and the
  user saw "analysis failed". Fixed, and the ML summary/parameter paths now
  have direct regression tests in `tests/test_gui_theme_viz.py` (the ML-on
  GUI path previously had no test, which is why it slipped through).
- **The Auto FEC row read `not run` after a scheme was validated and
  decoded.** The reference-validated joint search resolved and decoded
  the scheme against the transmitted reference but never wrote the
  `fec_identification` record the GUI/JSON row reads, so fully resolved
  captures (`16-QAM concatenated + block`, `reedsolomon + block`, ...)
  still showed `Auto FEC: not run`. The accepted reference identification
  is now stored in the same slot with `confirmed_by =
  "reference_payload_agreement"`, and the row renders its heuristic score
  (0-100) as a percentage instead of a bare number.
- **The GUI ML fallback row could still show the CNN argmax (with a
  stray `)`).** When a payload carried no explicit mirror fields (an
  older or partial result), `_format_ml_summary` fell back to
  `predicted_class` and appended a dangling `)`. It now falls back to the
  DSP classification, labels it `— from DSP analysis (CNN said ...)`, and
  shows a mirror's own confidence next to a mirrored class only.
- **Viterbi decoding was the bottleneck of every FEC hypothesis test.**
  `fec/convolutional.py` decoded in a nested Python loop (2.6 s for a
  2572-bit codeword). It now uses predecessor-indexed reverse tables and
  a vectorised add-compare-select (~45x faster: 59 ms), which is what
  makes a full interleaver hypothesis search affordable.

### Added

- **Symbols and bit-stream inspectors in the GUI.** Every analysis now
  offers `View Symbols (I/Q)` and `View Bits / BER` buttons (Constellation
  tab and the FEC / BER group of the detected-signal panel). Symbols opens
  a small read-only window listing the recovered synchronized symbols
  (`# / I / Q / |S|`, first 1000, with a pointer to the JSON export for the
  full set); Bits / BER opens the same window with the BER row, the
  interleaving/FEC verdict and every bitstream the receiver produced
  (demodulated, codeword-aligned, deinterleaved, FEC-decoded) with length,
  ones/zeros counts and a 512-bit preview. Both are exportable via
  Copy / Save…, and say what to do when nothing has been analyzed yet.
- **Reference-free BER read-out (EVM estimate).** Without a transmitted
  reference the BER row used to read `No reference loaded`; the demod
  stage now derives an EVM-based link-quality estimate from the recovered
  constellation (`evm`, `snr_db`, `ber_estimate`, `method="evm_estimate"`,
  explicitly *not* a measured BER) and every BER row shows it. The QPSK
  estimate is measured against the closer of the two lattice phases the
  blind receiver cannot distinguish, so a clean capture is not reported as
  a huge EVM because of a phase convention.
- **Interleaving read-out states what was applied.** A capture whose
  demodulated stream already matches its reference now reports
  `NONE (no deinterleaving required)` with `evidence.mode =
  "reference_agreement"` instead of `UNRESOLVED`, and the type/depth rows
  explain the remaining honest outcomes (`not identifiable from a blind
  capture (needs a bit reference or a decodable FEC code)`, `— (no depth
  claimed, none applied)`). The completion summary shows the same BER and
  interleaving lines the results panel does.
- **ML row mirrors the DSP classification when the CNN cannot
  classify.** `_mirror_ml_display` puts the deterministic class (and its
  confidence) in the ML row for an unvalidated network
  (`display_class`, `display_source="dsp_mirror"`, `mirrored_from_dsp`) and
  names the network's own answer after it (`ML Prediction: QPSK (97%) —
  from DSP analysis (CNN said 64QAM)`), while `ml_raw_class` and `top3`
  keep the network's verdict in the JSON export and the provenance.
- **GNU Radio spectrum + waterfall (visualization only).**
  `io/gnuradio/viz.py` runs the headless FFT flowgraph shipped in the
  standalone `gnuradio_integration` package in a subprocess
  (`SPECTRA_GNURADIO_PYTHON`, never importing `gnuradio` into Spectra)
  and returns the PSD and STFT waterfall matrices. Every failure returns
  `backend_used="numpy"` plus a human-readable reason, and the GUI draws
  the built-in NumPy plots in that case. No demodulation, carrier
  recovery or constellation work moved off the NumPy DSP path.
- **GUI light/dark theme.** A corner button in the title row switches
  themes (`gui/theme.py`: palette + stylesheet + Matplotlib canvas
  recolouring). Light is the original look and applies no stylesheet, so
  switching back restores exactly the previous appearance; the toggle
  adds no row and moves no existing control.
- **GNU Radio spectrum/waterfall toggle** in the GNU Radio tab with a
  status row that names the backend actually used ("GNU Radio 1024-point
  FFT, 8 frames in 0.64 s" or "built-in NumPy plots (<reason>)").
- **Tests**: `tests/test_gnuradio_viz.py` (bridge contract + matrix
  orientation + plot helpers), `tests/test_ml_evidence.py` (label
  canonicalisation, comparability, evidence gate) and
  `tests/test_gui_theme_viz.py` (offscreen theme + GNU Radio toggles),
  plus two reference-validated joint-search regressions
  (pseudo-random family resolved; an un-interleaved coded capture is not
  forced through a deinterleaver).

### Added (earlier)

- **FEC / de-interleaving closure (SIH-147)**:
  - Four de-interleaver families (`block`, `convolutional`, `diagonal`,
    `pseudo-random`) are now selectable via `FECConfig.interleave_family`,
    the pipeline MANUAL path, the new CLI `--interleave-family` flag, and a
    GUI "Family:" selector. AUTO identification stays block-only — no
    fabricated structural evidence for the other families.
  - New FEC schemes in the existing registry: `reedsolomon` (shortened
    systematic RS over GF(256), Berlekamp-Massey/Chien decode, t=4), `ldpc`
    (compact (3,6)-regular LDPC with hard-decision bit-flipping), and
    `concatenated` (RS outer + K=7 rate-1/2 convolutional inner).
  - Result/GUI now expose the recovered information: the real decoded
    bitstream plus `decoded_bit_count`, and dedicated "Recovered bits" and
    "Sync word" rows.
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
- **Protocol / frame layer** (`prototype/protocol/`): a shared, explicit-
  configuration frame decoder that runs *after* demodulation (and after
  an optional FEC pass) on the recovered bit stream. It reuses the
  normalized-correlation sync-word search in `dsp.correlation` and
  declares protocols via `FrameConfig` (name, sync word, payload size,
  optional CRC). Add a protocol without touching the pipeline: the
  pipeline exposes `AnalysisConfig.protocol` and `analyze_capture()`
  accepts it. Results (`result.protocol`) carry a first-class
  `Unknown`/no-sync outcome, an honest confidence, and the payload;
  CLI gains `--sync-word` + `--data-bytes`, and the GUI gains a
  "Protocol" parameter row that reports the matched protocol and
  whether sync was found. Verified: synthetic frame recovered;
  no-sync capture reported `Unknown` (never guessed).
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
- **GUI runs the V2 pipeline on a background thread
- **Automatic FEC identification (Phase 2)**: `prototype/fec/identification.py` is a deterministic candidate evaluator (repetition3, hamming74, conv12, none). Clean supported FEC codewords are detected (80 confidence), corrupted/ambiguous inputs are UNKNOWN, insufficient input is UNKNOWN, `MIN_CONFIDENCE = 60`. Integration: `pipeline.py` runs `fec_identification` right after demodulation with `identify_fec(bits, reference_bits)`, decodes only when `AUTO_DETECTED`, and always preserves the result on `result.demodulation["fec_identification"]` plus the `fec_identification` provenance step.
- **Configuration**: `AnalysisConfig.fec.mode` is `auto | manual | none` (enum-backed, frozen dataclass). AUTO = identification + decode on evidence (manual explicit scheme untouched), MANUAL = explicit scheme authoritative, NONE = no identification/decoding. Serialization and config-to-dict include the new mode.
- **CLI**: `--fec-mode auto | manual | none`.
- **GUI**: `FEC MODE` selector (Auto/Manual/None), result panel shows detected scheme, confidence, corrected errors, residual estimate, validation, and candidate evidence.
**
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

- **ML stage never ran** (`pipeline.py`): the ML classification/fusion
  stage called `predict_modulation` without importing it, so enabling the
  CNN (GUI checkbox or CLI `--ml`) only ever appended "ML
  classification/fusion failed: name 'predict_modulation' is not defined"
  and produced no ML result. Added the missing import; the stage now runs
  and records `result.ml` / `result.fusion`.
- **CLI `analyze` silently ignored `--sync-word`** (`cli.py`): the command
  built a `FrameConfig` from `--sync-word/--data-bytes` and then dropped it
  (unlike `demodulate`/`report`). The protocol config is now forwarded, so
  the frame/sync stage runs.
- **CLI `analyze --ml/--ml-fusion/--labels` were dead flags** (`cli.py`,
  `pipeline.py`): they were parsed but never reached `AnalysisConfig.ml`.
  `analyze_capture` now accepts `ml_enabled`/`ml_fusion`/`labels_path` and
  maps them onto the ML config, and `predict_modulation`/`get_engine`
  accept an optional `labels_json` override.
- **CLI `analyze --source gnuradio` raised** (`cli.py`): the branch used
  `analyze_samples`/`processing_mode_config` without importing them and
  passed a `reference_bits_path` argument `analyze_samples` does not
  accept. It now imports both, loads the reference bits, and applies the
  same FEC / interleaving / protocol / ML overrides as the file path.
- **CLI `demodulate` omitted the protocol result** (`cli.py`): the
  sync-word stage ran but its result was left out of the JSON payload.
- **Duplicate/dead `set_defaults(func=cmd_train)` calls** removed from the
  argument parser.
- **Pseudo-random interleaver was neither seeded nor invertible**
  (`fec/interleaving.py`): `seed` was ignored (all seeds produced the same
  permutation) and `pseudo_random_deinterleave` re-applied the forward
  permutation instead of inverting it, so round-trips failed. Replaced with
  a seeded injective affine key and a true inverse.
- **Recovered bitstream was discarded** (`pipeline.py`): both the auto and
  explicit FEC paths stored the FEC output **bit count** under
  `decoded_bits` instead of the recovered bits. Now the real bitstream is
  stored alongside `decoded_bit_count`.
- **CLI analyze/demodulate/report crashed on FEC/interleaving flags**
  (`cli.py`, `pipeline.py`): `analyze` passed unsupported `load_signal`
  kwargs, while `demodulate`/`report` read `args.interleaving_mode` that was
  never defined. Added a shared `_add_fec_arguments()` helper and mapped the
  overrides onto `FECConfig`.
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

### Added (final readiness pass, 2026-09-29)

- **Deterministic demo set** (`tests/demo_captures.py`): eleven cases
  covering PSK (QPSK, 8-PSK), QAM (16-QAM), FSK (BFSK), convolutional /
  Reed-Solomon / concatenated FEC, and all four de-interleaver families
  (block, convolutional, diagonal, pseudo-random). `write_capture()` also
  emits the `<stem>.reference.npz` sidecar the BER/alignment stage expects,
  so the GUI and CLI demonstrate the clean decoded chain where the backend
  supports it. An FSK builder (`build_fsk_capture`) was added.
- **`docs/SIH_REQUIREMENTS.md`** — the authoritative SIH-147 requirement
  matrix with per-row evidence and honest limitations; the user guide §9.5
  summary and this changelog link to it.
- **`tests/test_final_integration.py`** (8 tests): regressions for the CLI /
  pipeline integration bugs above, the ML stage, and the FSK + reference
  sidecar demo coverage.

### Verified (final readiness pass, 2026-09-29)

- `QT_QPA_PLATFORM=offscreen python -m pytest -q` → **433 passed**.
- Focused FEC / interleaving / GUI / CLI / ML set → 236 passed.
- Offscreen GUI walkthrough over all 11 demo captures: every case classifies
  correctly, all six visualisation tabs render, FEC/recovered/sync/BER rows
  populate, provenance present, JSON export and batch candidate detail
  verified, GNU Radio acquire works via the synthetic fallback.

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
