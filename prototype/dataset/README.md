# External real-world IQ dataset (optional)

This directory holds the **downloaded Mendeley dataset** described in
*"Real-World IQ Dataset for Automatic Radio Modulation Recognition under
Multipath Channels"* (Belousov & Ronkin; the paper is bundled here as a
`.docx`). It is used by SPECTRA as an **optional external real-world
validation input** — it does not change the pipeline, the DSP classifier,
the ML runtime or any default.

## Files

| File | Contents |
|---|---|
| `subset_train.h5` | 400,000 frames (≈591 MB) |
| `subset_val.h5` | 80,000 frames (≈121 MB) |
| `subset_test.h5` | 80,000 frames (≈121 MB) |
| `dataset.config.json` | path/metadata mechanism used by the loader |
| `README.txt` | original dataset note shipped with the download |
| `*.docx` | the dataset paper |

Each HDF5 file contains:

* `X` — `(N, 1024, 2)` float16 IQ pairs
* `y_mod` — modulation id (`0 BPSK, 1 QPSK, 2 QAM, 3 GMSK, 4 OFDM, 5 NBFM, 6 WBFM`)
* `y_chan` — channel id (`0 clean`, `1 multipath`)
* `y_snr` — labelled SNR in dB (`20, 22, 24, 26, 28, 30`)

## Not committed to Git

The three `.h5` files are **~833 MB of binary data and must not be
committed**. `dataset/.gitignore` excludes `*.h5` / `*.hdf5` here, and the
files are currently untracked. If they ever get staged accidentally:

```bash
git rm --cached prototype/dataset/*.h5
```

The code, config and README in this repository *are* committed; the data
is not. To relocate the data, either update `dataset.config.json` or pass
`--dataset-dir <path>` to the CLI commands below.

## Usage

```bash
# Inspect shapes, labels, balance, SNR and channel distribution.
python -m prototype.cli dataset-inspect

# Run SPECTRA's existing preprocessing + DSP classification + ML assist
# on a sampled subset and break results down by modulation / channel / SNR.
python -m prototype.cli dataset-eval --subset test --per-cell 3
```

Both commands are completely optional and require `h5py` (an optional,
lazily imported dependency). Nothing else in SPECTRA imports this package.

## Honest limits (read before quoting any number)

* **No low-SNR claim.** The dataset only spans **20–30 dB**. Any result
  computed on it says nothing about SPECTRA's behaviour at 0–18 dB, where
  the repository's own benchmarks operate.
* **The 84.3% figure is the dataset authors'**, produced by *their*
  TensorFlow baseline on *their* split. It is not a SPECTRA result and is
  never reported as one.
* **Only 2 of the 7 classes are in SPECTRA's deterministic vocabulary**
  (`BPSK`, `QPSK`). `QAM`, `GMSK`, `OFDM`, `NBFM` and `WBFM` are marked
  *unsupported* rather than scored as wrong.
* **Domain shift.** These are 2 MSps / 2.4 GHz hardware captures of
  real-bandwidth signals; SPECTRA's classifier and synthetic training data
  assume an 8 kHz baseband. Frames are also 1024 samples versus the ML
  runtime's 512, and the dataset's class set is coarse.
