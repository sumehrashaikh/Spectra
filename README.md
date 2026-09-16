<<<<<<< HEAD
# SIH Signal Analyzer

Prototype signal-analysis application for the SIH project.

The current prototype is built with Python, NumPy, SciPy, Matplotlib and PySide6.

---

## Current Capabilities

### Input

- WAV file loading
- Mono WAV support
- Stereo IQ WAV support

### Visualization

- Time-domain waveform
- FFT spectrum
- Waterfall / spectrogram
- IQ constellation

### Signal analysis

- Sample-rate extraction
- RMS and peak measurement
- Spectral peak detection
- Signal detection
- Multiple-signal detection
- Signal selection
- Signal isolation
- Dominant-frequency estimation
- Initial bandwidth estimation
- Noise-floor estimation

### Modulation

Current baseline classifier supports:

- BPSK
- QPSK
- 16-QAM
- BFSK

### Timing

A baseline autocorrelation-based symbol-rate estimator is included.

IMPORTANT:
Timing recovery is still experimental for noisy IQ data.

---

# Requirements

Python 3.13+

Recommended environment:

```powershell
python -m venv .venv

Activate it in PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install numpy scipy matplotlib PySide6 scikit-learn

Running the Prototype

From the project root:

C:\Users\eiraa\SIH_Project

run:

python prototype\main.py
Test 1 — Simple WAV

Use:

prototype\data\input\test_signal.wav

This test contains two tones:

3000 Hz
7000 Hz

Expected behavior:

Spectrum shows the two tones
Waterfall shows both tones
Detected Signals shows approximately 3000 Hz and 7000 Hz
Selecting a signal allows isolation
Test 2 — BPSK IQ

Use:

prototype\data\input\bpsk_iq.wav

Expected behavior:

Stereo WAV is interpreted as I/Q
IQ constellation becomes available
Signal is detected around baseband
BPSK baseline classification is available

Known generation parameters:

Sample rate: 8000 samples/s
Symbol rate: 100 symbols/s
Samples/symbol: 80
SNR: 15 dB

NOTE:
The current timing-recovery implementation may not yet recover the exact
100 symbols/s value. This is a known development area.

Basic Workflow
Click Open WAV
Select a test signal
Click Analyze Signal
Review detected signals
Select a signal from the table
Click Isolate Selected
Click Analyze Selected

Current Development Status
Completed
WAV input
Basic signal visualization
Spectrum
Waterfall
Signal detection
Multiple signal detection
Signal selection
Signal isolation
IQ constellation support
Baseline modulation classifier
In Progress
Robust symbol-rate estimation
Timing recovery
Per-signal modulation analysis
Robust SNR estimation
Planned
BPSK/QPSK/QAM/FSK demodulation
Bitstream extraction
De-interleaving
FEC decoding
Bit-stream correlation
CNN/LSTM based modulation classification
Raw IQ format support
Final integrated GUI
=======
# SIH-Signal-Analyzer
>>>>>>> 723010e2c510f090c2795c1f4db05a861c88b9fd
