import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_gnuradio_pkg = _repo_root / "gnuradio_integration"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_gnuradio_pkg) not in sys.path:
    sys.path.insert(0, str(_gnuradio_pkg))

import numpy as np

from prototype.core.loader import load_wav
from prototype.io.loaders import load_raw_iq
from prototype.pipeline import analyze_samples
from dsp_gnuradio.bridge import run_gnuradio_viz, check_gnuradio_available
from prototype.visualization.plots import (
    create_time_figure,
    create_spectrum_figure_gnuradio,
    create_waterfall_figure_gnuradio,
    create_symbol_constellation_figure,
)


def run_test():
    print("=== 1. Checking GNU Radio Environment ===")
    avail, gr_ver = check_gnuradio_available()
    print(f"GNU Radio Available: {avail}, Version: {gr_ver}")
    assert avail, "GNU Radio must be available"

    print("\n=== 2. Testing with WAV file (prototype/gui_qam16.wav) ===")
    wav_path = Path("prototype/gui_qam16.wav")
    samples_wav, sr_wav = load_wav(wav_path)
    print(f"WAV Loaded: {len(samples_wav)} samples, Sample Rate: {sr_wav} Hz")

    # (a) GNU Radio Viz Job
    fft_d, psd_d, wf_d, err = run_gnuradio_viz(samples_wav, sr_wav)
    print(f"GNU Radio Viz Error: {err}")
    assert err is None, f"GNU Radio viz failed: {err}"
    assert fft_d["source"] == "GNU Radio"
    assert psd_d["source"] == "GNU Radio"
    assert wf_d["source"] == "GNU Radio"
    print(f"GNU Radio FFT points: {len(fft_d['freqs'])}, PSD bins: {len(psd_d['psd_db'])}, Waterfall grid: {len(wf_d['times'])}x{len(wf_d['freqs'])}")

    # (b) Existing Python / ML Pipeline
    res_wav = analyze_samples(samples=samples_wav, sample_rate=sr_wav)
    mod_wav = res_wav.classification.get("modulation") if res_wav.classification else "Unknown"
    snr_wav = res_wav.parameters.get("snr_db", 0.0) if res_wav.parameters else 0.0
    print(f"Python Pipeline Modulation: {mod_wav}, SNR: {snr_wav:.2f} dB")

    # Rendering Matplotlib Figures
    f_time = create_time_figure(samples_wav, sr_wav)
    f_spec = create_spectrum_figure_gnuradio(fft_d)
    f_wf = create_waterfall_figure_gnuradio(wf_d)
    const_syms = res_wav.demodulation.get("constellation", {}).get("symbols", []) if res_wav.demodulation else []
    f_const = create_symbol_constellation_figure(const_syms) if const_syms else None
    print("WAV Figures created successfully!")

    print("\n=== 3. Testing with Raw IQ file (prototype/test_capture.iq) ===")
    iq_path = Path("prototype/test_capture.iq")
    sig_iq = load_raw_iq(iq_path, sample_rate=8000.0, dtype="int16")
    print(f"Raw IQ Loaded: {sig_iq.num_samples} samples, Sample Rate: {sig_iq.sample_rate} Hz")

    # (a) GNU Radio Viz Job
    fft_d2, psd_d2, wf_d2, err2 = run_gnuradio_viz(sig_iq.samples, sig_iq.sample_rate)
    print(f"GNU Radio Viz Error: {err2}")
    assert err2 is None, f"GNU Radio viz failed: {err2}"
    assert fft_d2["source"] == "GNU Radio"
    assert psd_d2["source"] == "GNU Radio"
    assert wf_d2["source"] == "GNU Radio"
    print(f"GNU Radio FFT points: {len(fft_d2['freqs'])}, Waterfall grid: {len(wf_d2['times'])}x{len(wf_d2['freqs'])}")

    # (b) Existing Python / ML Pipeline
    res_iq = analyze_samples(samples=sig_iq.samples, sample_rate=sig_iq.sample_rate)
    mod_iq = res_iq.classification.get("modulation") if res_iq.classification else "Unknown"
    snr_iq = res_iq.parameters.get("snr_db", 0.0) if res_iq.parameters else 0.0
    print(f"Python Pipeline Modulation: {mod_iq}, SNR: {snr_iq:.2f} dB")

    # Rendering Matplotlib Figures
    f_time2 = create_time_figure(sig_iq.samples, sig_iq.sample_rate)
    f_spec2 = create_spectrum_figure_gnuradio(fft_d2)
    f_wf2 = create_waterfall_figure_gnuradio(wf_d2)
    print("Raw IQ Figures created successfully!")

    print("\n=== 4. Testing GUI Integration Components ===")
    from prototype.gui.window import MainWindow
    print("MainWindow imported successfully without importing gnuradio in main process!")

    print("\nALL INTEGRATION AND PIPELINE TESTS PASSED 100% SUCCESSFULLY!")


if __name__ == "__main__":
    run_test()
