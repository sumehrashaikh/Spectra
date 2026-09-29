"""SPECTRA GNU Radio Bridge.

Manages subprocess execution of headless GNU Radio flowgraphs for demodulation
and DSP visualization, symbol file verification, and truth-in-reporting provenance.
NEVER imports gnuradio into the backend Python process.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Base paths
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_DIR.parent

_candidate_gnuradio_dirs = [
    _PACKAGE_DIR / "gnuradio",
    _REPO_ROOT / "gnuradio_integration" / "gnuradio",
    _REPO_ROOT / "gnuradio",
]
GNURADIO_DIR = next((d for d in _candidate_gnuradio_dirs if d.is_dir()), _candidate_gnuradio_dirs[0])
HEADLESS_DIR = GNURADIO_DIR / "headless"
DEMOD_SCRIPT = HEADLESS_DIR / "spectra_demod_headless.py"
VIZ_SCRIPT = HEADLESS_DIR / "spectra_viz_headless.py"
MAPPING_FILE = GNURADIO_DIR / "qpsk_mapping.json"

# Load environment variables from .env
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_PACKAGE_DIR / ".env")


def get_gnuradio_python():
    """Retrieve the configured Python executable for running GNU Radio."""
    env_val = os.environ.get("SPECTRA_GNURADIO_PYTHON")
    if env_val and os.path.exists(env_val):
        return env_val
    for cand in [
        r"C:\Users\karee\radioconda\python.exe",
        os.path.expanduser("~/radioconda/python.exe"),
        shutil.which("python"),
    ]:
        if cand and os.path.exists(cand):
            return cand
    return env_val or sys.executable


get_gnuradio_python_path = get_gnuradio_python


def check_gnuradio_available():
    """Check whether GNU Radio can be executed via the headless script in a subprocess."""
    py_exe = get_gnuradio_python()
    check_script = VIZ_SCRIPT if VIZ_SCRIPT.is_file() else DEMOD_SCRIPT
    if not check_script.is_file():
        return False, f"Headless script missing: expected at {check_script}"

    try:
        cmd = [py_exe, str(check_script), "--check"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and "GNU Radio" in res.stdout:
            ver = res.stdout.strip()
            return True, ver
        return False, res.stderr.strip() or "GNU Radio check exited with non-zero code"
    except Exception as e:
        return False, str(e)


def load_qpsk_mapping():
    """Load and return verified QPSK symbol-to-bit mapping."""
    if not MAPPING_FILE.is_file():
        return None, "QPSK symbol mapping not verified (qpsk_mapping.json missing)"
    try:
        with open(MAPPING_FILE, "r") as f:
            data = json.load(f)
        if "mapping" not in data:
            return None, "Invalid qpsk_mapping.json: missing 'mapping' table"
        return data, None
    except Exception as e:
        return None, f"Failed to parse qpsk_mapping.json: {e}"


def validate_symbol_file(file_path, modulation="QPSK"):
    """Validate the raw uint8 symbol output file from the headless flowgraph."""
    p = Path(file_path)
    if not p.is_file() or p.stat().st_size == 0:
        return False, None, "Symbol file missing or empty"

    symbols = np.fromfile(str(p), dtype=np.uint8)
    if len(symbols) == 0:
        return False, None, "Zero symbols read from file"

    max_val = 3 if modulation.upper() == "QPSK" else 1
    if np.any(symbols > max_val):
        return False, None, f"Symbol values out of range (expected 0..{max_val}, got max {np.max(symbols)})"

    return True, symbols, None


def run_gnuradio_demod(iq, modulation="QPSK", sps=8.0, samp_rate=100000.0, loop_bw=0.0628, timeout=15.0):
    """Execute headless GNU Radio demodulation flowgraph via subprocess.

    Returns:
        dict: Standard demodulation result format with provenance and phase ambiguity notice,
              or Unavailable status if GNU Radio execution fails.
    """
    mod = (modulation or "").upper()
    if mod not in ("BPSK", "QPSK"):
        return {
            "status": "Unavailable",
            "backend_used": "numpy",
            "reason": f"GNU Radio flowgraph only supports BPSK/QPSK (got '{modulation}')."
        }

    if sps is None or np.isnan(sps) or sps <= 0:
        return {
            "status": "Unavailable",
            "backend_used": "numpy",
            "reason": "Symbol rate NOT_DETERMINED or invalid; cannot determine SPS for GNU Radio."
        }

    is_available, gr_info = check_gnuradio_available()
    if not is_available:
        return {
            "status": "Unavailable",
            "backend_used": "numpy",
            "reason": f"GNU Radio unavailable: {gr_info}"
        }

    # Verify QPSK mapping if QPSK
    mapping_data = None
    if mod == "QPSK":
        mapping_data, err = load_qpsk_mapping()
        if err or not mapping_data:
            return {
                "status": "NOT_DETERMINED",
                "backend_used": "numpy",
                "reason": err or "QPSK symbol mapping not verified"
            }

    py_exe = get_gnuradio_python()
    tmp_dir = tempfile.mkdtemp(prefix="spectra_gr_")
    try:
        input_file = Path(tmp_dir) / "input.cf32"
        output_file = Path(tmp_dir) / "symbols.u8"
        complex_file = Path(tmp_dir) / "complex_symbols.cf32"

        # Write complex64 IQ data (capped to 131072 for optimal latency)
        iq_c64 = np.asarray(iq[:min(len(iq), 131072)], dtype=np.complex64)
        iq_c64.tofile(str(input_file))

        cmd = [
            py_exe,
            str(DEMOD_SCRIPT),
            "--input-file", str(input_file),
            "--output-symbols-file", str(output_file),
            "--output-complex-file", str(complex_file),
            "--sps", str(float(sps)),
            "--samp-rate", str(float(samp_rate)),
            "--loop-bw", str(float(loop_bw)),
            "--modulation", mod
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.perf_counter() - t0

        if proc.returncode != 0:
            return {
                "status": "Unavailable",
                "backend_used": "numpy",
                "reason": f"GNU Radio subprocess failed (exit code {proc.returncode}): {proc.stderr.strip()}"
            }

        valid, symbols, err = validate_symbol_file(output_file, modulation=mod)
        if not valid or symbols is None:
            return {
                "status": "Unavailable",
                "backend_used": "numpy",
                "reason": err or "Invalid symbol output file"
            }

        # Read synchronized complex symbols if available
        constel_data = None
        evm_val = None
        is_locked = True
        if complex_file.is_file() and complex_file.stat().st_size >= 8:
            from backend.dsp.synchronizer import compute_evm, slice_constellation
            c_syms = np.fromfile(str(complex_file), dtype=np.complex64)
            if len(c_syms) > 16:
                # Normalize complex symbol power
                p_avg = np.mean(np.abs(c_syms) ** 2)
                if p_avg > 0:
                    c_syms = (c_syms / np.sqrt(p_avg)).astype(np.complex64)
                # Discard transient settling
                settle = min(len(c_syms) // 4, 64)
                eval_syms = c_syms[settle:]
                dec_syms, ref_const = slice_constellation(eval_syms, mod)
                evm_val = compute_evm(eval_syms, dec_syms, ref_const)
                is_locked = bool(evm_val < 0.35)
                sub_syms = c_syms[:2048]
                constel_data = {
                    "status": "ok" if is_locked else "Not synchronized",
                    "reason": None if is_locked else f"Carrier/timing sync unlocked: EVM {evm_val*100:.1f}% exceeds lock threshold (35.0%)",
                    "i": np.real(sub_syms).tolist(),
                    "q": np.imag(sub_syms).tolist(),
                    "count": len(sub_syms),
                    "evm": evm_val,
                    "locked": is_locked,
                    "phase_ambiguity": True,
                    "phase_ambiguity_note": "4-fold phase ambiguity (0, 90, 180, 270 deg)" if mod == "QPSK" else "2-fold phase ambiguity (0, 180 deg)",
                    "source": "GNU Radio",
                    "sps_used": float(sps),
                }

        # Convert symbols to bitstring
        if mod == "QPSK":
            m = mapping_data["mapping"]
            bits_list = [m[str(s)] for s in symbols]
            bits_str = "".join(bits_list)
        else:  # BPSK: 0 -> "0", 1 -> "1"
            bits_str = "".join("1" if s == 1 else "0" for s in symbols)

        if len(bits_str) < 8:
            return {
                "status": "Unavailable",
                "backend_used": "numpy",
                "reason": "Insufficient bits recovered by GNU Radio flowgraph."
            }

        ambiguity_msg = (
            "4-fold phase ambiguity (0, 90, 180, 270 deg)" if mod == "QPSK"
            else "2-fold phase ambiguity (0, 180 deg)"
        )

        return {
            "status": "ok",
            "modulation": mod,
            "backend": "gnuradio",
            "backend_used": "gnuradio",
            "gnuradio_available": True,
            "bit_count": len(bits_str),
            "bits": bits_str[:4096],
            "truncated": bool(len(bits_str) > 4096),
            "evm": evm_val,
            "locked": is_locked,
            "phase_ambiguity": True,
            "phase_ambiguity_note": ambiguity_msg,
            "constellation": constel_data,
            "provenance": {
                "command": cmd,
                "exit_code": proc.returncode,
                "duration_s": round(duration, 4),
                "sps_used": float(sps),
                "gnuradio_version": gr_info,
                "mapping_date": mapping_data.get("verification_date") if mapping_data else None
            }
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "Unavailable",
            "backend_used": "numpy",
            "reason": f"GNU Radio flowgraph timed out after {timeout} seconds."
        }
    except Exception as e:
        return {
            "status": "Unavailable",
            "backend_used": "numpy",
            "reason": f"GNU Radio execution error: {e}"
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_gnuradio_viz(iq, samp_rate, fft_size=1024, timeout=10.0):
    """Run headless visualization flowgraph via GNU Radio and return visualization dictionaries.

    Returns:
        tuple (fft_dict, psd_dict, waterfall_dict, error_msg)
        with source='GNU Radio' on success and error_msg=None,
        or (None, None, None, error_msg) if GNU Radio visualization fails.
    """
    if samp_rate is None or samp_rate <= 0:
        return None, None, None, "sample rate unavailable"

    if iq is None or len(iq) == 0:
        return None, None, None, "Signal data is empty"

    is_available, gr_info = check_gnuradio_available()
    if not is_available:
        return None, None, None, f"GNU Radio unavailable ({gr_info})"

    if not VIZ_SCRIPT.is_file():
        return None, None, None, f"Visualization script missing at {VIZ_SCRIPT}"

    py_exe = get_gnuradio_python()
    tmp_dir = tempfile.mkdtemp(prefix="spectra_viz_")
    try:
        input_file = Path(tmp_dir) / "input.cf32"
        out_dir = Path(tmp_dir) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        iq_c64 = np.asarray(iq[:min(len(iq), 65536)], dtype=np.complex64)
        if len(iq_c64) < fft_size:
            return None, None, None, f"Signal length ({len(iq_c64)}) shorter than FFT size ({fft_size})"
        iq_c64.tofile(str(input_file))

        cmd = [
            py_exe,
            str(VIZ_SCRIPT),
            "--input-file", str(input_file),
            "--output-dir", str(out_dir),
            "--samp-rate", str(float(samp_rate)),
            "--fft-size", str(int(fft_size))
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            err_detail = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
            return None, None, None, f"Subprocess failed: {err_detail}"

        psd_file = out_dir / "psd_frames.f32"
        if not psd_file.is_file() or psd_file.stat().st_size == 0:
            return None, None, None, "GNU Radio output file empty or missing"

        raw = np.fromfile(str(psd_file), dtype=np.float32)
        n_frames = len(raw) // fft_size
        if n_frames == 0:
            return None, None, None, "Zero complete FFT frames produced"

        frames = raw[: n_frames * fft_size].reshape((n_frames, fft_size))

        # Frequency axis centered
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / samp_rate))

        # 1. PSD: average across frames
        avg_power = np.mean(frames, axis=0) / fft_size
        psd_db = 10.0 * np.log10(np.maximum(avg_power, 1e-15))
        psd_dict = {
            "freqs": freqs.tolist(),
            "psd": avg_power.tolist(),
            "psd_db": psd_db.tolist(),
            "nperseg": fft_size,
            "source": "GNU Radio",
            "window": "Blackman-Harris",
            "unit": "Relative power density (dB, uncalibrated)",
            "note": "Window (Blackman-Harris) and scaling differ from NumPy Welch estimate; levels are relative and not directly comparable."
        }

        # 2. FFT: average magnitude
        mag = np.sqrt(avg_power)
        fft_dict = {
            "freqs": freqs.tolist(),
            "magnitude": mag.tolist(),
            "power": avg_power.tolist(),
            "n_fft": fft_size,
            "source": "GNU Radio"
        }

        # 3. Waterfall: matrix over time slices
        n_slices = min(n_frames, 64)
        if n_frames > n_slices:
            frame_indices = np.linspace(0, n_frames - 1, n_slices).astype(int)
            sub_frames = frames[frame_indices, :]
        else:
            sub_frames = frames

        with np.errstate(divide="ignore"):
            pdb = 10 * np.log10(sub_frames.T + 1e-12)

        times = np.linspace(0, len(iq_c64) / samp_rate, sub_frames.shape[0]).tolist()

        waterfall_dict = {
            "freqs": freqs.tolist(),
            "times": times,
            "power_db": pdb.tolist(),
            "nfft": fft_size,
            "source": "GNU Radio"
        }

        return fft_dict, psd_dict, waterfall_dict, None
    except subprocess.TimeoutExpired:
        return None, None, None, f"GNU Radio flowgraph timed out after {timeout} seconds"
    except Exception as exc:
        return None, None, None, f"GNU Radio visualization failed: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
