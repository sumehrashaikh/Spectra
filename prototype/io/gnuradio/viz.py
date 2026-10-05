"""GNU Radio spectrum / waterfall visualization bridge.

This is the *visualization-only* GNU Radio entry point: it runs the
headless FFT flowgraph shipped in the standalone ``gnuradio_integration``
package and returns the resulting PSD and STFT waterfall matrices.  It
deliberately does **not** touch demodulation, costas loops, constellation
decoding or symbol mapping, which stay on the NumPy DSP path.

Contract
--------
* ``gnuradio_available()`` reports whether the flowgraph can actually be
  executed (GNU Radio is an optional dependency, never imported into this
  process; the flowgraph always runs in a subprocess).
* ``compute_spectrum_waterfall(...)`` returns ``status="ok"`` with
  ``backend_used="gnuradio"`` only when a subprocess ran the real
  flowgraph and produced data.  Every failure returns
  ``status="unavailable"``/``"error"`` with ``backend_used="numpy"`` and a
  human-readable reason, so the caller can fall back to the built-in
  plots and say so honestly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

__all__ = [
    "VIZ_SCRIPT",
    "compute_spectrum_waterfall",
    "gnuradio_available",
    "gnuradio_python",
    "viz_script_path",
]

#: Source root (".../SpectraV2") -- the directory that holds ``prototype/``.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Environment overrides (documented; no implicit guessing beyond these).
_ENV_PYTHON = "SPECTRA_GNURADIO_PYTHON"
_ENV_SCRIPT = "SPECTRA_GNURADIO_VIZ_SCRIPT"

#: Name of the headless flowgraph inside the integration package.
_VIZ_SCRIPT_NAME = "spectra_viz_headless.py"

_MAX_INPUT_SAMPLES = 1 << 16  # 65536 complex samples is plenty for an STFT
_CHECK_TIMEOUT = 15.0
_DEFAULT_TIMEOUT = 30.0

_availability_cache: dict[str, tuple[bool, str]] = {}


def viz_script_path() -> Path | None:
    """Locate the headless visualization flowgraph.

    Search order: the ``SPECTRA_GNURADIO_VIZ_SCRIPT`` override, then the
    standalone package next to this repository (``gnuradio_integration``
    and ``gnuradio``), then a copy dropped beside ``prototype``.
    """
    override = os.environ.get(_ENV_SCRIPT)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            _REPO_ROOT / "gnuradio_integration" / "gnuradio" / "headless" / _VIZ_SCRIPT_NAME,
            _REPO_ROOT / "gnuradio" / "headless" / _VIZ_SCRIPT_NAME,
            _REPO_ROOT / "prototype" / "gnuradio" / "headless" / _VIZ_SCRIPT_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def gnuradio_python() -> str:
    """Python executable used to run the flowgraph in a subprocess.

    GNU Radio lives in its own environment (typically radioconda), so the
    interpreter is resolved explicitly rather than assumed to be the one
    running Spectra.
    """
    env_val = os.environ.get(_ENV_PYTHON)
    if env_val and Path(env_val).is_file():
        return env_val

    candidates = [
        Path(os.path.expanduser("~/radioconda/python.exe")),
        Path(os.path.expanduser("~/radioconda/bin/python")),
        Path(os.path.expanduser("~/miniforge3/envs/gnuradio/bin/python")),
        Path(os.path.expanduser("~/mambaforge/envs/gnuradio/bin/python")),
    ]
    for user_dir in (os.environ.get("USERPROFILE"), os.environ.get("HOME")):
        if user_dir:
            candidates.append(Path(user_dir) / "radioconda" / "python.exe")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return env_val or sys.executable


def gnuradio_available() -> tuple[bool, str]:
    """Whether the flowgraph can run: ``(available, detail)``.

    The check executes ``<script> --check`` once per interpreter and is
    cached; ``gnuradio`` itself is never imported into this process.
    """
    script = viz_script_path()
    if script is None:
        return False, (
            "GNU Radio visualization flowgraph not found "
            f"(expected {_VIZ_SCRIPT_NAME} under gnuradio_integration/gnuradio/headless)"
        )

    python = gnuradio_python()
    cache_key = f"{python}|{script}"
    if cache_key in _availability_cache:
        return _availability_cache[cache_key]

    try:
        proc = subprocess.run(
            [python, str(script), "--check"],
            capture_output=True,
            text=True,
            timeout=_CHECK_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        result = (False, f"GNU Radio check failed: {exc}")
    else:
        stdout = (proc.stdout or "").strip()
        if proc.returncode == 0 and "GNU Radio" in stdout:
            result = (True, stdout)
        else:
            detail = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
            result = (False, f"GNU Radio unavailable: {detail}")

    _availability_cache[cache_key] = result
    return result


def compute_spectrum_waterfall(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """Compute PSD + STFT waterfall with the GNU Radio headless flowgraph.

    Returns a dict that always carries ``status`` and ``backend_used``:
    ``status="ok"`` and ``backend_used="gnuradio"`` only on a successful
    flowgraph run, otherwise an honest fallback record with ``reason``.
    """
    samples = np.asarray(samples).reshape(-1)
    if samples.size == 0:
        return _unavailable("signal is empty")
    if sample_rate is None or float(sample_rate) <= 0:
        return _unavailable("sample rate is unavailable")
    if samples.size < fft_size:
        return _unavailable(
            f"signal length ({samples.size}) shorter than FFT size ({fft_size})"
        )

    available, detail = gnuradio_available()
    if not available:
        return _unavailable(detail)

    script = viz_script_path()
    assert script is not None  # guaranteed by gnuradio_available()

    python = gnuradio_python()
    tmp_dir = Path(tempfile.mkdtemp(prefix="spectra_viz_"))
    try:
        input_file = tmp_dir / "input.cf32"
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        iq = np.asarray(samples[:_MAX_INPUT_SAMPLES], dtype=np.complex64)
        iq.tofile(str(input_file))

        cmd = [
            python,
            str(script),
            "--input-file", str(input_file),
            "--output-dir", str(out_dir),
            "--samp-rate", str(float(sample_rate)),
            "--fft-size", str(int(fft_size)),
        ]

        started = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.perf_counter() - started

        if proc.returncode != 0:
            detail = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
            return _unavailable(f"flowgraph failed: {detail}")

        psd_file = out_dir / "psd_frames.f32"
        if not psd_file.is_file() or psd_file.stat().st_size == 0:
            return _unavailable("flowgraph produced no output frames")

        raw = np.fromfile(str(psd_file), dtype=np.float32)
        n_frames = raw.size // fft_size
        if n_frames == 0:
            return _unavailable("flowgraph produced no complete FFT frame")

        frames = raw[: n_frames * fft_size].reshape(n_frames, fft_size)
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / float(sample_rate)))

        # PSD: power averaged over time.
        avg_power = frames.mean(axis=0) / float(fft_size)
        psd_db = 10.0 * np.log10(np.maximum(avg_power, 1e-15))

        # Waterfall: one row per retained time slice.
        n_slices = int(min(n_frames, 64))
        if n_frames > n_slices:
            indices = np.linspace(0, n_frames - 1, n_slices).astype(int)
            sub_frames = frames[indices, :]
        else:
            sub_frames = frames
        # Transposed to (frequency bins, time slices) so rows line up with
        # the frequency axis and columns with the time axis.
        power_db = 10.0 * np.log10(np.maximum(sub_frames.T, 1e-12))
        times = np.linspace(
            0.0, iq.size / float(sample_rate), sub_frames.shape[0]
        )

        return {
            "status": "ok",
            "backend": "gnuradio",
            "backend_used": "gnuradio",
            "source": "GNU Radio",
            "fft_size": int(fft_size),
            "n_frames": int(n_frames),
            "spectrum": {
                "freqs": freqs.tolist(),
                "magnitude": np.sqrt(avg_power).tolist(),
                "power": avg_power.tolist(),
                "power_db": psd_db.tolist(),
                "n_fft": int(fft_size),
                "window": "Blackman-Harris",
                "unit": "Relative power density (dB, uncalibrated)",
                "source": "GNU Radio",
            },
            "waterfall": {
                "freqs": freqs.tolist(),
                "times": times.tolist(),
                "power_db": power_db.tolist(),
                "n_fft": int(fft_size),
                "source": "GNU Radio",
            },
            "provenance": {
                "command": cmd,
                "duration_s": round(duration, 4),
                "gnuradio": detail,
                "input_samples": int(iq.size),
            },
        }
    except subprocess.TimeoutExpired:
        return _unavailable(f"flowgraph timed out after {timeout:.0f} s")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return _unavailable(f"flowgraph error: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _unavailable(reason: str) -> dict:
    """Fallback record: the caller draws the NumPy plots instead."""
    return {
        "status": "unavailable",
        "backend": "numpy-fallback",
        "backend_used": "numpy",
        "source": "NumPy",
        "spectrum": None,
        "waterfall": None,
        "reason": reason,
    }
