"""GNU Radio spectrum / waterfall bridge (visualization only).

GNU Radio is an optional dependency: these tests never require it to be
installed.  They pin the honest contract (success -> ``backend_used ==
"gnuradio"``, anything else -> ``"numpy"`` plus a reason) and exercise the
matrix parsing/orientation with a stub flowgraph script, so the fallback
and the real path are both covered on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from prototype.io.gnuradio import viz

FAKE_SCRIPT = r'''
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--samp-rate", type=float, default=1000.0)
parser.add_argument("--fft-size", type=int, default=64)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

if args.check:
    print("GNU Radio 3.10.test")
    raise SystemExit(0)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

input_samples = np.fromfile(args.input_file, dtype=np.complex64)
frames = int(max(1, input_samples.size // max(args.fft_size, 1)))
data = np.linspace(1.0, 4.0, frames * args.fft_size, dtype=np.float32)
data.tofile(str(out_dir / "psd_frames.f32"))
'''


def _signal(n=4096):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
        np.complex64
    )


def test_script_and_interpreter_are_resolved_without_importing_gnuradio():
    script = viz.viz_script_path()
    # The integration package ships in this repository.
    assert script is None or script.is_file()
    assert isinstance(viz.gnuradio_python(), str)
    assert "gnuradio" not in sys.modules, (
        "the bridge must never import gnuradio into this process"
    )


def test_availability_returns_an_explainable_record():
    available, detail = viz.gnuradio_available()
    assert isinstance(available, bool)
    assert isinstance(detail, str) and detail


def test_unavailable_reports_the_numpy_fallback(monkeypatch):
    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (False, "not installed (test)")
    )

    result = viz.compute_spectrum_waterfall(_signal(), 8000.0)

    assert result["status"] == "unavailable"
    assert result["backend_used"] == "numpy"
    assert result["spectrum"] is None
    assert result["waterfall"] is None
    assert "not installed" in result["reason"]


def test_runtime_status_uses_the_executed_runtime(monkeypatch):
    """GNU Radio installed outside this interpreter must still count.

    ``gnuradio_available()`` only asks whether ``gnuradio`` imports into
    this process; the user-facing badge needs the runtime that can be
    executed, so the subprocess probe wins.
    """
    from prototype.io import gnuradio as gnuradio_pkg

    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (True, "GNU Radio 3.10.12.0")
    )

    assert gnuradio_pkg.gnuradio_runtime_status() == (True, "GNU Radio 3.10.12.0")


def test_runtime_status_falls_back_to_the_in_process_import(monkeypatch):
    from prototype.io import gnuradio as gnuradio_pkg

    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (False, "no runtime (test)")
    )

    available, detail = gnuradio_pkg.gnuradio_runtime_status()

    if gnuradio_pkg.gnuradio_available():
        # This interpreter itself can import gnuradio: it is usable.
        assert available is True
        assert detail.startswith("GNU Radio")
    else:
        assert available is False
        assert "no runtime (test)" in detail


def test_short_capture_is_refused_before_any_subprocess(monkeypatch):
    def _boom():  # pragma: no cover - must not be called
        raise AssertionError("availability should not be probed")

    monkeypatch.setattr(viz, "gnuradio_available", _boom)

    result = viz.compute_spectrum_waterfall(_signal(16), 8000.0, fft_size=64)
    assert result["status"] == "unavailable"
    assert "shorter than FFT size" in result["reason"]


def test_flowgraph_output_is_parsed_with_the_right_orientation(
    tmp_path, monkeypatch
):
    script = tmp_path / "fake_viz.py"
    script.write_text(FAKE_SCRIPT, encoding="utf-8")

    monkeypatch.setattr(viz, "viz_script_path", lambda: script)
    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (True, "GNU Radio 3.10.test")
    )

    fft_size = 64
    result = viz.compute_spectrum_waterfall(_signal(4096), 8000.0, fft_size=fft_size)

    assert result["status"] == "ok"
    assert result["backend_used"] == "gnuradio"
    assert result["source"] == "GNU Radio"

    spectrum = result["spectrum"]
    assert len(spectrum["freqs"]) == fft_size
    assert len(spectrum["magnitude"]) == fft_size
    assert all(np.isfinite(spectrum["magnitude"]))

    waterfall = result["waterfall"]
    power = np.asarray(waterfall["power_db"])
    freqs = np.asarray(waterfall["freqs"])
    times = np.asarray(waterfall["times"])
    # Rows are frequency bins, columns are time slices.
    assert power.shape == (freqs.size, times.size)
    assert power.shape[0] == fft_size
    assert result["provenance"]["input_samples"] > 0
    assert result["provenance"]["gnuradio"] == "GNU Radio 3.10.test"


def test_flowgraph_failure_is_reported_not_raised(tmp_path, monkeypatch):
    script = tmp_path / "broken_viz.py"
    script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")

    monkeypatch.setattr(viz, "viz_script_path", lambda: script)
    monkeypatch.setattr(
        viz, "gnuradio_available", lambda: (True, "GNU Radio 3.10.test")
    )

    result = viz.compute_spectrum_waterfall(_signal(), 8000.0)

    assert result["status"] == "unavailable"
    assert result["backend_used"] == "numpy"
    assert "flowgraph failed" in result["reason"]


def test_gnuradio_matrices_render_with_the_plot_helpers():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from prototype.visualization.plots import (
        create_spectrum_figure_from_psd,
        create_waterfall_figure_from_matrix,
    )

    freqs = np.linspace(-4000.0, 4000.0, 64)
    magnitude = np.abs(np.sin(np.linspace(0, 3, 64))) + 0.1
    spectrum_figure = create_spectrum_figure_from_psd(freqs, magnitude)
    assert spectrum_figure.axes

    times = np.linspace(0.0, 0.5, 8)
    power = np.random.default_rng(1).normal(size=(64, 8))
    waterfall_figure = create_waterfall_figure_from_matrix(freqs, times, power)
    assert waterfall_figure.axes


def test_mismatched_matrices_are_rejected():
    from prototype.visualization.plots import create_waterfall_figure_from_matrix

    freqs = np.linspace(0, 1, 8)
    times = np.linspace(0, 1, 4)
    with pytest.raises(ValueError):
        create_waterfall_figure_from_matrix(freqs, times, np.zeros((4, 8)))
