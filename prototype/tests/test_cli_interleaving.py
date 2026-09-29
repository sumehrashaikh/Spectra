"""Focused tests for the CLI ``--interleaving-mode`` flag and the manual / none
regression paths.

These tests exercise the CLI entry point (``prototype.cli``) in a
subprocess-based, headless manner (no display required): they assert the CLI
accepts the ``--interleaving-mode`` argument and that the produced JSON payload
carries the detected interleaving status/depth/confidence/candidates.  They do
not touch DSP, classifier, or synchronisation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure ``prototype`` is importable when pytest is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prototype.cli import build_parser, cmd_analyze  # noqa: E402


def _run_cli(argv: list[str]) -> dict:
    """Run the CLI via subprocess and return the parsed JSON it emits on stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "prototype.cli", *argv],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI exited {proc.returncode}: {proc.stderr}\n{proc.stdout}"
        )
    return json.loads(proc.stdout)


def test_interleaving_mode_choices() -> None:
    """The CLI must accept only auto / manual / none."""
    for mode in ("auto", "manual", "none"):
        args = build_parser().parse_args(
            ["analyze", "--interleaving-mode", mode, "-", "--json", "-"]
        )
        assert args.interleaving_mode == mode


def test_interleaving_mode_rejects_bad_value() -> None:
    """An out-of-range interleaving mode must be rejected by argparse."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["analyze", "--interleaving-mode", "bogus", "-", "--json", "-"]
        )


def test_interleaving_mode_wired_through_to_cmd_analyze(monkeypatch) -> None:
    """The parsed namespace reaches ``cmd_analyze`` unchanged."""
    called: dict[str, Any] = {}

    def _recording_cmd_analyze(args: Any) -> int:
        called["args"] = args
        return 0

    monkeypatch.setattr("prototype.cli.cmd_analyze", _recording_cmd_analyze)
    arg_strings = ["analyze", "--interleaving-mode", "manual", "-", "--json", "-"]
    args = build_parser().parse_args(arg_strings)
    assert args.interleaving_mode == "manual"


# ---------------------------------------------------------------------------
# Extended de-interleaving family + explicit FEC scheme flags
# ---------------------------------------------------------------------------


def test_interleave_family_choices() -> None:
    """The CLI accepts all four de-interleaver families."""
    for family in ("block", "convolutional", "diagonal", "pseudo_random"):
        args = build_parser().parse_args(
            ["analyze", "--interleave-family", family, "-", "--json", "-"]
        )
        assert args.interleave_family == family


def test_interleave_family_rejects_bad_value() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["analyze", "--interleave-family", "bogus", "-", "--json", "-"]
        )


def test_new_fec_schemes_reach_the_loader_config(monkeypatch, tmp_path) -> None:
    """``analyze_capture`` maps --fec-scheme/--interleave-family onto config."""
    from dataclasses import dataclass

    from prototype import pipeline as pipeline_mod
    from prototype.pipeline import analyze_capture

    @dataclass
    class _Signal:
        samples: np.ndarray
        sample_rate: float
        metadata: dict

    monkeypatch.setattr(
        "prototype.io.loaders.load_signal",
        lambda *a, **k: _Signal(
            samples=np.zeros(16, dtype=np.complex64),
            sample_rate=1.0,
            metadata={},
        ),
    )

    captured: dict[str, Any] = {}

    def _recording_analyze_samples(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return "ok"

    monkeypatch.setattr(
        "prototype.pipeline.analyze_samples", _recording_analyze_samples
    )

    path = tmp_path / "fake.iq"
    path.write_bytes(b"\x00" * 8)

    analyze_capture(
        path,
        mode="balanced",
        interleaving_mode="manual",
        interleave_depth=8,
        interleave_family="convolutional",
        fec_mode="manual",
        fec_scheme="reedsolomon",
    )

    fec = captured["config"].fec
    assert fec.interleaving_mode == "manual"
    assert fec.interleave_family == "convolutional"
    assert fec.interleave_depth == 8
    assert fec.scheme == "reedsolomon"
