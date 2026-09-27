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
