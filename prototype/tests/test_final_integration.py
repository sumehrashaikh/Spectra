"""Focused integration regressions for the SIH-147 final readiness pass.

These lock down the integration bugs discovered while auditing the GUI /
CLI flow, plus the demo-set coverage gaps:

* ``analyze`` built a ``FrameConfig`` from ``--sync-word`` and then never
  handed it to the pipeline (the sync word was silently ignored);
* ``analyze --ml/--ml-fusion/--labels`` were parsed but never reached the
  ``AnalysisConfig`` (the flags did nothing);
* the GNU Radio branch of ``analyze`` referenced ``analyze_samples`` /
  ``processing_mode_config`` without importing them, and passed a
  ``reference_bits_path`` argument ``analyze_samples`` does not accept;
* the pipeline ML stage called ``predict_modulation`` without importing
  it, so enabling ML only ever appended a warning;
* the deterministic demo set had no FSK case.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from tests import demo_captures as dc


# ---------------------------------------------------------------------------
# CLI analyse: explicit configuration must reach the pipeline
# ---------------------------------------------------------------------------


def test_analyze_forwards_protocol_and_ml_configuration(monkeypatch, tmp_path):
    from prototype import cli
    from prototype.pipeline import AnalysisResult

    recorded: dict = {}

    def _recording_analyze_capture(path, **kwargs):
        recorded.update(kwargs)
        return AnalysisResult()

    monkeypatch.setattr(
        "prototype.pipeline.analyze_capture", _recording_analyze_capture
    )

    capture = tmp_path / "x.wav"
    capture.write_bytes(b"")

    args = cli.build_parser().parse_args(
        [
            "analyze",
            str(capture),
            "--sync-word",
            "0xAA55AA55",
            "--data-bytes",
            "16",
            "--ml",
            "--ml-fusion",
            "max_confidence",
            "--labels",
            "labels.json",
        ]
    )
    assert cli.cmd_analyze(args) == 0

    protocol = recorded.get("protocol")
    assert protocol is not None, "the sync word never reached the pipeline"
    assert protocol.sync_word == 0xAA55AA55
    assert protocol.data_bytes == 16
    assert recorded.get("ml_enabled") is True
    assert recorded.get("ml_fusion") == "max_confidence"
    assert recorded.get("labels_path") == "labels.json"


def test_analyze_gnuradio_branch_runs(monkeypatch):
    """``--source gnuradio`` must not raise NameError / TypeError."""
    from prototype import cli
    from prototype.io import gnuradio as gr

    samples = dc.build_capture(modulation="QPSK", nbits=512).samples
    fake_signal = types.SimpleNamespace(
        samples=samples,
        sample_rate=dc.SAMPLE_RATE,
        metadata={"capture": {}},
    )

    monkeypatch.setattr(gr, "make_gnuradio_source", lambda *a, **k: object())
    monkeypatch.setattr(
        gr, "signal_from_gnuradio_source", lambda *a, **k: fake_signal
    )

    args = cli.build_parser().parse_args(["--verbose", "analyze", "-", "--source", "gnuradio"])
    args.verbose = False
    assert cli.cmd_analyze(args) == 0


# ---------------------------------------------------------------------------
# Pipeline ML stage actually runs
# ---------------------------------------------------------------------------


def test_pipeline_ml_stage_runs_when_enabled():
    from dataclasses import replace

    from prototype.core.config import processing_mode_config
    from prototype.pipeline import analyze_samples

    capture = dc.build_capture(modulation="16-QAM", nbits=1024)
    config = processing_mode_config("balanced")
    config = replace(config, ml=replace(config.ml, enabled=True))

    result = analyze_samples(capture.samples, dc.SAMPLE_RATE, config=config)

    assert result.fusion is not None, "the ML/fusion stage never produced a result"
    assert not any(
        "ML classification/fusion failed" in w for w in result.warnings
    ), result.warnings


# ---------------------------------------------------------------------------
# Demo set covers FSK + every de-interleaver family
# ---------------------------------------------------------------------------


def test_fsk_demo_capture_classifies_as_bfsk():
    from prototype.pipeline import analyze_samples

    capture = dc.build_fsk_capture()
    result = analyze_samples(
        capture.samples, dc.SAMPLE_RATE, reference_bits=capture.coded_bits
    )
    assert result.classification["modulation"] == "BFSK"
    assert result.demodulation["num_bits"] >= 900
    assert result.ber is not None and result.ber["ber"] < 0.05


def test_demo_set_covers_fsk_and_all_deinterleaver_families():
    captures = dc.build_demo_captures()
    names = {capture.name for capture in captures}
    assert any("BFSK" in name for name in names)

    families = {capture.interleave_family for capture in captures}
    assert {"block", "convolutional", "diagonal", "pseudo_random"} <= families

    modulations = {capture.modulation for capture in captures}
    assert {"16-QAM", "QPSK", "8-PSK", "BFSK"} <= modulations


def test_fsk_demo_round_trips_through_wav(tmp_path):
    from prototype.core.loader import load_wav
    from prototype.pipeline import analyze_samples

    capture = dc.build_fsk_capture()
    path = tmp_path / "bfsk.wav"
    dc.write_wav(str(path), capture.samples)

    samples, sample_rate = load_wav(str(path))
    result = analyze_samples(samples, sample_rate)
    assert result.classification["modulation"] == "BFSK"
    assert np.asarray(result.demodulation["received_bits"]).size > 0


# ---------------------------------------------------------------------------
# Demo reference sidecar: enables the clean GUI/CLI demonstration
# ---------------------------------------------------------------------------


def test_demo_reference_sidecar_enables_clean_decode(tmp_path):
    from dataclasses import replace

    from prototype.core.ber import load_transmitted_bits
    from prototype.core.config import FECMode, processing_mode_config
    from prototype.core.loader import load_wav
    from prototype.pipeline import analyze_samples

    capture = dc.build_capture(
        modulation="16-QAM",
        fec_scheme="reedsolomon",
        interleave_family="block",
        interleave_param=8,
    )
    wav = tmp_path / "demo.wav"
    dc.write_capture(str(wav), capture)

    bits, reference_path = load_transmitted_bits(str(wav))
    assert bits is not None and reference_path.is_file()
    assert np.array_equal(bits, capture.coded_bits)

    samples, sample_rate = load_wav(str(wav))
    config = processing_mode_config("balanced")
    fec = replace(
        config.fec,
        mode=FECMode.MANUAL,
        scheme="reedsolomon",
        interleaving_mode=FECMode.MANUAL,
        interleave_family="block",
        interleave_depth=8,
    )
    result = analyze_samples(
        samples, sample_rate, config=replace(config, fec=fec), reference_bits=bits
    )
    decoded = (result.demodulation or {}).get("fec") or {}
    assert decoded.get("uncorrectable_blocks") == 0
    assert result.ber is not None and result.ber["ber"] < 0.01


def test_demo_reference_skipped_for_unresolvable_cases(tmp_path):
    """No sidecar for 8-PSK demo captures: they classify as 16-QAM."""
    from prototype.core.ber import reference_path_for_wav

    capture = dc.build_capture(modulation="8-PSK", nbits=1020)
    assert dc.reference_is_usable(capture) is False

    wav = tmp_path / "psk8.wav"
    dc.write_capture(str(wav), capture)
    assert not Path(reference_path_for_wav(str(wav))).is_file()


def test_uncoded_qpsk_demo_measures_a_zero_ber(tmp_path):
    """Uncoded QPSK carries a reference now, and the BER is exact.

    The receiver's QPSK folds (90-degree lattice fold, axis-aligned twin,
    symbol origin) resolve against the transmitted bits, so an uncoded
    QPSK capture reports a measured BER like any coded one.
    """
    from prototype.core.ber import load_transmitted_bits
    from prototype.core.loader import load_wav
    from prototype.pipeline import analyze_samples

    capture = dc.build_capture(modulation="QPSK", nbits=1024)
    assert dc.reference_is_usable(capture) is True

    wav = tmp_path / "qpsk.wav"
    dc.write_capture(str(wav), capture)
    bits, reference_path = load_transmitted_bits(str(wav))
    assert bits is not None and reference_path.is_file()

    samples, sample_rate = load_wav(str(wav))
    result = analyze_samples(samples, sample_rate, reference_bits=bits)

    assert result.ber is not None and result.ber["status"] == "measured"
    assert result.ber["ber"] == 0.0, result.ber
    interleaving = (result.demodulation or {}).get("interleaving_result") or {}
    assert interleaving.get("status") == "NONE", interleaving
