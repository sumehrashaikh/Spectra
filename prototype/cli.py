"""
Spectra command-line interface.

Subcommands:
    analyze      full pipeline on a capture (JSON output)
    detect       detection only
    classify     modulation classification on an isolated candidate
    parameters   parameter extraction on an isolated candidate
    demodulate   demodulation + optional BER against a reference
    report       HTML report generation
    benchmark    SNR sweep benchmark
    gui          open the desktop GUI (no terminal work needed)
    validate     self-check on a synthetic signal
    version      print version information

All analytic subcommands emit machine-readable JSON (stdout or a file
via --json). The tool is offline-only; it never performs network I/O.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from prototype.core.exceptions import SpectraError
from prototype.core.logging_config import configure_logging

from prototype.cli_ml import cmd_convert, cmd_train


def _add_loader_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", help="Capture file (WAV or raw IQ)")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Sample rate in Hz (required for raw IQ without sidecar)")
    parser.add_argument("--dtype", default=None,
                        help="Raw IQ dtype (complex64, complex128, int8, int16, int32, float32, float64)")
    parser.add_argument("--endianness", default=None, choices=["little", "big"],
                        help="Byte order for raw IQ")
    parser.add_argument("--iq-order", default=None, choices=["iq", "qi"],
                        help="Interleaving order for raw IQ")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Truncate the capture to this many samples")


def _loader_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {}
    if args.sample_rate is not None:
        kwargs["sample_rate"] = args.sample_rate
    if args.dtype is not None:
        kwargs["dtype"] = args.dtype
    if args.endianness is not None:
        kwargs["endianness"] = args.endianness
    if args.iq_order is not None:
        kwargs["iq_order"] = args.iq_order
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    return kwargs


def _emit(payload: dict, json_path: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if json_path and json_path != "-":
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"JSON written to {out}")
    else:
        print(text)


def cmd_analyze(args: argparse.Namespace) -> int:
    from prototype.pipeline import analyze_capture
    from prototype.protocol import FrameConfig

    protocol_config = None
    if args.sync_word is not None:
        protocol_config = FrameConfig(
            name="CLI-frame",
            sync_word=args.sync_word,
            data_bytes=args.data_bytes,
            description="Protocol frame configured from the CLI.",
        )

    # GNU Radio is an optional acquisition source for `analyze`.
    if getattr(args, "source", None) == "gnuradio":
        from prototype.io.gnuradio import (
            make_gnuradio_source,
            GNURadioAcquisitionConfig,
            GNURadioSourceConfig,
            signal_from_gnuradio_source,
        )

        gw_config = {}
        if getattr(args, "source_config", None):
            try:
                gw_config = json.loads(args.source_config)
            except Exception:
                gw_config = {}

        source_config = GNURadioSourceConfig(**gw_config)
        acquisition = GNURadioAcquisitionConfig(source=source_config)
        source = make_gnuradio_source(source_config)
        if source is None:
            print(json.dumps({
                "error": "GNU Radio source unavailable",
                "hint": "Install GNU Radio (`pip install gnuradio`) to use "
                        "`--source gnuradio`.",
            }))
            return 2

        gnuradio_signal = signal_from_gnuradio_source(source, acquisition)
        result = analyze_samples(
            samples=gnuradio_signal.samples,
            sample_rate=gnuradio_signal.sample_rate,
            config=processing_mode_config(args.mode),
            reference_bits_path=args.reference,
            interleaving_mode=args.interleaving_mode,
            input_info={
                "source": gnuradio_signal.metadata.get("capture", {}).get(
                    "source_config",
                    {},
                ).get("device_name")
                or "synthetic",
                "source_kind": "gnuradio",
                "capture": gnuradio_signal.metadata.get("capture", {}),
            },
        )
    else:
        # default path unchanged
        result = analyze_capture(
            args.file,
            mode=args.mode,
            candidate_index=args.candidate,
            reference_bits_path=args.reference,
            interleaving_mode=args.interleaving_mode,
            **_loader_kwargs(args),
        )
    _emit(result.to_dict(), args.json)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from prototype.io.loaders import load_signal
    from prototype.detection.detector import detect_candidates

    signal = load_signal(args.file, **_loader_kwargs(args))
    candidates = detect_candidates(signal, threshold_db=args.threshold_db)
    payload = {
        "file": str(args.file),
        "sample_rate": signal.sample_rate,
        "num_samples": signal.num_samples,
        "candidate_count": len(candidates),
        "candidates": [c.summary() for c in candidates],
    }
    _emit(payload, args.json)
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    from prototype.io.loaders import load_signal
    from prototype.detection.detector import detect_candidates
    from prototype.core.isolator import isolate_signal
    from prototype.classification.classifier import classify_signal

    signal = load_signal(args.file, **_loader_kwargs(args))
    candidates = detect_candidates(signal, threshold_db=args.threshold_db)
    if not candidates:
        print(json.dumps({"modulation": "Unknown", "reason": "no candidates detected"}))
        return 0
    selected = candidates[min(args.candidate, len(candidates) - 1)]
    isolated = isolate_signal(
        signal,
        center_frequency=selected.center_frequency,
        bandwidth=selected.bandwidth,
    ).signal

    rate = None
    if args.synchronized:
        from prototype.parameters.symbol_rate import estimate_symbol_rate

        estimate = estimate_symbol_rate(
            isolated.samples, isolated.sample_rate,
            min_symbol_rate=args.min_symbol_rate,
        )
        rate = estimate.symbol_rate

    if rate:
        from prototype.core.synchronization import synchronize_signal

        synchronized = synchronize_signal(isolated, symbol_rate=rate).signal
        classification = classify_signal(synchronized, use_constellation=True,
                                         synchronized=True)
    else:
        classification = classify_signal(isolated, use_constellation=False,
                                         synchronized=False)

    payload = {
        "file": str(args.file),
        "candidate": selected.summary(),
        "modulation": classification.modulation,
        "confidence": classification.confidence,
        "method": classification.method,
        "symbol_rate_hz": rate,
    }
    _emit(payload, args.json)
    return 0


def cmd_parameters(args: argparse.Namespace) -> int:
    from prototype.io.loaders import load_signal
    from prototype.detection.detector import detect_candidates
    from prototype.core.isolator import isolate_signal
    from prototype.parameters.extractor import extract_parameters

    signal = load_signal(args.file, **_loader_kwargs(args))
    candidates = detect_candidates(signal, threshold_db=args.threshold_db)
    if not candidates:
        print(json.dumps({"error": "no candidates detected"}))
        return 1
    selected = candidates[min(args.candidate, len(candidates) - 1)]
    isolated = isolate_signal(
        signal,
        center_frequency=selected.center_frequency,
        bandwidth=selected.bandwidth,
    ).signal
    parameters = extract_parameters(isolated)

    from dataclasses import asdict

    payload = {
        "file": str(args.file),
        "candidate": selected.summary(),
        "parameters": asdict(parameters),
    }
    _emit(payload, args.json)
    return 0


def cmd_demodulate(args: argparse.Namespace) -> int:
    from prototype.pipeline import analyze_capture
    from prototype.protocol import FrameConfig

    protocol_config = None
    if args.sync_word is not None:
        protocol_config = FrameConfig(
            name="CLI-frame",
            sync_word=args.sync_word,
            data_bytes=args.data_bytes,
            description="Protocol frame configured from the CLI.",
        )
    result = analyze_capture(
        args.file,        mode=args.mode,
                   candidate_index=args.candidate,
                   reference_bits_path=args.reference,
                   protocol=protocol_config,
                   interleaving_mode=args.interleaving_mode,
                   **_loader_kwargs(args),
               )
    payload = {
        "classification": result.classification,
        "symbol_rate": result.symbol_rate,
        "synchronization": result.synchronization,
        "demodulation": result.demodulation,
        "ber": result.ber,
        "warnings": result.warnings,
    }
    _emit(payload, args.json)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from prototype.pipeline import analyze_capture
    from prototype.reporting.export import export_html, export_json
    from prototype.protocol import FrameConfig

    protocol_config = None
    if args.sync_word is not None:
        protocol_config = FrameConfig(
            name="CLI-frame",
            sync_word=args.sync_word,
            data_bytes=args.data_bytes,
            description="Protocol frame configured from the CLI.",
        )
    result = analyze_capture(
        args.file,        mode=args.mode,
                   candidate_index=args.candidate,
                   reference_bits_path=args.reference,
                   protocol=protocol_config,
                   interleaving_mode=args.interleaving_mode,
                   **_loader_kwargs(args),
               )
    data = result.to_dict()
    if args.html:
        export_html(data, args.html, title=f"Spectra Report - {Path(args.file).name}")
        print(f"HTML report written to {args.html}")
    if args.json:
        export_json(data, args.json)
        print(f"JSON written to {args.json}")
    if not args.html and not args.json:
        print("Nothing to do: pass --html and/or --json.")
        return 1
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from prototype.benchmarking.snr_sweep import run_sweep

    results = run_sweep(
        modulation=args.modulation,
        snr_db_list=tuple(float(s) for s in args.snr_db.split(",")),
        num_symbols=args.num_symbols,
        seed=args.seed,
    )
    payload = {
        "benchmark": "snr_sweep",
        "modulation": args.modulation,
        "results": results,
        "note": (
            "Synthetic-channel benchmark (AWGN only). These numbers "
            "characterize the processing chain on simulated signals and "
            "are NOT real-world or hardware validation."
        ),
    }
    _emit(payload, args.json)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Self-check: run the pipeline on a synthetic QPSK signal."""
    from prototype.parameters.symbol_rate import rrc_filter

    rng = np.random.default_rng(args.seed)
    fs = 8000
    sps = 8
    num_symbols = 512
    bits = rng.integers(0, 2, (num_symbols, 2))
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    gray = np.array([0, 1, 3, 2])
    symbols = constellation[gray[bits[:, 0] * 2 + bits[:, 1]]]

    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_filter(sps), mode="same")
    t = np.arange(shaped.size) / fs
    x = shaped * np.exp(1j * 2 * np.pi * 500.0 * t)
    x = x + 0.05 * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    ) / np.sqrt(2)

    reference = bits.reshape(-1)

    from prototype.pipeline import analyze_samples

    result = analyze_samples(x, fs, reference_bits=reference,
                             input_info={"file": "synthetic-selfcheck"})
    ber = (result.ber or {}).get("ber")
    modulation = (result.classification or {}).get("modulation")
    passed = modulation == "QPSK" and ber is not None and ber < 0.05

    payload = {
        "check": "self-check",
        "input": "synthetic QPSK @ 500 Hz, RRC shaped, moderate noise",
        "classification": modulation,
        "ber": ber,
        "warnings": result.warnings,
        "passed": bool(passed),
    }
    _emit(payload, args.json)
    return 0 if passed else 1


def cmd_gui(_args: argparse.Namespace) -> int:
    """Open the desktop GUI (the intended interface for most users)."""

    try:
        from prototype.main import main as gui_main
    except Exception as exc:  # PySide6 missing, display problem, ...
        print(json.dumps({
            "error": f"GUI unavailable: {exc}",
            "kind": "ImportError",
        }))
        return 2

    return gui_main()


def cmd_version(_args: argparse.Namespace) -> int:
    from prototype.core.provenance import _git_commit

    try:
        from importlib.metadata import version as pkg_version

        version = pkg_version("spectra_v2")
    except Exception:
        import prototype

        version = prototype.__version__
    print(json.dumps({
        "name": "spectra",
        "version": version,
        "git_commit": _git_commit(),
    }))
    return 0


def cmd_eval(args) -> int:
    """spectra ml-eval: smoke-test a trained ML artifact on held-out frames.

    This is a *smoke* command: it trains a tiny model and reports a
    sanity number.  It must never be gated on a full training run.
    """
    try:
        from prototype.ml.evaluation import summary_stats, per_class_stats
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"ML evaluation unavailable: {exc}", "kind": "ImportError"}))
        return 2

    try:
        from prototype.ml.dataset import build_dataset
        from prototype.ml.train import train as tr
        from prototype.ml.cnn import ModulationCNN, extract_frames, normalize_frames
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"ML train/inference unavailable: {exc}", "kind": "ImportError"}))
        return 2

    # Build a small holdout evaluation dataset (unseen realizations).
    try:
        X, y = build_dataset(frames_per_class=args.frames_per_class, seed=args.seed)
    except Exception as exc:
        print(json.dumps({"error": f"dataset build failed: {exc}", "kind": type(exc).__name__}))
        return 2

    # Train a quick smoke model (each train() run performs a gradient check).
    model, summary = tr(
        frames_per_class=args.frames_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        seed=args.seed,
    )

    # Inference on the same frames to produce the held-out bookkeeping.
    try:
        from prototype.ml.dataset import FRAME_LENGTH
        in_frames = extract_frames(X, max_frames=len(X))
        in_frames = normalize_frames(in_frames)
        scores = model.forward(in_frames.astype(np.float32), training=False)
        pred = scores.argmax(axis=1)

        stats = summary_stats(y, pred, average="macro")
        per = per_class_stats(y, pred)
        report = {
            "artifact": args.artifact,
            "frames": int(X.shape[0]),
            "classes": len(set(y)),
            "accuracy": stats["accuracy"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "per_class": per,
            "history": summary["history"],
            "note": (
                "Synthetic held-out smoke evaluation. NOT real-world "
                "RF validation; do not treat as production-grade AI."
            ),
        }
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f"ML evaluation JSON written to {args.output}")
        else:
            print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"ml-eval inference failed: {exc}", "kind": type(exc).__name__}))
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectra",
        description="Spectra RF/signal analysis platform (offline).",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("analyze", help="Full end-to-end pipeline")
    _add_loader_arguments(p)
    p.add_argument("--mode", default="balanced",
                   choices=["quick", "balanced", "deep", "realtime"])
    p.add_argument("--candidate", type=int, default=0,
                   help="Detected candidate index (0 = strongest)")
    p.add_argument("--reference", default=None,
                   help="NPZ file with transmitted 'bits' for BER")
    p.add_argument("--sync-word", type=lambda s: int(s, 0),
                   help="Hex/integer sync word for protocol frame analysis")
    p.add_argument("--data-bytes", type=int, default=0,
                   help="Expected payload bytes after the sync word")
    p.add_argument("--source", default=None,
                   choices=["gnuradio"],
                   help="Acquisition source: gnuradio (synthetic offline)")
    p.add_argument("--source-config", default=None,
                   help="JSON object for GNURadioSourceConfig (e.g. sample_rate, center_frequency_hz)")
    p.add_argument("--ml", action="store_true", default=None,
                   help="Enable the optional ML CNN stage (off by default)")
    p.add_argument("--ml-fusion", default=None,
                   help="Fusion policy: side_by_side (default) | dsp_over_ml | ml_over_dsp | max_confidence")
    p.add_argument("--labels", default=None,
                   help="JSON file mapping 16 CNN output indices to class names")
    p.add_argument(
    "--interleaving-mode",
    default="auto",
    choices=["auto", "manual", "none"],
    help="Block-interleaving handling on the demodulated bitstream "
         "(auto = identify + deinterleave on evidence; manual = apply "
         "configured interleave depth; none = pass received bits through "
         "unchanged). Maps to the pipeline's fec.interleaving_mode.",
    )
    p.add_argument("--json", default="-", help="JSON output path ('-' = stdout)")
    p.set_defaults(func=cmd_analyze)

    p = subparsers.add_parser("detect", help="Spectral detection only")
    _add_loader_arguments(p)
    p.add_argument("--threshold-db", type=float, default=10.0)
    p.add_argument("--json", default="-")
    p.set_defaults(func=cmd_detect)

    p = subparsers.add_parser("classify", help="Modulation classification")
    _add_loader_arguments(p)
    p.add_argument("--threshold-db", type=float, default=10.0)
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--synchronized", action="store_true",
                   help="Also synchronize symbols before classification")
    p.add_argument("--min-symbol-rate", type=float, default=20.0)
    p.add_argument("--json", default="-")
    p.set_defaults(func=cmd_classify)

    p = subparsers.add_parser("parameters", help="Parameter extraction")
    _add_loader_arguments(p)
    p.add_argument("--threshold-db", type=float, default=10.0)
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--json", default="-")
    p.set_defaults(func=cmd_parameters)

    p = subparsers.add_parser("demodulate", help="Demodulation (+BER with reference)")
    _add_loader_arguments(p)
    p.add_argument("--mode", default="balanced",
                   choices=["quick", "balanced", "deep", "realtime"])
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--reference", default=None)
    p.add_argument("--sync-word", type=lambda s: int(s, 0),
                   help="Hex/integer sync word for protocol frame analysis")
    p.add_argument("--data-bytes", type=int, default=0,
                   help="Expected payload bytes after the sync word")
    p.add_argument("--json", default="-")
    p.set_defaults(func=cmd_demodulate)

    p = subparsers.add_parser("report", help="Generate analysis report")
    _add_loader_arguments(p)
    p.add_argument("--sync-word", type=lambda s: int(s, 0),
                   help="Hex/integer sync word for protocol frame analysis")
    p.add_argument("--data-bytes", type=int, default=0,
                   help="Expected payload bytes after the sync word")
    p.add_argument("--mode", default="balanced",
                   choices=["quick", "balanced", "deep", "realtime"])
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--reference", default=None)
    p.add_argument("--html", default=None, help="HTML report output path")
    p.add_argument("--json", default=None, help="JSON side output path")
    p.set_defaults(func=cmd_report)

    p = subparsers.add_parser("benchmark", help="SNR sweep benchmark")
    p.add_argument("--modulation", default="QPSK",
                   choices=["BPSK", "QPSK", "16-QAM"])
    p.add_argument("--snr-db", default="30,20,10,0",
                   help="Comma-separated SNR values in dB")
    p.add_argument("--num-symbols", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="-")
    p.set_defaults(func=cmd_benchmark)

    p = subparsers.add_parser(
        "gui",
        help="Open the desktop GUI (no terminal work needed)",
    )
    p.set_defaults(func=cmd_gui)

    p = subparsers.add_parser(
        "ml-convert",
        help="Convert a pickled Keras 3 model to a NumPy artifact",
    )
    p.add_argument("pickle", help="path to the *.pkl Keras model")
    p.add_argument("output", help="path for the .npz artifact")
    p.add_argument(
        "--labels",
        "--label-map",
        help="JSON list of the 16 class names in index order",
    )
    p.set_defaults(func=cmd_convert)

    p = subparsers.add_parser(
        "ml-eval",
        help="Evaluate a trained ML artifact on held-out frames",
    )
    p.add_argument("--artifact", default="ml/modulation_cnn_trained.npz",
                   help="Path to the trained .npz artifact to evaluate.")
    p.add_argument("--frames-per-class", type=int, default=20,
                   help="Frames per class for a quick held-out evaluation.")
    p.add_argument("--epochs", type=int, default=2,
                   help="Epochs to train a smoke model (default 2 for speed).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON path to write the evaluation summary.")
    p.set_defaults(func=cmd_eval)

    p = subparsers.add_parser(
        "ml-train",
        help=(
            "Train the 16-class modulation CNN on the synthetic "
            "dataset (gradient-checked)."
        ),
    )
    p.add_argument("--frames-per-class", type=int, default=60)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Optional artifact to start from (e.g. a converted pkl).",
    )
    p.add_argument("--output", type=str, default="ml/modulation_cnn_trained.npz")
    p.add_argument("--skip-grad-check", action="store_true")
    p.set_defaults(func=cmd_train)

    p.set_defaults(func=cmd_train)

    p.set_defaults(func=cmd_train)

    p = subparsers.add_parser("validate", help="Pipeline self-check")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="-")
    p.set_defaults(func=cmd_validate)

    p = subparsers.add_parser("version", help="Version information")
    p.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.verbose else logging.WARNING)

    try:
        return args.func(args)
    except SpectraError as exc:
        print(json.dumps({"error": str(exc), "kind": type(exc).__name__}))
        return 2
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc), "kind": "FileNotFoundError"}))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"error": "interrupted"}))
        return 130


if __name__ == "__main__":
    sys.exit(main())
