"""Thin CLI frontdoors for the ML subsystem.

    spectra ml-convert model.pkl artifact.npz
    spectra ml-train --frames-per-class 60 --epochs 15

They exist so that the person who actually trained/audio-delivered the
model does not need to learn the ``python -m prototype.ml`` path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spectra machine-learning tools (NumPy runtime)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch progress lines.",
    )
    return parser


def cmd_convert(args) -> int:
    """spectra ml-convert model.pkl artifact.npz [--labels LABELS_JSON]

    Converts the pickled Keras 3 model into the NumPy artifact format
    used by the Spectra ML runtime, without unpickling it. Add
    ``--labels labels.json`` (index-ordered 16 names) so the model's
    16 softmax outputs get real human-readable class names.
    """
    from pathlib import Path

    pickle_path = Path(args.pickle)
    output_path = Path(args.output)
    labels_path = Path(args.labels) if args.labels else None

    from prototype.ml.tools.convert_keras_pickle import convert

    convert(pickle_path, output_path, labels_path)

    print(f"Converted {args.pickle} -> {args.output}")
    return 0


def cmd_train(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="spectra ml-train",
        description=(
            "Train the 16-class modulation CNN on synthetic frames. "
            "Requires prototype/tests to exist (part of the repo)."
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
    p.add_argument("--skip-grad-check", action="store_true",
                   help="Bypass the finite-difference check (not recommended).")
    args = p.parse_args(argv)

    from prototype.ml.train import main as tr

    # argparse Namespace -> list: reuse the existing train() CLI surface.
    argv_list = [
        "train",
        "--frames-per-class", str(args.frames_per_class),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--output", args.output,
    ]
    if args.init_from:
        argv_list += ["--init-from", args.init_from]
    if args.skip_grad_check:
        argv_list += ["--skip-grad-check"]
    if args.quiet:
        # Suppress train() prints to keep this command chatty-free.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return tr(argv_list)
    return tr(argv_list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spectra machine-learning commands."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ml-convert",
        help="Convert a pickled Keras 3 model into a NumPy artifact",
    )
    sub.add_parser(
        "ml-train",
        help=(
            "Train the 16-class modulation CNN (synthetic dataset, "
            "gradient-checked)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ml-convert":
        return cmd_convert(vars(args))
    return cmd_train(vars(args))


if __name__ == "__main__":
    sys.exit(main())
