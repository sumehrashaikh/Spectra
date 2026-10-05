"""CLI commands for the optional external real-world validation dataset.

These two subcommands are strictly additive: they are the only entry points
into ``prototype.external_validation``, and the rest of SPECTRA neither
imports nor depends on them or on ``h5py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from prototype.external_validation.hdf5_dataset import load_dataset


def _emit(payload: dict, json_path: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if json_path and json_path != "-":
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"JSON written to {out}")
    else:
        print(text)


def cmd_dataset_inspect(args) -> int:
    """Report shape/dtype/labels/balance of the external dataset."""

    dataset = load_dataset(getattr(args, "dataset_dir", None))
    if not dataset.available:
        print(json.dumps({
            "available": False,
            "reason": dataset.unavailable_reason,
            "hint": "Place subset_*.h5 under dataset/ (or pass --dataset-dir) "
                    "and install h5py. See dataset/README.md.",
        }))
        return 2

    subsets = args.subset or None
    payload = dataset.inspect(subsets)
    payload["command"] = "dataset-inspect"
    _emit(payload, getattr(args, "json", "-"))
    return 0


def cmd_dataset_eval(args) -> int:
    """Run the existing chain on a sampled subset of the external dataset."""

    from prototype.external_validation.evaluate import (
        evaluate_dataset,
        render_summary,
    )

    subsets = args.subset or ["test"]
    try:
        payload = evaluate_dataset(
            dataset_dir=getattr(args, "dataset_dir", None),
            subsets=tuple(subsets),
            per_cell=args.per_cell,
            seed=args.seed,
            ml_enabled=args.ml,
            max_frames=args.max_frames,
            dump_frames=args.dump_frames,
            max_workers=args.workers,
        )
    except RuntimeError as exc:
        print(json.dumps({"available": False, "reason": str(exc)}))
        return 2

    payload["command"] = "dataset-eval"

    if not payload.get("available"):
        print(json.dumps(payload, indent=2))
        return 2

    if args.table:
        for line in render_summary(payload):
            print(line, file=sys.stderr)

    _emit(payload, args.json)
    return 0
