"""Optional external real-world validation for SPECTRA.

This package is **additive only**. It loads the downloaded Mendeley
real-world IQ dataset (see ``dataset/README.md``) and runs SPECTRA's
*existing* preprocessing, deterministic DSP classifier and optional ML
assist over a sampled subset, so the project can report how its current
chain behaves on genuinely off-air data.

Nothing here changes pipeline behaviour, default configuration, the CNN
runtime, the DSP classifier or their artifacts. Importing this package
has no side effects, and ``h5py`` is imported lazily so SPECTRA keeps
working without it.

Honesty rules enforced by this package:

* The dataset's own published baseline accuracy is never re-reported as a
  SPECTRA result.
* The dataset spans 20-30 dB SNR only, so no low-SNR claim may be drawn
  from it in either direction.
* Classes outside SPECTRA's deterministic vocabulary are reported as
  *unsupported*, never silently scored as wrong.
"""

from __future__ import annotations

__all__ = [
    "RealWorldIqDataset",
    "load_dataset",
    "evaluate_dataset",
    "h5py_available",
    "resolve_dataset_dir",
]


def __getattr__(name: str):
    """Resolve the public API lazily (keeps ``h5py`` optional)."""

    if name in ("RealWorldIqDataset", "load_dataset", "h5py_available",
                "resolve_dataset_dir"):
        from prototype.external_validation import hdf5_dataset

        return getattr(hdf5_dataset, name)
    if name == "evaluate_dataset":
        from prototype.external_validation import evaluate

        return evaluate.evaluate_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
