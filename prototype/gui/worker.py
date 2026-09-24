"""
Background worker for GUI-driven analysis.

Runs the V2 pipeline (or the multi-candidate batch) on a QThread so the
UI stays responsive during heavy DSP. Results are delivered through Qt
signals; failures are delivered as structured errors, never raised into
the Qt event loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal as QtSignal

from prototype.core.config import processing_mode_config
from prototype.core.logging_config import logger


def _attach_batch_timeline(payload: dict, sample_rate: float) -> None:
    """Attach the frequency-ordered candidate timeline to a batch payload.

    The batch analyzer exposes ``candidate_timeline`` (frequency bounds,
    classification, confidence, rank per candidate) but nothing in the
    GUI consumed it; attaching it here keeps the exported JSON and the
    candidate table equally informed.
    """

    from prototype.pipeline_batch import candidate_timeline

    try:
        payload["candidate_timeline"] = candidate_timeline(
            _ReconstructedBatch(payload),
            sample_rate,
        )
    except Exception:  # timeline is supplementary; never fail the run
        logger.debug("candidate timeline unavailable", exc_info=True)


class _ReconstructedBatch:
    """Minimal duck-typed BatchResult for candidate_timeline."""

    def __init__(self, payload: dict) -> None:
        self.detections = payload.get("detections") or []
        self.results = [
            (item.get("candidate_index", index), _Det(item))
            for index, item in enumerate(payload.get("analyzed_candidates") or [])
        ]


class _Det:
    def __init__(self, item: dict) -> None:
        self.classification = item.get("classification") or {}


class AnalysisWorker(QThread):
    """
    Run ``pipeline.analyze_samples`` (or the batch analyzer) off the UI
    thread.

    Signals
    -------
    finished_with_result
        Emitted with a JSON-safe ``AnalysisResult.to_dict()`` (single
        candidate mode) or a ``BatchResult.to_dict()`` (batch mode).
    failed
        Emitted with a human-readable error message.
    """

    finished_with_result = QtSignal(object)
    failed = QtSignal(str)

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: float,
        mode: str = "balanced",
        analyze_all: bool = False,
        reference_bits: np.ndarray | None = None,
        fec_scheme: str | None = None,
        ml_enabled: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self._samples = np.asarray(samples)
        self._sample_rate = float(sample_rate)
        self._mode = str(mode)
        self._analyze_all = bool(analyze_all)
        self._reference_bits = reference_bits
        self._fec_scheme = fec_scheme
        self._ml_enabled = bool(ml_enabled)

    def run(self) -> None:  # noqa: D102 - Qt override
        try:
            # FEC is explicit configuration, never guessed; an unset
            # selector ("none") means the pipeline default (no FEC).
            fec_scheme: str | None = (
                None
                if self._fec_scheme in (None, "", "none")
                else str(self._fec_scheme)
            )

            from dataclasses import replace

            config = processing_mode_config(self._mode)

            if fec_scheme is not None:
                config = replace(
                    config,
                    fec=replace(config.fec, scheme=fec_scheme),
                )

            if self._ml_enabled:
                config = replace(
                    config,
                    ml=replace(config.ml, enabled=True),
                )

            if self._analyze_all:
                from prototype.pipeline_batch import analyze_all_candidates

                batch = analyze_all_candidates(
                    samples=self._samples,
                    sample_rate=self._sample_rate,
                    reference_bits=self._reference_bits,
                    config=config,
                )
                payload: Any = batch.to_dict()

                _attach_batch_timeline(payload, self._sample_rate)
            else:
                from prototype.pipeline import analyze_samples

                result = analyze_samples(
                    samples=self._samples,
                    sample_rate=self._sample_rate,
                    config=config,
                    reference_bits=self._reference_bits,
                    capture_symbol_samples=True,
                )
                payload = result.to_dict()

        except Exception as exc:  # broad: must not kill the thread
            logger.exception("Background analysis failed")
            self.failed.emit(str(exc))
            return

        self.finished_with_result.emit(payload)
