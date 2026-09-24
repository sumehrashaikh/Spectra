"""
Provenance tracking for reproducible analysis.

Every pipeline run records software version, git commit, input file
hash, configuration, and per-stage timings so any analysis can be
reproduced and audited.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _package_version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _git_commit() -> str | None:
    """Best-effort git commit lookup. Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ProcessingStep:
    """One executed processing stage with its timing and outcome."""

    name: str
    duration_ms: float
    status: str  # "ok" | "skipped" | "failed"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisProvenance:
    """Complete provenance record for one analysis run."""

    software_version: str = field(default_factory=_package_version)
    python_version: str = field(default_factory=_python_version)
    git_commit: str | None = field(default_factory=_git_commit)
    started_utc: str = ""
    finished_utc: str = ""
    input_file: str | None = None
    input_sha256: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    steps: list[ProcessingStep] = field(default_factory=list)

    def start(self) -> None:
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def finish(self) -> None:
        self.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def record_step(self, name: str, status: str = "ok"):
        """Context manager recording a timed processing step.

        Usage::

            with provenance.record_step("detection") as step:
                ...
                step["candidates"] = 3
        """

        provenance = self  # captured by the step context below

        class _StepCtx:
            def __init__(self):
                self.start = time.perf_counter()
                self.detail: dict[str, Any] = {}

            def __enter__(self):
                return self.detail

            def __exit__(self, exc_type, exc, tb):
                duration_ms = (time.perf_counter() - self.start) * 1000.0
                if exc_type is None:
                    status_value = "ok"
                elif exc_type is _SkippedStage:
                    status_value = "skipped"
                    exc = exc.args[0] if exc.args else None
                else:
                    status_value = "failed"
                provenance.steps.append(
                    ProcessingStep(
                        name=name,
                        duration_ms=round(duration_ms, 3),
                        status=status_value,
                        detail=self.detail,
                    )
                )
                if exc_type is None or exc_type is _SkippedStage:
                    return True  # swallow skip
                return False  # propagate real failures

        return _StepCtx()


class _SkippedStage(Exception):
    """Internal: raise inside record_step to mark a stage skipped."""


def skip_stage(message: str) -> None:
    """Raise inside a record_step block to mark the stage as skipped."""
    raise _SkippedStage(message)


def hash_samples(samples, sample_rate: float) -> str:
    """Deterministic hash of in-memory samples (for non-file inputs)."""
    import numpy as np

    data = np.asarray(samples)
    digest = hashlib.sha256()
    digest.update(data.tobytes())
    digest.update(f"|{sample_rate!r}".encode())
    return digest.hexdigest()


def new_provenance(
    input_file: str | None = None,
    configuration: dict[str, Any] | None = None,
) -> AnalysisProvenance:
    """Create and start a provenance record."""
    provenance = AnalysisProvenance(
        input_file=input_file,
        configuration=configuration or {},
    )
    if input_file is not None:
        try:
            provenance.input_sha256 = sha256_file(input_file)
        except OSError:
            provenance.input_sha256 = None
    provenance.start()
    return provenance


def provenance_to_dict(provenance: AnalysisProvenance) -> dict[str, Any]:
    """Serialize provenance for JSON export."""
    return {
        "software_version": provenance.software_version,
        "python_version": provenance.python_version,
        "git_commit": provenance.git_commit,
        "started_utc": provenance.started_utc,
        "finished_utc": provenance.finished_utc,
        "input_file": provenance.input_file,
        "input_sha256": provenance.input_sha256,
        "configuration": provenance.configuration,
        "steps": [
            {
                "name": step.name,
                "duration_ms": step.duration_ms,
                "status": step.status,
                "detail": step.detail,
            }
            for step in provenance.steps
        ],
    }
