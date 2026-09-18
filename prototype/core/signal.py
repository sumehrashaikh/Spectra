"""
Spectra V2 - Signal Data Model

Defines the common signal representation used throughout
the Spectra processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Signal:
    """
    Common representation of an RF/IQ signal.

    Parameters
    ----------
    samples : np.ndarray
        Complex-valued signal samples.

    sample_rate : float
        Sampling frequency in Hz.

    metadata : dict
        Additional information produced by processing stages.
    """

    samples: np.ndarray
    sample_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and standardize the signal."""

        self.samples = np.asarray(
            self.samples,
            dtype=np.complex128,
        )

        if self.samples.size == 0:
            raise ValueError(
                "Signal contains no samples."
            )

        if not np.all(np.isfinite(self.samples)):
            raise ValueError(
                "Signal contains NaN or infinite values."
            )

        if self.sample_rate <= 0:
            raise ValueError(
                "Sample rate must be positive."
            )

        self.sample_rate = float(self.sample_rate)

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""

        return len(self.samples)

    @property
    def duration(self) -> float:
        """Return signal duration in seconds."""

        return self.num_samples / self.sample_rate

    @property
    def rms(self) -> float:
        """Return RMS magnitude of the signal."""

        return float(
            np.sqrt(
                np.mean(
                    np.abs(self.samples) ** 2
                )
            )
        )

    @property
    def peak(self) -> float:
        """Return peak magnitude of the signal."""

        return float(
            np.max(
                np.abs(self.samples)
            )
        )

    def copy(self) -> "Signal":
        """Return an independent copy of the signal."""

        return Signal(
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            metadata=self.metadata.copy(),
        )

    def add_metadata(self, **values):
        """
        Add or update signal metadata.

        Example
        -------
        signal.add_metadata(
            snr_db=18.5,
            modulation="QPSK",
        )
        """

        self.metadata.update(values)

    def summary(self) -> dict[str, Any]:
        """
        Return a basic signal summary.
        """

        return {
            "num_samples": self.num_samples,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "rms": self.rms,
            "peak": self.peak,
            "metadata": self.metadata.copy(),
        }