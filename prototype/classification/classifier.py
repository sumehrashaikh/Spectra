from dataclasses import dataclass
from typing import Any

import numpy as np

from prototype.core.signal import Signal

from prototype.modulation.classifier import (
    classify_modulation,
    classify_from_constellation,
)


@dataclass
class ClassificationResult:
    """Result produced by the V2 classification layer."""

    modulation: str
    confidence: float
    method: str
    features: dict[str, Any]


def _calculate_feature_confidence(
    features: dict[str, Any],
    modulation: str,
) -> float:
    """
    Calculate a simple confidence estimate from the
    existing V1 classifier features.

    This is a heuristic confidence estimate for the V2
    wrapper. It does not replace the V1 classifier rules.
    """

    if modulation == "Unknown":
        return 0.0

    scores = []

    if modulation == "BPSK":
        r2 = float(
            features.get(
                "r2_phase_coherence",
                0.0,
            )
        )

        scores.append(
            np.clip(
                r2,
                0.0,
                1.0,
            )
        )

    elif modulation == "QPSK":
        r4 = float(
            features.get(
                "r4_phase_coherence",
                0.0,
            )
        )

        scores.append(
            np.clip(
                r4,
                0.0,
                1.0,
            )
        )

    elif modulation == "BFSK":
        freq_std = float(
            features.get(
                "instantaneous_frequency_std_hz",
                0.0,
            )
        )

        # The existing V1 decision boundary uses
        # 80 Hz as the frequency-variation threshold.
        scores.append(
            np.clip(
                freq_std / 160.0,
                0.0,
                1.0,
            )
        )

    elif modulation == "16-QAM":
        spread = float(
            features.get(
                "robust_amplitude_spread",
                0.0,
            )
        )

        scores.append(
            np.clip(
                spread,
                0.0,
                1.0,
            )
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores) * 100.0
    )


def classify_waveform(
    signal: Signal,
) -> ClassificationResult:
    """
    Classify a Signal using the existing V1 waveform
    classifier.

    This preserves the existing V1 classification logic.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "classify_waveform expects a Signal object."
        )

    modulation, features = classify_modulation(
        signal.samples,
        signal.sample_rate,
    )

    confidence = _calculate_feature_confidence(
        features,
        modulation,
    )

    return ClassificationResult(
        modulation=modulation,
        confidence=confidence,
        method="waveform_features",
        features=features,
    )


def classify_constellation(
    signal: Signal,
) -> ClassificationResult:
    """
    Classify symbol-rate samples using the existing
    constellation classifier.

    Intended for synchronized symbol samples.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "classify_constellation expects a Signal object."
        )

    modulation, features = classify_from_constellation(
        signal.samples,
    )

    confidence = 0.0

    if modulation == "BPSK":
        axis_ratio = float(
            features.get(
                "axis_ratio",
                0.0,
            )
        )

        confidence = np.clip(
            (0.20 - axis_ratio) / 0.20,
            0.0,
            1.0,
        ) * 100.0

    elif modulation == "QPSK":
        i_symmetry = float(
            features.get(
                "i_two_symmetry",
                0.0,
            )
        )

        q_symmetry = float(
            features.get(
                "q_two_symmetry",
                0.0,
            )
        )

        confidence = (
            min(
                i_symmetry,
                q_symmetry,
            )
            * 100.0
        )

    elif modulation == "16-QAM":
        i_spacing = float(
            features.get(
                "i_spacing_ratio",
                0.0,
            )
        )

        q_spacing = float(
            features.get(
                "q_spacing_ratio",
                0.0,
            )
        )

        confidence = (
            min(
                i_spacing,
                q_spacing,
            )
            * 100.0
        )

    return ClassificationResult(
        modulation=modulation,
        confidence=float(
            np.clip(
                confidence,
                0.0,
                100.0,
            )
        ),
        method="constellation_geometry",
        features=features,
    )

def classify_signal(
    signal: Signal,
    use_constellation: bool = True,
    synchronized: bool = False,
) -> ClassificationResult:
    """
    Main V2 classification entry point.

    Parameters
    ----------
    signal:
        V2 Signal object.

    use_constellation:
        Whether constellation geometry may be used.

    synchronized:
        True only when the input contains recovered
        symbol-rate samples.

        Raw waveform samples should use False.

    Strategy
    --------
    Raw waveform:
        waveform features -> final decision

    Synchronized symbols:
        waveform features + constellation geometry
        -> final decision

    This prevents raw BFSK waveforms from being
    incorrectly interpreted as PSK/QAM constellations.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "classify_signal expects a Signal object."
        )

    waveform_result = classify_waveform(signal)

    # ---------------------------------------------------------
    # Raw waveform classification
    # ---------------------------------------------------------

    if not use_constellation or not synchronized:
        return waveform_result

    # ---------------------------------------------------------
    # Synchronized symbol classification
    # ---------------------------------------------------------

    constellation_result = classify_constellation(
        signal
    )

    if constellation_result.modulation != "Unknown":
        return constellation_result

    return waveform_result

def classification_to_dict(
    result: ClassificationResult,
) -> dict[str, Any]:
    """Convert ClassificationResult into a serializable dictionary."""

    return {
        "modulation": result.modulation,
        "confidence": result.confidence,
        "method": result.method,
        "features": result.features,
    }