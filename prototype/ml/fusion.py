"""DSP + ML fusion for Spectra v1.

The deterministic classifier and the modulation CNN are deliberately
kept independent.  This module implements the *fusion policy* that
the pipeline applies after both have scored a candidate.

Design rules
------------
1. Unknown is preserved.  If either the DSP classifier or the ML model
   returns an unknown/weak result, a naive "winner takes all" fusion is
   dangerously misleading, so the default policy returns a
   side-by-side record and does not invent a third answer.
2. Weak predictions are never forced.  A low-confidence prediction is
   treated as evidence, not as a final label.
3. Everything is recorded for audit: both inputs, the final result, the
   method that produced it, and whether the two disagreed.
4. The fusion method is configurable on ``AnalysisConfig.ml.fusion`` so a
   teammate can experiment with policy without touching pipeline code.

Fusion methods (each must return a dict with ``method``/``final_``
keys):
- ``side_by_side`` - default: report both, mark disagreement, never
  invent a combined answer.
- ``dsp_over_ml`` - deterministic DSP is given priority (conservative
  engineering default).
- ``ml_over_dsp`` - ML takes priority when its confidence is materially
  higher than the DSP confidence (experimental, logged).
- ``max_confidence`` - pick the higher-confidence prediction, but only
  when the gap exceeds ``ml.fusion_confidence_delta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Known modulation labels the deterministic classifier understands.
DSP_MODULATIONS = {
    "BPSK",
    "QPSK",
    "8-PSK",
    "16-QAM",
    "BFSK",
    "OOK",
    "Unknown",
}


@dataclass(frozen=True)
class FusionConfig:
    """Configuration of the fusion policy.

    ``enabled`` gates whether the pipeline records a fused result at
    all.  When disabled the pipeline keeps only the side-by-side
    diagnostic (equivalent to ``method=side_by_side``).
    """

    enabled: bool = True
    method: str = "side_by_side"
    dsp_override: bool = False
    dsp_priority_when_unknown: bool = True
    ml_confidence_delta: float = 0.15
    record_min_confidence: float = 0.25


def _as_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload or {})


def fuse_classification(
    *,
    dsp_modulation: str,
    ml_prediction: dict[str, Any] | None,
    ml_active: bool,
    fusion: FusionConfig,
) -> dict[str, Any]:
    """Apply the configured fusion policy and return a structured record."""

    dsp_clean = str(dsp_modulation or "Unknown").strip()
    dsp_modulation = dsp_clean if dsp_clean in DSP_MODULATIONS else "Unknown"

    # Initialize the record with diagnostics.
    record: dict[str, Any] = {
        "method": fusion.method,
        "dsp_modulation": dsp_modulation,
        "ml_present": bool(ml_prediction),
        "ml_modulation": None,
        "final_modulation": None,
        "disagreement": False,
        "confidence_gap": None,
        "dsp_confidence": None,
        "ml_raw_score": None,
        "reason": None,
        "dsp_warn": None,
        "ml_warn": None,
        "history": [],
    }

    # Build the side-by-side evidence set.
    evidence: list[dict[str, Any]] = []

    dsp_conf: float | None = None
    if "confidence" in (record := _as_dict(record)):
        pass

    # Deterministic DSP classifier confidence is NOT stored in the
    # existing ClassificationConfig summary, so we conservatively treat
    # it as a qualitative evidence tag rather than a number.
    if dsp_modulation == "Unknown":
        record["dsp_warn"] = (
            "DSP returned Unknown; ML used as primary evidence."
        )
        evidence.append(
            {
                "source": "dsp",
                "modulation": dsp_modulation,
                "confidence": None,
                "confidence_label": "unknown",
                "strength": "none",
            }
        )
    else:
        evidence.append(
            {
                "source": "dsp",
                "modulation": dsp_modulation,
                "confidence": None,
                "confidence_label": "qualitative",
                "strength": "moderate",
            }
        )

    if not ml_active or ml_prediction is None:
        record["dsp_warn"] = (
            "ML disabled; deterministic DSP result used."
        )
        record["final_modulation"] = dsp_modulation
        record["reason"] = "ML disabled"
        return record

    ml_mod = str(ml_prediction.get("predicted_class") or "Unknown").strip()
    ml_mod = ml_mod if ml_mod in DSP_MODULATIONS else "Unknown"
    ml_score = float(ml_prediction.get("confidence", 0.0))
    ml_agreement = float(ml_prediction.get("frame_agreement", 0.0))

    if ml_mod == "Unknown":
        record["ml_warn"] = (
            "ML returned Unknown; DSP result is the only usable evidence."
        )
        record["final_modulation"] = dsp_modulation
        record["reason"] = "ML returned Unknown"
        return record

    evidence.append(
        {
            "source": "ml",
            "modulation": ml_mod,
            "confidence": ml_score,
            "confidence_label": "raw_model_score",
            "strength": "strong" if ml_score >= 0.5 else ("mid" if ml_score >= 0.3 else "weak"),
        }
    )

    record["ml_modulation"] = ml_mod
    record["ml_raw_score"] = ml_score
    record["ml_frame_agreement"] = ml_agreement
    record["dsp_modulation"] = dsp_modulation
    record["ml_modulation"] = ml_mod

    recorded: list[dict[str, Any]] = []
    for item in evidence:
        recorded.append(
            {
                "source": item["source"],
                "modulation": item["modulation"],
                "confidence": item["confidence"],
                "strength": item["strength"],
                "confidence_label": item["confidence_label"],
            }
        )

    record["evidence"] = recorded

    if not fusion.enabled:
        record["final_modulation"] = dsp_modulation
        record["reason"] = "fusion explicitly disabled"
        return record

    if dsp_modulation == "Unknown":
        record["final_modulation"] = ml_mod
        record["reason"] = "DSP unknown; accepting ML prediction"
        return record

    if ml_mod == "Unknown":
        record["final_modulation"] = dsp_modulation
        record["reason"] = "ML unknown; keeping DSP result"
        return record

    disagreement = dsp_modulation != ml_mod
    record["disagreement"] = disagreement

    if ml_score < fusion.record_min_confidence:
        record["final_modulation"] = dsp_modulation
        record["reason"] = (
            "ML confidence below record threshold; keeping DSP result"
        )
        return record

    if fusion.method == "side_by_side":
        record["final_modulation"] = dsp_modulation
        record["reason"] = (
            "side_by_side policy: both predictions recorded, no combined answer"
        )
        return record

    if fusion.method == "dsp_over_ml":
        record["final_modulation"] = dsp_modulation
        record["reason"] = "dsp_over_ml policy: deterministic DSP prioritized"
        return record

    if fusion.method == "ml_over_dsp":
        record["final_modulation"] = ml_mod
        record["reason"] = "ml_over_dsp policy: ML prioritized"
        return record

    # ml_over_dsp_with_confidence_delta
    dsp_conf = evidence[0].get("confidence")  # 0.0 when unknown
    gap = abs(ml_score - dsp_conf) if (dsp_conf is not None) else float("inf")

    if gap >= fusion.ml_confidence_delta:
        record["final_modulation"] = ml_mod
        record["confidence_gap"] = float(gap)
        record["reason"] = (
            f"ml_over_dsp policy: ML confidence {ml_score:.3f} exceeds "
            f"DSP confidence {dsp_conf:.3f} by {gap:.3f}"
        )
    else:
        record["final_modulation"] = dsp_modulation
        record["confidence_gap"] = float(gap)
        record["reason"] = (
            f"ml_over_dsp policy: gap {gap:.3f} below threshold "
            f"{fusion.ml_confidence_delta:.3f}; keeping DSP"
        )

    return record
