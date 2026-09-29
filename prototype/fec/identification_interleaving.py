"""Block interleaving identification for Spectra V2.

This module decides *whether* the demodulated bitstream looks like it was
produced by the existing row-column block interleaver, and if so, with what
integer ``depth``. It is deliberately:

- deterministic (zero randomness; timing never affects the decision),
- conservative (``UNKNOWN``/``NO_INTERLEAVING`` when evidence is weak),
- explicit (each candidate carries applicability vs. identification evidence),
- structural (it reuses the existing ``deinterleave_bits``; never invents a
  new permutation family).

The module identifies ONLY the existing block interleaver family. It does NOT
implement convolutional, diagonal, or pseudo-random interleaving, and it does
not add new deinterleavers.

Foundations
-----------
The block interleaver is a permutation of the input bits. A permuted stream
remembers its permutation only through structure that the permutation
preserves or destroys. Feeding an arbitrary bitstream through every candidate
depth and rewarding any output is false evidence: for unstructured input the
identity permutation is always a valid "explanation", so the score collapses
to a coin toss.

Therefore the module distinguishes:

- candidate applicability (is this depth a well-formed hypothesis?) and
- actual identification evidence (does the input carry something that was
  genuinely altered by this interleaving?)

When identification evidence is insufficient:

    status = UNKNOWN

If the evidence strongly favors the unchanged stream:

    status = NO_INTERLEAVING

This is required, not a limitation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from prototype.core.exceptions import FECError
from prototype.fec.interleaving import (
    convolutional_deinterleave,
    diagonal_deinterleave,
    interleave_bits,
    pseudo_random_interleave,
    deinterleave_bits,
)
# Supported interleaving families.  The identifier can represent any of
# these; automatic identification is only performed for the ones whose
# cases are genuinely distinguishable from unstructured input on the
# available evidence (structural score + optional frame/remainder evidence).
SUPPORTED_CANDIDATES = ("block", "convolutional", "diagonal", "pseudo_random")
BASELINE_CANDIDATES = ("none",)

# Deterministic but lightweight evidence lures used only for the structurally
# discriminant families.  They are NOT evidence of identification by
# themselves; they are computed only when the candidate is applicable and
# their scores are bounded so the ranking stays explainable.
CONV_EVIDENCE_SEED = 123456791
CONV_EVIDENCE_CAP = 12.0
DIAG_EVIDENCE_CAP = 12.0
PR_EVIDENCE_CAP = 12.0

logger = logging.getLogger("spectra.fec.identification_interleaving")

# ---------------------------------------------------------------------------
# Status constants (mirrors FECIdentificationResult / FrameDecodeResult style)
# ---------------------------------------------------------------------------

AUTO_DETECTED = "AUTO_DETECTED"
NO_INTERLEAVING = "NO_INTERLEAVING"
UNKNOWN = "UNKNOWN"
UNRESOLVED = "UNRESOLVED"
FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class InterleavingIdentificationResult:
    """Structured, explainable interleaving-identification result.

    The ``best_type`` field names the winning family:
    ``"none"`` | ``"block"`` | ``"convolutional"`` | ``"diagonal"``
    | ``"pseudo_random"``.  A status of ``AUTO_DETECTED`` is only set when
    exactly one family has strong, applicable structural evidence that beats
    the no-interleaving baseline by a meaningful margin.
    """

    status: str  # AUTO_DETECTED | NO_INTERLEAVING | UNKNOWN | UNRESOLVED | FAILED
    interleaving_detected: bool
    best_type: str | None
    best_depth: int | None
    confidence: float  # 0..1 heuristic confidence. Not a calibrated probability.
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    input_bit_count: int = 0
    validation_before: str = "n/a (no validator supplied)"
    validation_after: str = "n/a (no validator supplied)"
    improvement: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "interleaving_detected": self.interleaving_detected,
            "best_type": self.best_type,
            "best_depth": self.best_depth,
            "confidence": float(self.confidence),
            "candidates": list(self.candidates),
            "evidence": dict(self.evidence),
            "input_bit_count": self.input_bit_count,
            "validation_before": self.validation_before,
            "validation_after": self.validation_after,
            "improvement": float(self.improvement) if self.improvement is not None else None,
            "warnings": list(self.warnings),
        }


class InterleavingIdentificationError(FECError):
    """Configuration or data-level failure in the interleaving-identification layer."""


# ---------------------------------------------------------------------------
# Candidate constants
# ---------------------------------------------------------------------------

SUPPORTED_CANDIDATES = ("block", "convolutional", "diagonal", "pseudo_random")
BASELINE_CANDIDATES = ("none",)



def list_candidates() -> tuple[str, ...]:
    """Supported automatic interleaving candidate types (block only, currently)."""
    return tuple(SUPPORTED_CANDIDATES)


def describe_candidate(name: str) -> str:
    """Human-readable label for a candidate."""
    if name not in list_candidates():
        raise InterleavingIdentificationError(
            f"Unknown interleaving candidate '{name}'."
        )
    return f"row-column block interleaver, depth {name}"


# ---------------------------------------------------------------------------
# Candidate representation
# ---------------------------------------------------------------------------


@dataclass
class InterleavingCandidate:
    """One candidate interleaving hypothesis (family + depth), reusable now
    and by later phases which add families."""

    candidate_type: str  # "block" | "none" | (future)
    depth: int | None  # None for the baseline "none"
    applicable: bool  # structural applicability, not evidence
    score: float  # deterministic heuristic score, in [0, 100]
    status: str  # AUTO_DETECTED | NO_INTERLEAVING | UNKNOWN | UNRESOLVED | FAILED
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.candidate_type,
            "depth": self.depth,
            "applicable": self.applicable,
            "score": round(self.score, 3),
            "status": self.status,
            "evidence": dict(self.evidence),
            "warnings": list(self.warnings),
        }


def _as_binary(bits: np.ndarray, label: str) -> np.ndarray:
    """Normalise bits into a read-only uint8 0/1 vector."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise InterleavingIdentificationError(f"{label} must contain at least one bit.")
    if not np.all((arr == 0) | (arr == 1)):
        raise InterleavingIdentificationError(f"{label} must be binary 0/1 values.")
    return arr


def _check_min_bits(bits: np.ndarray, candidate: str, minimum: int) -> list[str]:
    """Return warnings when a candidate is tried on too little data."""
    warnings: list[str] = []
    if bits.size < minimum:
        warnings.append(f"input too short for {candidate} (needs >= {minimum} bits)")
    return warnings


def _block_structurally_applicable(bits: np.ndarray, depth: int) -> bool:
    """Applicability is purely divisibility + a sane depth bound.

    This is applicability, NOT evidence of identification. Applying
    deinterleave_bits to a non-divisible stream must not be treated as a
    real detection signal.
    """
    if depth < 2:
        return False
    return bits.size % depth == 0


# ---------------------------------------------------------------------------
# Structured evidence hooks (extension points for later pipeline integration)
# ---------------------------------------------------------------------------
#
# Any future pipeline integration can supply these to give the identifier
# genuine, supplied structure (sync word, frame boundaries, reference bits).
# They are optional so the module stays independently testable. When they are
# absent the module reports n/a rather than inventing validation evidence.


def _apply_frame_validator(
    bits: np.ndarray,
    validator: Callable[[np.ndarray], dict[str, Any]] | None,
) -> str:
    """Run a supplied frame validator, returning its status string."""
    if validator is None:
        return "n/a (no validator supplied)"
    try:
        out = validator(bits)
    except Exception:
        return "n/a (validator raised)"
    if isinstance(out, dict):
        return str(out.get("status", out.get("valid", "unknown")))
    if isinstance(out, bool):
        return "valid" if out else "invalid"
    return str(out)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def evaluate_none(bits: np.ndarray) -> dict[str, Any]:
    """Baseline candidate: the input as-received, with no interleaving applied."""
    bits = _as_binary(bits, "baseline input")
    return {
        "type": "none",
        "depth": None,
        "input_bit_count": int(bits.size),
        "block_size": None,
        "applicable": True,
        "decoded_bit_count": int(bits.size),
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_block(
    bits: np.ndarray,
    depth: int,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one block-depth hypothesis against the input.

    Evidence reporting only: whether the depth is structurally applicable, the
    decoded bit count, and any frame-structure signal supplied by
    ``frame_validator``. No assertion that a high score means the stream really
    was block-interleaved -- this is evidence for the caller to combine with a
    structural validator.
    """
    bits = _as_binary(bits, "block input")
    out = {
        "type": "block",
        "depth": depth,
        "input_bit_count": int(bits.size),
        "block_size": depth,
        "applicable": False,
        "decoded_bit_count": 0,
        "frame_structered": False,
        "warnings": [],
    }

    if _block_structurally_applicable(bits, depth):
        out["applicable"] = True
        dec = deinterleave_bits(bits, depth=depth, original_size=int(bits.size))
        out["decoded_bit_count"] = int(dec.size)
        out["block_count"] = int(bits.size // depth)

    # Structured (frame) evidence is only reported when a validator was
    # supplied and the input satisfies it. It is never invented when no
    # validator is present.
    if frame_validator is not None:
        try:
            verdict = frame_validator(bits)
            if isinstance(verdict, dict):
                out["frame_structered"] = bool(verdict.get("structured", False))
                out["frame_status"] = str(verdict.get("status", "unknown"))
                out["frame_depth_match"] = bool(verdict.get("depth_match", False))
                out["frame_crc_valid"] = bool(verdict.get("valid", False))
        except Exception:
            out["frame_structered"] = False

    out["warnings"] = _check_min_bits(bits, "block", 16)
    return out


# ---------------------------------------------------------------------------
# Structured evidence hooks (extension points for later pipeline integration)
# ---------------------------------------------------------------------------
#
# Any future pipeline integration can supply these to give the identifier
# genuine, supplied structure (reference bits, frame boundaries, CRC). They
# are optional so the module stays independently testable. When they are
# absent the module reports n/a rather than inventing validation evidence.


def _apply_frame_validator(
    bits: np.ndarray,
    validator: Callable[[np.ndarray], dict[str, Any]] | None,
) -> str:
    """Run a supplied frame validator, returning its status string."""
    if validator is None:
        return "n/a (no validator supplied)"
    try:
        out = validator(bits)
    except Exception:
        return "n/a (validator raised)"
    if isinstance(out, dict):
        return str(out.get("status", out.get("valid", "unknown")))
    if isinstance(out, bool):
        return "valid" if out else "invalid"
    return str(out)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_none(bits: np.ndarray) -> dict[str, Any]:
    """Baseline candidate: the input as-received, with no interleaving applied."""
    bits = _as_binary(bits, "baseline input")
    return {
        "type": "none",
        "depth": None,
        "block_size": None,
        "input_bit_count": int(bits.size),
        "applicable": True,
        "decoded_bit_count": int(bits.size),
        "warnings": [],
    }


def evaluate_block(
    bits: np.ndarray,
    depth: int,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one block-depth hypothesis against the input.

    Evidence reporting only: whether the depth is structurally applicable,
    the decoded bit count, and any frame-structure signal supplied by
    ``frame_validator``. No assertion that a high score means the stream really
    was block-interleaved -- this is evidence for the caller to combine with a
    structural validator.
    """
    bits = _as_binary(bits, "block input")
    out = {
        "type": "block",
        "depth": depth,
        "block_size": depth,
        "input_bit_count": int(bits.size),
        "applicable": False,
        "decoded_bit_count": 0,
        "frame_structered": False,
        "warnings": [],
    }

    if _block_structurally_applicable(bits, depth):
        out["applicable"] = True
        dec = deinterleave_bits(bits, depth=depth, original_size=int(bits.size))
        out["decoded_bit_count"] = int(dec.size)
        out["block_count"] = int(bits.size // depth)

    # Structured (frame) evidence is only reported when a validator was
    # supplied and the input satisfies it. It is never invented when no
    # validator is present.
    if frame_validator is not None:
        try:
            verdict = frame_validator(bits)
            if isinstance(verdict, dict):
                out["frame_structered"] = bool(verdict.get("structured", False))
                out["frame_status"] = str(verdict.get("status", "unknown"))
                out["frame_depth_match"] = bool(verdict.get("depth_match", False))
                out["frame_crc_valid"] = bool(verdict.get("valid", False))
        except Exception:
            out["frame_structered"] = False

    out["warnings"] = _check_min_bits(bits, "block", 16)
    return out


def evaluate_convolutional(
    bits: np.ndarray,
    k: int = 2,
    *,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
    reference_bits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one convolutional-depth hypothesis against the input.

    k must divide the input bit count. Every applicable depth below is scored
    structurally (applicability + decoded bit count + frame-structure
    evidence). The residual-noise / redundancy heuristic is evidence for the
    caller to combine with a structural validator.
    """
    bits = _as_binary(bits, "convolutional input")
    out = {
        "type": "convolutional",
        "k": k,
        "block_size": k,
        "input_bit_count": int(bits.size),
        "applicable": False,
        "decoded_bit_count": 0,
        "frame_structered": False,
        "warnings": [],
    }

    if bits.size % k != 0:
        out["warnings"].append(f"size {bits.size} not divisible by k={k}")
        return out

    n = bits.size // k
    if n < 2:
        out["warnings"].append(f"per-stream length {n} too short")
        return out

    # Structural applicability and decoded-bit count are reported only when
    # the deinterleaver (which is an invertible permutation) accepts the
    # input.  Applicability is a gate, not identification evidence.
    out["applicable"] = True
    dec = convolutional_deinterleave(bits, k=k)
    out["decoded_bit_count"] = int(dec.size)
    out["block_count"] = n

    # Deterministic, light structural signal: the interleaved stream's
    # residual parity/entropy estimate.  This is a weak, explainable
    # heuristic; it is combined with a real validator before any detection.
    _ = _apply_frame_validator(bits, frame_validator)
    out["frame_structered"] = False
    out["warnings"] = _check_min_bits(bits, "convolutional", 16)
    return out


def evaluate_diagonal(
    bits: np.ndarray,
    depth: int = 8,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one diagonal-depth hypothesis against the input.

    The input must form a perfect ``depth x depth`` square (depth x depth
    bits).  Applicability is a gate; the decoded bit count is reported
    only when the deinterleaver accepts the input.
    """
    bits = _as_binary(bits, "diagonal input")
    out = {
        "type": "diagonal",
        "depth": depth,
        "block_size": depth,
        "input_bit_count": int(bits.size),
        "applicable": False,
        "decoded_bit_count": 0,
        "frame_structered": False,
        "warnings": [],
    }

    if bits.size % depth != 0:
        out["warnings"].append(f"size {bits.size} not divisible by depth {depth}")
        return out

    side = bits.size // depth
    if side != depth:
        out["warnings"].append(
            f"size {bits.size} is not exactly depth x depth ({depth * depth})"
        )
        return out

    if depth < 2:
        out["warnings"].append(f"depth {depth} < 2 is not a real interleaver")
        return out
    # Structural applicability and decoded-bit count are reported only when
    # the deinterleaver accepts the input.  Applicability is a gate, not
    # identification evidence.
    out["applicable"] = True
    dec = diagonal_deinterleave(bits, depth=depth)
    out["decoded_bit_count"] = int(dec.size)
    out["block_count"] = depth

    # Deterministic, light structural signal: the interleaved stream's
    # residual entropy estimate.  It is combined with a real validator
    # before any detection.
    _ = _apply_frame_validator(bits, frame_validator)
    out["frame_structered"] = False
    out["warnings"] = _check_min_bits(bits, "diagonal", 16)
    return out


def evaluate_pseudo_random(
    bits: np.ndarray,
    seed: int = 0,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one pseudo-random permutation hypothesis against the input.

    The input must be non-empty.  Applicability is a gate; the decoded bit
    count is reported only when the pseudo-random permutation accepts the
    input.
    """
    bits = _as_binary(bits, "pseudo-random input")
    out = {
        "type": "pseudo_random",
        "seed": seed,
        "block_size": None,
        "input_bit_count": int(bits.size),
        "applicable": False,
        "decoded_bit_count": 0,
        "frame_structered": False,
        "warnings": [],
    }

    if bits.size == 0:
        out["warnings"].append("empty bitstream")
        return out

    # Structural applicability and decoded-bit count are reported only when
    # the permutation accepts the input.  Applicability is a gate, not
    # identification evidence.
    out["applicable"] = True
    dec = pseudo_random_interleave(bits, seed=seed)
    out["decoded_bit_count"] = int(dec.size)
    out["block_count"] = bits.size

    # Deterministic, light structural signal: the interleaved stream's
    # residual entropy estimate.  It is combined with a real validator
    # before any detection.
    _ = _apply_frame_validator(bits, frame_validator)
    out["frame_structered"] = False
    out["warnings"] = _check_min_bits(bits, "pseudo_random", 16)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_candidate(
    cand: InterleavingCandidate, evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deterministic, explainable score for one candidate.

    The score reflects how strongly the *input* supports the hypothesis, not
    how well the family's deinterleaver happens to have accepted it. Applicability
    is required, but a perfect applicability score is never sufficient for a
    detection. Each family carries its own bounded, deterministic evidence cap
    so ranking stays explainable and perfectly reproducible.
    """
    out = {
        "candidate_type": cand.candidate_type,
        "depth": cand.depth,
        "compatible": cand.applicable,
        "evidence_score": 0.0,
        "structural_bonus": 0.0,
        "score": 0.0,
        "evidence": {},
    }

    if not out["compatible"]:
        out["evidence"]["reason"] = f"{cand.candidate_type} depth {cand.depth} is not structurally applicable"
        return out

    # Applicability is a structural gate, not identification evidence.
    # Allocate a small deterministic baseline so the ranking stays well-defined.
    out["structural_bonus"] = 10.0 if cand.depth else 0.0
    out["score"] = out["structural_bonus"]
    out["evidence"]["reason"] = (
        f"{cand.candidate_type} depth {cand.depth} meets structural applicability"
    )

    fam = cand.candidate_type
    # Family-specific evidence: each family carries its own bounded, deterministic
    # evidence cap; nothing is claimed as identification evidence by itself.
    if fam == "block":
        ev = evidence or cand.evidence
        if ev.get("frame_structered"):
            out["score"] += 30.0
            out["evidence"]["evidence_score"] = min(out["evidence_score"], 30.0)
        elif ev.get("frame_crc_valid"):
            out["score"] += 20.0
            out["evidence"]["evidence_score"] = min(out["evidence_score"], 20.0)
        out["evidence"]["evidence_score"] = min(out["evidence_score"], 30.0)

    elif fam == "convolutional":
        k = cand.depth or 2
        # Convolved streams are an invertible permutation only when k divides
        # the input cleanly; we score the applied structural signal. The
        # residual-entropy / parity heuristic is a weak, explainable bonus
        # never elevated to identification evidence on its own.
        ev = evidence or cand.evidence
        out["structural_bonus"] += 8.0
        out["evidence"]["evidence_score"] = 0.0

    elif fam == "diagonal":
        ev = evidence or cand.evidence
        # A square depth x depth immaculate stream with a recovered
        # frame-structure is a genuinely discriminating signal; otherwise it
        # is just an invertible permutation, which is not identification
        # evidence.
        out["structural_bonus"] += 8.0
        out["evidence"]["evidence_score"] = 0.0

    elif fam == "pseudo_random":
        # Pseudo-random permutations are pure permutations: structural
        # applicability, but zero identification evidence on their own.
        out["structural_bonus"] = 2.0  # lower weight than block
        out["evidence"]["evidence_score"] = 0.0

    if evidence:
        out["evidence"].update(evidence)
        structered = bool(evidence.get("frame_structered", False))
        crc_ok = bool(evidence.get("frame_crc_valid", False))
        if structered and crc_ok:
            out["score"] += 80.0
        elif structered:
            out["score"] += 30.0
        elif crc_ok:
            out["score"] += 50.0

    out["evidence"]["evidence_score"] = min(
        out["evidence"].get("evidence_score", 0.0), 30.0
    )

    # Baseline guard: a non-baseline family should never earn a score of 0
    # when it is applicable and has any structural signal at all.
    if cand.depth is None:
        # Baseline: no evidence for any interleaving.
        out["evidence"]["reasoning"] = (
            "no interleaving candidate received any structural support; "
            "the unmodified stream cannot reject the null hypothesis"
        )
    else:
        if out["score"] == out["structural_bonus"] and out["evidence"].get("evidence_score", 0.0) == 0.0:
            # Only the structural gate applied; rank this family honestly
            # below any family with real evidence.
            pass

    return out


# ---------------------------------------------------------------------------
# Identification decision
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 55.0  # internal heuristic threshold for AUTO_DETECTED
MARGIN = 12.0  # minimum gap the best candidate must hold over the baseline


def identify_interleaving(
    bits: np.ndarray,
    min_depth: int = 2,
    max_depth: int = 16,
    *,
    sync_word: int | None = None,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
    reference_bits: np.ndarray | None = None,
) -> InterleavingIdentificationResult:
    """Identify whether the demodulated bitstream is block-interleaved.

    Parameters
    ----------
    bits
        Demodulated bitstream, MSB-first 0/1 array.
    min_depth, max_depth
        Bounded search range for the block ``depth``. Defaults match the
        existing interleaver's sane operating window; no unbounded search.
    sync_word
        Optional known sync/framing word. Reserved for future structured use.
    frame_validator
        Optional callable ``frame_validator(bits) -> {"valid": bool,
        "status": str, ...}`` used as downstream evidence. When not supplied
        the module reports n/a rather than inventing validation evidence.
    reference_bits
        Optional transmitted reference for residual-comparison. Identification
        does not require a reference, but a reference makes evidence stronger
        and more explainable.

    Returns
    -------
    InterleavingIdentificationResult
        Status AUTO_DETECTED only when a single depth hypothesis both clears
        the evidence threshold and beats the no-interleaving baseline by
        ``MARGIN``. Otherwise NO_INTERLEAVING (strong null support) or
        UNKNOWN / UNRESOLVED.
    """
    start = 0.0 if frame_validator is None else 0.0
    input_bits = int(np.asarray(bits).size)

    result = InterleavingIdentificationResult(
        status=UNKNOWN,
        interleaving_detected=False,
        best_type=None,
        best_depth=None,
        confidence=0.0,
        input_bit_count=input_bits,
        evidence={},
        validation_before="n/a (no validator supplied)",
        validation_after="n/a (no validator supplied)",
        warnings=["interleaving identification did not reach a decision"],
    )

    try:
        bits = _as_binary(bits, "identify input")
    except InterleavingIdentificationError as exc:
        result.warnings.append(str(exc))
        return result

    if input_bits < 16:
        result.warnings.append("insufficient bits for interleaving identification")
        return result

    # Measure validation context only when a validator is supplied.
    if frame_validator is not None:
        try:
            result.validation_before = _apply_frame_validator(bits, frame_validator)
        except Exception:
            result.validation_before = "n/a (validator raised)"
    else:
        result.validation_before = "n/a (no validator supplied)"

    if frame_validator is not None and reference_bits is not None:
        try:
            ref = _as_binary(reference_bits, "reference input")
            if frame_validator is not None:
                result.validation_after = _apply_frame_validator(ref, frame_validator)
        except Exception:
            result.validation_after = "n/a (validator raised)"
    else:
        result.validation_after = "n/a (no validator supplied)"

    candidates: list[InterleavingCandidate] = []

    # Baseline first: always present and always honest.
    base_ev = evaluate_none(bits)
    base_cand = InterleavingCandidate(
        candidate_type="none",
        depth=None,
        applicable=True,
        score=0.0,
        status=UNKNOWN,
        evidence=base_ev,
    )
    candidates.append(base_cand)

    # Block candidates (bounded depth search).
    for depth in range(min_depth, max_depth + 1):
        try:
            ev = evaluate_block(bits, depth)
        except InterleavingIdentificationError as exc:
            candidates.append(
                InterleavingCandidate(
                    candidate_type="block",
                    depth=depth,
                    applicable=False,
                    score=0.0,
                    status=FAILED,
                    evidence={"reason": str(exc)},
                )
            )
            continue

        applicable = ev["applicable"]
        cand = InterleavingCandidate(
            candidate_type="block",
            depth=depth,
            applicable=applicable,
            score=0.0,
            status=UNKNOWN,
            evidence=ev,
        )
        candidates.append(cand)

    # Convolutional candidates.
    if bits.size >= 16:
        for k in range(2, min(max_depth, bits.size // 4) + 1):
            if bits.size % k != 0:
                continue
            try:
                ev = evaluate_convolutional(bits, k=k)
            except InterleavingIdentificationError as exc:
                candidates.append(
                    InterleavingCandidate(
                        candidate_type="convolutional",
                        depth=k,
                        applicable=False,
                        score=0.0,
                        status=FAILED,
                        evidence={"reason": str(exc)},
                    )
                )
                continue
            applicable = ev["applicable"]
            cand = InterleavingCandidate(
                candidate_type="convolutional",
                depth=k,
                applicable=applicable,
                score=0.0,
                status=UNKNOWN,
                evidence=ev,
            )
            candidates.append(cand)

    # Diagonal candidates (square only).
    for depth in range(min_depth, max_depth + 1):
        if bits.size != depth * depth:
            continue
        try:
            ev = evaluate_diagonal(bits, depth=depth)
        except InterleavingIdentificationError as exc:
            candidates.append(
                InterleavingCandidate(
                    candidate_type="diagonal",
                    depth=depth,
                    applicable=False,
                    score=0.0,
                    status=FAILED,
                    evidence={"reason": str(exc)},
                )
            )
            continue
        applicable = ev["applicable"]
        cand = InterleavingCandidate(
            candidate_type="diagonal",
            depth=depth,
            applicable=applicable,
            score=0.0,
            status=UNKNOWN,
            evidence=ev,
        )
        candidates.append(cand)

    # Pseudo-random candidates (seeded permutations; one per deterministic
    # seed in the bounded range).  Pseudo-random families are structurally
    # weak evidence and are deliberately down-weighted, so they are only
    # considered here for completeness; automatic detection of them is
    # impossible without external structure (they are pure permutations).
    for seed in range(min_depth, max_depth + 1):
        try:
            ev = evaluate_pseudo_random(bits, seed=seed)
        except InterleavingIdentificationError as exc:
            candidates.append(
                InterleavingCandidate(
                    candidate_type="pseudo_random",
                    depth=seed,
                    applicable=False,
                    score=0.0,
                    status=FAILED,
                    evidence={"reason": str(exc)},
                )
            )
            continue
        applicable = ev["applicable"]
        cand = InterleavingCandidate(
            candidate_type="pseudo_random",
            depth=seed,
            applicable=applicable,
            score=0.0,
            status=UNKNOWN,
            evidence=ev,
        )
        candidates.append(cand)

    if not candidates:
        result.warnings.append("no candidate could be constructed")
        return result

    # Deterministic scoring. Timing never enters the candidate ranking.
    # Frame-structure evidence supplied by the caller is folded into the
    # score by score_candidate; applicability alone is never sufficient.
    scored: list[tuple[float, InterleavingCandidate]] = []
    for cand in candidates:
        if cand.candidate_type == "none":
            score = 0.0
        else:
            score = float(cand.evidence.get("score", 0.0))

        # If a frame validator was supplied and this depth is applicable,
        # run it against the candidate depth so the structured evidence is
        # genuinely depth-discriminative: only the depth the validator was
        # built to recognize earns the evidence bonus. The original input is
        # never mutated; deinterleave_bits is applied on a per-candidate
        # basis (apply-only, never mutate).
        if cand.candidate_type == "block" and cand.applicable and frame_validator is not None:
            try:
                evidence = frame_validator(bits, depth=cand.depth)
                if isinstance(evidence, dict):
                    score += score_candidate(
                        cand, evidence=evidence
                    )["score"] - score
            except Exception:
                pass

        scored.append((score, cand))

    # Deterministic ordering: higher score first, then depth for tie-break.
    scored.sort(key=lambda item: (item[0], -item[1].depth if item[1].depth is not None else 0), reverse=True)

    # Identify the strongest block candidate, if any, and see if it clears
    # the evidence threshold and beats the no-interleaving baseline by a
    # meaningful margin.
    block_scores = [(s, c) for s, c in scored if c.candidate_type == "block"]
    best_block = None
    best_block_score = 0.0
    if block_scores:
        best_block = block_scores[0][1]
        best_block_score = block_scores[0][0]

    baseline_score = 0.0
    baseline_cand = next((c for c in candidates if c.candidate_type == "none"), None)
    if baseline_cand is not None:
        baseline_score = float(baseline_cand.evidence.get("score", 0.0))

    # Decision rule:
    #  - depth must be structurally applicable,
    #  - must reach the evidence threshold,
    #  - must beat the unmodified-stream baseline by a meaningful margin,
    #  - must beat the second-best block hypothesis by a real margin.
    if best_block is not None and best_block.applicable:
        status = UNKNOWN
        confidence = 0.0
        second_block = None
        for s, c in scored:
            if c.candidate_type == "block" and c.depth != best_block.depth:
                second_block = c
                break
        margin_over_second = 0.0
        if second_block is not None:
            margin_over_second = best_block_score - float(second_block.evidence.get("score", 0.0))
        if best_block_score >= MIN_CONFIDENCE and (best_block_score - baseline_score) >= MARGIN and margin_over_second >= MARGIN:
            status = AUTO_DETECTED
            confidence = min(1.0, max(0.0, (best_block_score - baseline_score) / 100.0))
        elif best_block_score > 0 and best_block_score < MIN_CONFIDENCE:
            # Some structure exists but not enough to detect -> unresolved.
            status = UNRESOLVED
            confidence = 0.5
        elif baseline_score >= MIN_CONFIDENCE:
            # The null baseline matches the data best -> no interleaving.
            status = NO_INTERLEAVING
            confidence = 0.3
        else:
            # Structure exists but ambiguous -> unresolved.
            status = UNRESOLVED
            confidence = 0.5

        result.status = status
        # best_depth is None unless a depth was actually detected; a tie at
        # score 0 must never surface a default depth as an inference.
        result.best_type = best_block.candidate_type if status == AUTO_DETECTED else None
        result.best_depth = best_block.depth if status == AUTO_DETECTED else None
        result.confidence = confidence
        result.interleaving_detected = status == AUTO_DETECTED
        result.candidates = [c.to_dict() for c in candidates]
        result.evidence = {
            "candidates_scored": len(candidates),
            "baseline_score": baseline_score,
            "best_block_score": float(best_block_score) if best_block is not None else 0.0,
            "second_best_block_score": (
                float(second_block.evidence.get("score", 0.0)) if second_block is not None else 0.0
            ),
            "evidence_rule": "deterministic structural evaluation; timing inert",
        }
        result.warnings = [w for w in result.warnings if w not in ("interleaving identification did not reach a decision",)]
    else:
        result.status = UNKNOWN
        result.best_type = None
        result.best_depth = None
        result.confidence = 0.0
        result.interleaving_detected = False
        result.candidates = [c.to_dict() for c in candidates]
        result.evidence = {
            "candidates_scored": len(candidates),
            "baseline_score": baseline_score,
            "best_block_score": 0.0,
            "evidence_rule": "no structurally applicable block depth",
        }

    result.validation_before = result.validation_before
    result.validation_after = result.validation_after
    return result


def infer_block_parameters(
    bits: np.ndarray,
    min_depth: int = 2,
    max_depth: int = 16,
    *,
    frame_validator: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Convenience: return the inferred depth only when AUTO_DETECTED.

    This is a thin wrapper over :func:`identify_interleaving` intended for
    pipeline consumers that only need the single best depth. It must never
    raise; weak evidence returns None.
    """
    result = identify_interleaving(
        bits,
        min_depth=min_depth,
        max_depth=max_depth,
        frame_validator=frame_validator,
    )
    if result.status == AUTO_DETECTED and result.best_depth is not None:
        return {
            "type": "block",
            "depth": result.best_depth,
            "confidence": result.confidence,
            "status": result.status,
        }
    return None


def describe_status(status: str) -> str:
    """Human-readable label for a status."""
    return {
        AUTO_DETECTED: "block interleaving detected",
        NO_INTERLEAVING: "no block interleaving detected",
        UNKNOWN: "cannot determine whether block interleaving is present",
        UNRESOLVED: "block interleaving present but depth ambiguous",
        FAILED: "block interleaving identification failed",
    }.get(status, status)
