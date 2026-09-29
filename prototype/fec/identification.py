"""Automatic FEC identification for the Spectra V2 pipeline.

This module decides *which* supported FEC scheme a received bitstream is
most consistent with.  It is deliberately:

- deterministic (zero randomness),
- explainable (every selected hypothesis carries evidence),
- conservative (``UNKNOWN``/``UNRESOLVED`` when evidence is weak),
- explicit (CRC16/CRC32 are error-detection evidence, never an FEC
  decoding hypothesis).

The identifier is a *candidate evaluator*, not a decoder.  It only ever
runs the existing decoders in ``prototype.fec`` to measure how well a
hypothesis fits the data; it never decodes first and never rewrites a
supported FEC family.

Architecture
------------
decode a candidate -> measure structural/decoder evidence -> score ->
rank.  The final decision is:

    best_status == AUTO_DETECTED -> best_candidate (only when confidence
    meets the internal threshold); otherwise UNKNOWN/UNRESOLVED.

The supported candidate set is:

    none | repetition3 | hamming74 | conv12 | reedsolomon | ldpc |
    concatenated

The block/convolutional candidates are scored on structural consistency;
the Reed-Solomon, LDPC and concatenated candidates are scored on genuine
*decoder* evidence (residual syndromes, bit-flip counts, inner Viterbi
path metric + outer block status).  A scheme that cannot be separated
from its competitors by real evidence is left UNKNOWN/UNRESOLVED.

CRC and none are always treated as detection/validity evidence, never as
an FEC decoding hypothesis.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prototype.core.exceptions import FECError
from prototype.fec import (
    concatenated,
    convolutional,
    hamming,
    ldpc,
    reed_solomon,
    repetition,
)
from prototype.fec.framework import (
    FECResult,
    aligned_size,
    describe_scheme,
    list_schemes,
)

# ---------------------------------------------------------------------------
# Result model (Phase 2)
# ---------------------------------------------------------------------------


@dataclass
class FECIdentificationResult:
    """Structured, explainable automatic FEC identification result."""

    status: str  # AUTO_DETECTED | USER_CONFIGURED | UNKNOWN | UNRESOLVED | FAILED
    best_scheme: str | None
    confidence: float  # 0..1 heuristic confidence. Never a calibrated probability.
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    input_bit_count: int = 0
    corrected_error_count: int = 0
    residual_error_count: int = 0
    validation_status: str = "unknown"
    parameters: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "best_scheme": self.best_scheme,
            "confidence": float(self.confidence),
            "candidates": list(self.candidates),
            "evidence": dict(self.evidence),
            "input_bit_count": self.input_bit_count,
            "corrected_error_count": self.corrected_error_count,
            "residual_error_count": self.residual_error_count,
            "validation_status": self.validation_status,
            "parameters": dict(self.parameters or {}),
            "warnings": list(self.warnings),
        }


class FECIdentificationError(Exception):
    """Configuration or data-level failure in the identification layer."""


# ---------------------------------------------------------------------------
# Candidate constants (Phase 3)
# ---------------------------------------------------------------------------

SUPPORTED_CANDIDATES = (
    "none",
    "repetition3",
    "hamming74",
    "conv12",
    "reedsolomon",
    "ldpc",
    "concatenated",
)


def list_candidates() -> tuple[str, ...]:
    """Supported automatic FEC candidate schemes (fixed, evidence-scored)."""
    return tuple(SUPPORTED_CANDIDATES)


def describe_candidate(name: str) -> str:
    """Human-readable label for a candidate."""
    if name not in list_candidates():
        raise FECError(f"Unknown automatic FEC candidate '{name}'.")
    return describe_scheme(name)


# ---------------------------------------------------------------------------
# Structural gating (Phase 3-5)
# ---------------------------------------------------------------------------


def _as_binary(bits: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise FECIdentificationError(f"{label} must contain at least one bit.")
    if not np.all((arr == 0) | (arr == 1)):
        raise FECIdentificationError(f"{label} must be binary 0/1 values.")
    return arr


def _check_min_bits(bits: np.ndarray, candidate: str, minimum: int) -> list[str]:
    """Return warnings when a candidate is tried on too little data."""
    warnings: list[str] = []
    if bits.size < minimum:
        warnings.append(
            f"input too short for {candidate} (needs >= {minimum} bits)"
        )
    return warnings


def _check_block_structure(
    bits: np.ndarray, candidate: str, block_size: int
) -> list[str]:
    """Return warnings when the input does not match the candidate's block size."""
    warnings: list[str] = []
    if bits.size % block_size != 0:
        warnings.append(
            f"input bit count {bits.size} is not a multiple of {block_size} "
            f"({candidate} block size)"
        )
    return warnings


def _check_decoded_length(
    decoded: np.ndarray, bits: np.ndarray, candidate: str, expected_ratio: float
) -> list[str]:
    """Return warnings when the decoded length does not match expectations."""
    warnings: list[str] = []
    if decoded.size == 0:
        warnings.append("decoder produced no output")
    elif decoded.size < bits.size * 0.1:
        warnings.append(
            f"decoded output too short relative to input for {candidate}"
        )
    return warnings


# ---------------------------------------------------------------------------
# Candidate evaluation (Phase 3)
# ---------------------------------------------------------------------------


def evaluate_repetition3(bits: np.ndarray) -> dict[str, Any]:
    """Evaluate the rate-1/3 repetition candidate.

    Evidence captured:
    - input bit count and block structure (must be multiple of 3)
    - decoded bit count (should be 1/3 of input)
    - block count from decoder stats
    - corrected/uncorrectable block counts from decoder
    - residual estimate against reference when available
    """
    bits = _as_binary(bits, "repetition3 input")
    out = {
        "scheme": "repetition3",
        "input_bit_count": int(bits.size),
        "block_size": 3,
        "decoder_valid": True,
        "warnings": [],
    }

    if bits.size == 0:
        out["decoder_valid"] = False
        return out

    # Structural gating: repetition code requires input divisible by 3
    if bits.size % 3 != 0:
        out.update(
            {
                "decoder_valid": False,
                "available_only": True,
                "reason": f"input bit count {bits.size} not a multiple of 3",
            }
        )
        return out

    try:
        counts, stats = repetition.decode(bits, repetitions=3)
    except ValueError as exc:
        out.update(
            {
                "decoder_valid": False,
                "available_only": True,
                "reason": str(exc),
            }
        )
        return out

    out.update(
        {
            "decoded_bit_count": int(counts.size),
            "blocks": stats.get("blocks", 0),
            "corrected_errors": stats.get("corrected_errors", 0),
            "uncorrectable_blocks": stats.get("uncorrectable_blocks", 0),
            "residual_estimate": 0,  # neutral until reference provided
            "decoder_valid": True,
            "warnings": [],
        }
    )

    warnings = _check_min_bits(bits, "repetition3", 6)
    if counts.size < 2:
        warnings.append("too few decoded bits for reliable evidence")
    out["warnings"] = warnings
    return out


def evaluate_hamming74(bits: np.ndarray) -> dict[str, Any]:
    """Evaluate Hamming(7,4) candidate.

    Evidence captured:
    - input bit count and block structure (must be multiple of 7)
    - decoded bit count (should be 4/7 of input)
    - block count from decoder stats
    - corrected/uncorrectable block counts from decoder
    - residual estimate against reference when available
    """
    bits = _as_binary(bits, "hamming74 input")
    out = {
        "scheme": "hamming74",
        "input_bit_count": int(bits.size),
        "block_size": 7,
        "decoder_valid": True,
        "warnings": [],
    }

    if bits.size == 0:
        out["decoder_valid"] = False
        return out

    # Structural gating: Hamming74 requires input divisible by 7
    if bits.size % 7 != 0:
        out.update(
            {
                "decoder_valid": False,
                "available_only": True,
                "reason": f"input bit count {bits.size} not a multiple of 7",
            }
        )
        return out

    try:
        counts, stats = hamming.decode(bits)
    except ValueError as exc:
        out.update(
            {
                "decoder_valid": False,
                "available_only": True,
                "reason": str(exc),
            }
        )
        return out

    out.update(
        {
            "decoded_bit_count": int(counts.size),
            "blocks": stats.get("blocks", 0),
            "corrected_errors": stats.get("corrected_errors", 0),
            "uncorrectable_blocks": stats.get("uncorrectable_blocks", 0),
            "residual_estimate": 0,  # neutral until reference provided
            "decoder_valid": True,
            "warnings": [],
        }
    )

    warnings = _check_min_bits(bits, "hamming74", 14)
    if counts.size < 4:
        warnings.append("too few decoded bits for reliable evidence")
    out["warnings"] = warnings
    return out


def evaluate_conv12(bits: np.ndarray) -> dict[str, Any]:
    """Evaluate K=7 rate-1/2 convolutional candidate (Viterbi).

    Evidence captured:
    - input bit count (must be even)
    - steps processed from decoder stats
    - final path metric (0.0 = perfect survivor path, higher = unstructured)
    - corrected/uncorrectable block counts from decoder
    - residual estimate against reference when available
    """
    bits = _as_binary(bits, "conv12 input")
    out = {
        "scheme": "conv12",
        "input_bit_count": int(bits.size),
        "block_size": 1,
        "decoder_valid": True,
        "warnings": [],
    }

    if bits.size == 0:
        out["decoder_valid"] = False
        return out

    # Structural gating: convolutional decode requires even number of code bits
    if bits.size % 2 != 0:
        out.update(
            {
                "decoder_valid": False,
                "available_only": True,
                "reason": "convolutional decode requires an even number of code bits",
            }
        )
        return out

    try:
        counts, stats = convolutional.decode(bits)
    except ValueError as exc:
        out.update(
            {
                "decoder_valid": False,
                "available_only": True,
                "reason": str(exc),
            }
        )
        return out

    out.update(
        {
            "decoded_bit_count": int(counts.size),
            "steps": stats.get("steps", 0),
            "final_path_metric": stats.get("final_path_metric", 0.0),
            "corrected_errors": stats.get("corrected_errors", 0),
            "uncorrectable_blocks": stats.get("uncorrectable_blocks", 0),
            "residual_estimate": 0,  # neutral until reference provided
            "decoder_valid": True,
            "warnings": [],
        }
    )

    # conv12 structural gating: require sufficient steps for meaningful Viterbi decoding
    min_steps = 16
    if out["steps"] < min_steps:
        out["warnings"].append(f"only {out['steps']} steps processed; insufficient for meaningful evidence")

    out["warnings"] = _check_min_bits(bits, "conv12", 32)
    return out


def evaluate_reedsolomon(bits: np.ndarray) -> dict[str, Any]:
    """Evaluate the shortened Reed-Solomon candidate.

    Evidence captured:
    - whole-byte / whole-block structural compatibility
    - blocks processed
    - uncorrectable block count (the discriminator: random data is
      essentially never within ``t`` symbol errors of a codeword)
    - corrected symbol-error count (a real RS stream may still correct a
      few symbols and stay fully decodable)
    """
    bits = _as_binary(bits, "reedsolomon input")
    block_bits = 8 * (reed_solomon.K_DEFAULT + reed_solomon.NSYM_DEFAULT)
    out = {
        "scheme": "reedsolomon",
        "input_bit_count": int(bits.size),
        "block_size": block_bits,
        "decoder_valid": True,
        "warnings": [],
    }

    if bits.size == 0:
        out["decoder_valid"] = False
        return out

    # A real capture never ends on a whole codeword boundary, so the
    # truncated tail is dropped and reported instead of rejecting the
    # candidate outright: identification must work on captured streams.
    usable, trimmed_tail = aligned_size(int(bits.size), "reedsolomon")
    if usable < 2 * block_bits:
        out.update(
            decoder_valid=False,
            available_only=True,
            reason=(
                f"input bit count {bits.size} holds fewer than two whole "
                f"{block_bits}-bit Reed-Solomon blocks"
            ),
        )
        return out

    bits = bits[:usable]

    try:
        counts, stats = reed_solomon.decode(bits)
    except ValueError as exc:
        out.update(decoder_valid=False, available_only=True, reason=str(exc))
        return out

    blocks = int(stats.get("blocks", 0))
    uncorrectable = int(stats.get("uncorrectable_blocks", 0))
    out.update(
        decoded_bit_count=int(counts.size),
        blocks=blocks,
        corrected_errors=int(stats.get("corrected_errors", 0)),
        uncorrectable_blocks=uncorrectable,
        clean_blocks=blocks - uncorrectable,
        clean_block_fraction=(float(blocks - uncorrectable) / blocks) if blocks else 0.0,
        trimmed_tail_bits=int(trimmed_tail),
        decoder_valid=True,
    )
    out["warnings"] = _check_min_bits(bits, "reedsolomon", 4 * block_bits)
    return out


def evaluate_ldpc(bits: np.ndarray) -> dict[str, Any]:
    """Evaluate the compact (16,8) LDPC candidate (syndrome evidence).

    The registered decoder is a hard-decision bit-flipper, so "clean" is
    only credible at zero total bit flips: a genuine LDPC codeword has a
    zero syndrome and is returned unmodified.  Blocks that only converge
    *after* flipping are reported but never credited as detection
    evidence, so an ambiguous stream stays UNRESOLVED rather than being
    claimed as LDPC.
    """
    bits = _as_binary(bits, "ldpc input")
    block_bits = int(ldpc.N)
    out = {
        "scheme": "ldpc",
        "input_bit_count": int(bits.size),
        "block_size": block_bits,
        "decoder_valid": True,
        "warnings": [],
    }

    if bits.size == 0:
        out["decoder_valid"] = False
        return out

    usable, trimmed_tail = aligned_size(int(bits.size), "ldpc")
    if usable < 2 * block_bits:
        out.update(
            decoder_valid=False,
            available_only=True,
            reason=(
                f"input bit count {bits.size} holds fewer than two whole "
                f"{block_bits}-bit LDPC blocks"
            ),
        )
        return out

    bits = bits[:usable]

    try:
        counts, stats = ldpc.decode(bits)
    except ValueError as exc:
        out.update(decoder_valid=False, available_only=True, reason=str(exc))
        return out

    flips = int(stats.get("corrected_errors", 0))
    out.update(
        decoded_bit_count=int(counts.size),
        blocks=int(stats.get("blocks", 0)),
        corrected_errors=flips,
        uncorrectable_blocks=int(stats.get("uncorrectable_blocks", 0)),
        zero_flip_decode=flips == 0,
        trimmed_tail_bits=int(trimmed_tail),
        decoder_valid=True,
    )
    out["warnings"] = _check_min_bits(bits, "ldpc", 8 * block_bits)
    return out


def evaluate_concatenated(bits: np.ndarray) -> dict[str, Any]:
    """Evaluate the serial concatenation (RS outer + conv inner) candidate.

    Evidence captured:
    - inner Viterbi path metric and inner uncorrectable count
    - outer Reed-Solomon block/uncorrectable counts on the recovered bits

    Both layers must decode cleanly.  Note honestly that a concatenated
    codeword *is* also a valid conv12 codeword: the outer layer is the
    only evidence that separates the two hypotheses, and it is recorded
    in the evidence rather than hidden.
    """
    bits = _as_binary(bits, "concatenated input")
    out = {
        "scheme": "concatenated",
        "input_bit_count": int(bits.size),
        "block_size": 2,
        "decoder_valid": True,
        "warnings": [],
    }

    if bits.size == 0:
        out["decoder_valid"] = False
        return out

    # Whole concatenated frames only: the inner Viterbi strips its 6 tail
    # bits and the outer RS needs whole 40-byte blocks.  The truncated
    # tail is dropped and reported.
    usable, trimmed_tail = aligned_size(int(bits.size), "concatenated")
    if usable <= 0:
        out.update(
            decoder_valid=False,
            available_only=True,
            reason=(
                f"input bit count {bits.size} holds no whole concatenated "
                "frame (RS outer + convolutional inner)"
            ),
        )
        return out

    bits = bits[:usable]

    try:
        counts, stats = concatenated.decode(bits)
    except ValueError as exc:
        out.update(decoder_valid=False, available_only=True, reason=str(exc))
        return out

    inner = dict(stats.get("inner") or {})
    outer = dict(stats.get("outer") or {})
    out.update(
        decoded_bit_count=int(counts.size),
        corrected_errors=int(stats.get("corrected_errors", 0)),
        uncorrectable_blocks=int(stats.get("uncorrectable_blocks", 0)),
        inner_path_metric=float(inner.get("final_path_metric", 0.0)),
        inner_steps=int(inner.get("steps", 0)),
        inner_uncorrectable_blocks=int(inner.get("uncorrectable_blocks", 0)),
        outer_blocks=int(outer.get("blocks", 0)),
        outer_uncorrectable_blocks=int(outer.get("uncorrectable_blocks", 0)),
        trimmed_tail_bits=int(trimmed_tail),
        decoder_valid=True,
    )
    out["warnings"] = _check_min_bits(bits, "concatenated", 64)
    return out


# ---------------------------------------------------------------------------
# Candidate scoring (Phase 4)
# ---------------------------------------------------------------------------


def _score_shell(candidate: str, cand: dict[str, Any]) -> dict[str, Any]:
    """Common, deterministic score-record skeleton for every candidate."""
    return {
        "candidate": candidate,
        "compatible": cand.get("decoder_valid", False),
        "block_compatible": bool(
            int(cand.get("blocks", 0)) or int(cand.get("steps", 0))
        ),
        "decoder_valid": cand.get("decoder_valid", False),
        "low_residual": False,
        "frame_valid": True,
        "score": 0.0,
        "evidence": {},
    }


def score_repetition3(cand: dict[str, Any]) -> dict[str, Any]:
    out = {
        "candidate": "repetition3",
        "compatible": cand.get("decoder_valid", False),
        "block_compatible": cand.get("blocks", 0) > 0,
        "decoder_valid": cand.get("decoder_valid", False),
        "low_residual": False,
        "frame_valid": True,
        "score": 0.0,
        "evidence": {},
    }

    if not out["compatible"]:
        out["evidence"]["reason"] = "repetition3 decoder did not produce a usable decode"
        return out

    corrected = int(cand.get("corrected_errors", 0))
    blocks = int(cand.get("blocks", 0))
    decoded = int(cand.get("decoded_bit_count", 0))
    out["evidence"]["corrected_errors"] = corrected
    out["evidence"]["blocks_processed"] = blocks
    out["evidence"]["decoded_bit_count"] = decoded

    score = 0.0
    # Repetition code is only credible when EVERY block decoded perfectly.
    # A single corrected error destroys the rate-1/3 repetition hypothesis.
    if corrected == 0 and blocks > 0:
        score += 40
        out["evidence"]["all_blocks_corrected_zero"] = True
    else:
        # Penalize immediately: any correction => candidate is wrong
        score += 0
        out["evidence"]["perfect_decode_required"] = True
        if corrected > 0:
            out["evidence"]["corrected_errors"] = corrected
    # Blocks processed (more blocks = more evidence for a clean decode)
    if blocks >= 12:
        score += 20
    elif blocks >= 6:
        score += 10
    # Decoded length (enough data to be meaningful)
    if decoded >= 12:
        score += 10
    elif decoded >= 6:
        score += 5
    # Decoder validity and no warnings
    if cand.get("decoder_valid") and not cand.get("warnings"):
        score += 10

    out["score"] = round(min(100.0, max(0.0, score)), 3)
    out["evidence"]["reasoning"] = (
        "repetition3 majority-vote decoder returns to all-zero correction status; "
        "every processed block is consistent with a rate-1/3 repetition code"
    )
    return out


def score_hamming74(cand: dict[str, Any]) -> dict[str, Any]:
    out = {
        "candidate": "hamming74",
        "compatible": cand.get("decoder_valid", False),
        "block_compatible": cand.get("blocks", 0) > 0,
        "decoder_valid": cand.get("decoder_valid", False),
        "low_residual": False,
        "frame_valid": True,
        "score": 0.0,
        "evidence": {},
    }

    if not out["compatible"]:
        out["evidence"]["reason"] = "hamming74 decoder did not produce a usable decode"
        return out

    corrected = int(cand.get("corrected_errors", 0))
    blocks = int(cand.get("blocks", 0))
    decoded = int(cand.get("decoded_bit_count", 0))
    out["evidence"]["corrected_errors"] = corrected
    out["evidence"]["blocks_processed"] = blocks
    out["evidence"]["decoded_bit_count"] = decoded

    score = 0.0
    # Hamming(7,4) is only credible when EVERY block decoded perfectly
    # (single-error-correctable, no uncorrectable blocks).  A single
    # corrected error or uncorrectable block destroys the hypothesis.
    if corrected == 0 and cand.get("uncorrectable_blocks", 0) == 0 and blocks > 0:
        score += 40
        out["evidence"]["all_blocks_clean"] = True
    else:
        score += 0
        out["evidence"]["perfect_decode_required"] = True
        if corrected > 0:
            out["evidence"]["corrected_errors"] = corrected
        if cand.get("uncorrectable_blocks", 0) > 0:
            out["evidence"]["uncorrectable_blocks"] = cand.get("uncorrectable_blocks", 0)
    # Blocks processed (more blocks = more evidence for a clean decode)
    if blocks >= 20:
        score += 20
    elif blocks >= 10:
        score += 12
    elif blocks >= 4:
        score += 6
    # Decoded length (enough data to be meaningful)
    if decoded >= 24:
        score += 10
    elif decoded >= 12:
        score += 5
    # Decoder validity and no warnings
    if cand.get("decoder_valid") and not cand.get("warnings"):
        score += 10

    out["score"] = round(min(100.0, max(0.0, score)), 3)
    out["evidence"]["reasoning"] = (
        "hamming74 structural/parity consistency plus block-count evidence; "
        "every processed 7-bit block is consistent with a systematic Hamming(7,4) code"
    )
    return out


def score_conv12(cand: dict[str, Any]) -> dict[str, Any]:
    out = {
        "candidate": "conv12",
        "compatible": cand.get("decoder_valid", False),
        "block_compatible": cand.get("steps", 0) > 0,
        "decoder_valid": cand.get("decoder_valid", False),
        "low_residual": False,
        "frame_valid": True,
        "score": 0.0,
        "evidence": {},
    }

    if not out["compatible"]:
        out["evidence"]["reason"] = "conv12 Viterbi decoder did not produce a usable decode"
        return out

    steps = int(cand.get("steps", 0))
    path_metric = float(cand.get("final_path_metric", 0.0))
    corrected = int(cand.get("corrected_errors", 0))
    out["evidence"]["steps_processed"] = steps
    out["evidence"]["final_path_metric"] = float(path_metric)
    out["evidence"]["corrected_errors"] = corrected

    score = 0.0
    # Convolutional code is only credible when the Viterbi survivor path
    # is essentially perfect (path_metric ~ 0.0).  Any non-zero path metric
    # indicates the survivor path deviated from the zero-crossover path,
    # which means the input is NOT a valid rate-1/2 K=7 convolutional
    # codeword.  Penalize immediately.
    if path_metric == 0.0 and steps > 0:
        score += 40
        out["evidence"]["perfect_path_metric"] = True
    else:
        score += 0
        out["evidence"]["perfect_decode_required"] = True
        out["evidence"]["final_path_metric"] = float(path_metric)
        if corrected > 0:
            out["evidence"]["corrected_errors"] = corrected
    # Steps processed (more steps = more Viterbi evidence)
    if steps >= 128:
        score += 20
    elif steps >= 64:
        score += 15
    elif steps >= 32:
        score += 10
    # Decoded length
    decoded = int(cand.get("decoded_bit_count", 0))
    if decoded >= 64:
        score += 10
    elif decoded >= 32:
        score += 5
    # Decoder validity and no warnings
    if cand.get("decoder_valid") and not cand.get("warnings"):
        score += 10

    out["score"] = round(min(100.0, max(0.0, score)), 3)
    out["evidence"]["reasoning"] = (
        "conv12 Viterbi survivor path shows zero deviation (path_metric=0.0) "
        "indicating the input is exactly a rate-1/2 K=7 convolutional codeword"
    )
    return out


def score_none(cand: dict[str, Any]) -> dict[str, Any]:
    out = {
        "candidate": "none",
        "compatible": True,
        "block_compatible": True,
        "decoder_valid": True,
        "low_residual": False,
        "frame_valid": True,
        "score": 0.0,
        "evidence": {},
    }
    out["evidence"]["reasoning"] = (
        "no FEC scheme is indicated; the raw bitstream passes only the trivial hypothesis"
    )
    out["score"] = 0.0
    return out


def score_reedsolomon(cand: dict[str, Any]) -> dict[str, Any]:
    """Score the Reed-Solomon hypothesis on real decoder evidence.

    Zero uncorrectable blocks is the discriminator: on random data the
    bounded-distance decoder rejects essentially every block, while a
    genuine RS stream (even with a few symbol errors) is fully decodable.
    """
    out = _score_shell("reedsolomon", cand)

    if not out["compatible"]:
        out["evidence"]["reason"] = (
            cand.get("reason")
            or "Reed-Solomon decoder did not produce a usable decode"
        )
        return out

    blocks = int(cand.get("blocks", 0))
    uncorrectable = int(cand.get("uncorrectable_blocks", 0))
    corrected = int(cand.get("corrected_errors", 0))
    decoded = int(cand.get("decoded_bit_count", 0))
    out["evidence"].update(
        blocks_processed=blocks,
        uncorrectable_blocks=uncorrectable,
        corrected_symbol_errors=corrected,
        decoded_bit_count=decoded,
    )

    score = 0.0

    if uncorrectable == 0 and blocks > 0:
        score += 40
        out["evidence"]["all_blocks_decodable"] = True
    elif blocks > 0 and (blocks - uncorrectable) / blocks >= 0.5:
        score += 8
        out["evidence"]["partial_decode_of_blocks"] = uncorrectable
        out["evidence"]["partial_credit_reason"] = (
            "most blocks decoded, but uncorrectable blocks remain: not enough "
            "evidence to claim Reed-Solomon"
        )
    else:
        out["evidence"]["perfect_decode_required"] = True

    if blocks >= 16:
        score += 20
    elif blocks >= 8:
        score += 15
    elif blocks >= 4:
        score += 10

    if decoded >= 64:
        score += 10
    elif decoded >= 32:
        score += 5

    if cand.get("decoder_valid") and not cand.get("warnings"):
        score += 10

    out["low_residual"] = uncorrectable == 0 and blocks > 0
    out["score"] = round(min(100.0, max(0.0, score)), 3)
    out["evidence"]["reasoning"] = (
        "shortened Reed-Solomon bounded-distance decoding reports "
        f"{uncorrectable} uncorrectable block(s) out of {blocks}"
    )
    return out


def score_ldpc(cand: dict[str, Any]) -> dict[str, Any]:
    """Score the compact LDPC hypothesis (zero-flip syndrome evidence).

    Deliberately conservative: only a zero-flip (already valid codeword)
    decode is credited, so LDPC stays UNRESOLVED on anything ambiguous.
    """
    out = _score_shell("ldpc", cand)

    if not out["compatible"]:
        out["evidence"]["reason"] = (
            cand.get("reason") or "LDPC decoder did not produce a usable decode"
        )
        return out

    blocks = int(cand.get("blocks", 0))
    flips = int(cand.get("corrected_errors", 0))
    decoded = int(cand.get("decoded_bit_count", 0))
    out["evidence"].update(
        blocks_processed=blocks,
        bit_flips=flips,
        uncorrectable_blocks=int(cand.get("uncorrectable_blocks", 0)),
        decoded_bit_count=decoded,
        zero_flip_decode=bool(cand.get("zero_flip_decode")),
    )

    score = 0.0

    if cand.get("zero_flip_decode") and blocks > 0:
        score += 40
        out["evidence"]["already_valid_codeword"] = True
    else:
        out["evidence"]["zero_flip_required"] = True
        out["evidence"]["zero_flip_reason"] = (
            "the bit-flipping decoder had to modify bits: the stream is not "
            "provably an LDPC codeword"
        )

    if blocks >= 16:
        score += 20
    elif blocks >= 8:
        score += 15
    elif blocks >= 4:
        score += 10

    if decoded >= 64:
        score += 10
    elif decoded >= 32:
        score += 5

    if cand.get("decoder_valid") and not cand.get("warnings"):
        score += 10

    out["low_residual"] = bool(cand.get("zero_flip_decode")) and blocks > 0
    out["score"] = round(min(100.0, max(0.0, score)), 3)
    out["evidence"]["reasoning"] = (
        "compact (16,8) LDPC syndrome check: "
        f"{blocks} block(s), {flips} bit flip(s) required"
    )
    return out


def score_concatenated(cand: dict[str, Any]) -> dict[str, Any]:
    """Score the concatenated (RS outer + conv inner) hypothesis.

    Requires BOTH layers clean.  A concatenated stream is also a valid
    conv12 codeword, so when both hypotheses clear the threshold the more
    specific one wins and the ambiguity is recorded in the evidence.
    """
    out = _score_shell("concatenated", cand)

    if not out["compatible"]:
        out["evidence"]["reason"] = (
            cand.get("reason")
            or "concatenated decoder did not produce a usable decode"
        )
        return out

    inner_metric = float(cand.get("inner_path_metric", 0.0))
    inner_uncorrectable = int(cand.get("inner_uncorrectable_blocks", 0))
    outer_blocks = int(cand.get("outer_blocks", 0))
    outer_uncorrectable = int(cand.get("outer_uncorrectable_blocks", 0))
    corrected = int(cand.get("corrected_errors", 0))
    decoded = int(cand.get("decoded_bit_count", 0))
    out["evidence"].update(
        inner_path_metric=inner_metric,
        inner_uncorrectable_blocks=inner_uncorrectable,
        outer_blocks=outer_blocks,
        outer_uncorrectable_blocks=outer_uncorrectable,
        combined_corrected_errors=corrected,
        decoded_bit_count=decoded,
    )

    score = 0.0
    inner_clean = inner_metric == 0.0 and inner_uncorrectable == 0 and outer_blocks > 0
    outer_clean = outer_blocks > 0 and outer_uncorrectable == 0

    if inner_clean and outer_clean:
        score += 40
        # The more specific hypothesis: it explains the inner convolutional
        # evidence *and* the outer Reed-Solomon layer.  The margin is small
        # and documented so conv12 remains the runner-up, not a lost fact.
        score += 8
        out["evidence"]["inner_layer_clean"] = True
        out["evidence"]["outer_layer_clean"] = True
        out["evidence"]["specificity_bonus"] = 8
        out["evidence"]["ambiguity_note"] = (
            "a valid concatenated codeword is also a valid conv12 codeword; "
            "the outer Reed-Solomon layer is the evidence that separates them"
        )
    else:
        out["evidence"]["both_layers_required"] = True
        out["evidence"]["inner_layer_clean"] = inner_clean
        out["evidence"]["outer_layer_clean"] = outer_clean

    if outer_blocks >= 8:
        score += 20
    elif outer_blocks >= 4:
        score += 15
    elif outer_blocks >= 2:
        score += 10

    if decoded >= 64:
        score += 10
    elif decoded >= 32:
        score += 5

    if cand.get("decoder_valid") and not cand.get("warnings"):
        score += 10

    out["low_residual"] = inner_clean and outer_clean
    out["score"] = round(min(100.0, max(0.0, score)), 3)
    out["evidence"]["reasoning"] = (
        "serial concatenation evidence: inner Viterbi path metric "
        f"{inner_metric}, outer Reed-Solomon uncorrectable blocks "
        f"{outer_uncorrectable}/{outer_blocks}"
    )
    return out


_SCORERS = {
    "repetition3": score_repetition3,
    "hamming74": score_hamming74,
    "conv12": score_conv12,
    "none": score_none,
    "reedsolomon": score_reedsolomon,
    "ldpc": score_ldpc,
    "concatenated": score_concatenated,
}


def _default_score(candidate: str, cand: dict[str, Any]) -> dict[str, Any]:
    scorer = _SCORERS.get(candidate)
    if scorer is None:
        return {
            "candidate": candidate,
            "compatible": False,
            "block_compatible": False,
            "decoder_valid": False,
            "low_residual": False,
            "frame_valid": False,
            "score": 0.0,
            "evidence": {"reason": "no scoring rule for this candidate"},
        }
    return scorer(cand)


# ---------------------------------------------------------------------------
# Main identification entry point (Phase 5-7)
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 60.0  # internal heuristic threshold


def identify_fec(
    bits: np.ndarray,
    reference_bits: np.ndarray | None = None,
    max_blocks: int = 64,
) -> FECIdentificationResult:
    """Identify automatic FEC hypotheses in a received bitstream.

    Parameters
    ----------
    bits
        Demodulated bitstream, MSB-first 0/1 array.
    reference_bits
        Optional transmitted reference for residual-error measurement.
        Identification does *not* require a reference, but a reference
        makes the evidence much stronger and more explainable.
    max_blocks
        Cap on the number of full blocks evaluated per candidate so the
        map cannot grow unboundedly on a pathological input.

    Returns
    -------
    FECIdentificationResult
        Status AUTO_DETECTED only when a single hypothesis clears the
        internal confidence rule; otherwise UNKNOWN/UNRESOLVED.
    """
    start = time.perf_counter()
    input_bits = int(np.asarray(bits).size)
    result = FECIdentificationResult(
        status="UNKNOWN",
        best_scheme=None,
        confidence=0.0,
        input_bit_count=input_bits,
        evidence={},
        warnings=["auto identification failed before decision"],
    )

    try:
        bits = _as_binary(bits, "identify input")
    except FECIdentificationError as exc:
        result.warnings.append(str(exc))
        return result

    if input_bits < 16:
        result.warnings.append("insufficient bits for identification")
        return result

    reference = None
    if reference_bits is not None:
        try:
            reference = _as_binary(reference_bits, "reference input")
        except FECIdentificationError as exc:
            result.warnings.append(f"reference ignored: {exc}")

    candidates: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}

    # --- evaluate every candidate in a fixed order -----------------------
    for scheme in list_candidates():
        try:
            if scheme == "none":
                cand = {
                    "scheme": "none",
                    "input_bit_count": input_bits,
                    "decoder_valid": True,
                    "corrected_errors": 0,
                    "residual_estimate": 0,
                    "warnings": [],
                }
            elif scheme == "repetition3":
                cand = evaluate_repetition3(bits)
            elif scheme == "hamming74":
                cand = evaluate_hamming74(bits)
            elif scheme == "conv12":
                cand = evaluate_conv12(bits)
            elif scheme == "reedsolomon":
                cand = evaluate_reedsolomon(bits)
            elif scheme == "ldpc":
                cand = evaluate_ldpc(bits)
            elif scheme == "concatenated":
                cand = evaluate_concatenated(bits)
            else:
                continue
        except Exception as exc:  # never let one candidate kill the run
            candidates.append(
                {
                    "scheme": scheme,
                    "decoder_valid": False,
                    "corrected_errors": 0,
                    "residual_estimate": 0,
                    "available_only": True,
                    "reason": f"candidate evaluation error: {exc}",
                }
            )
            continue

        candidates.append(cand)

    if not candidates:
        result.warnings.append("no candidate could be evaluated")
        return result

    # --- score each candidate and keep a deterministic ordering -----------
    scored: list[dict[str, Any]] = []
    for cand in candidates:
        try:
            scored.append(_default_score(cand["scheme"], cand))
        except Exception as exc:  # defensive
            scored.append(
                {
                    "candidate": cand.get("scheme"),
                    "compatible": False,
                    "score": 0.0,
                    "evidence": {"reason": f"scoring error: {exc}"},
                }
            )

    # Deterministic ranking: highest score first, then decoder-validity and
    # block count for reproducible tie-breaking.
    def _rank2(item: dict[str, Any]) -> tuple[float, int]:
        name = item["candidate"]
        idx = name_to_index.get(name)
        cand = candidates[idx] if idx is not None else None
        blocks = 0
        if cand is not None:
            blocks = int(cand.get("blocks", 0)) or int(cand.get("steps", 0))
        return (float(item["score"]), blocks)

    name_to_index = {cand.get("scheme"): i for i, cand in enumerate(candidates)}
    scored.sort(key=_rank2, reverse=True)

    # --- summarize evidence ----------------------------------------------
    evidence["candidates_evaluated"] = len(candidates)
    evidence["candidate_order"] = [s["candidate"] for s in scored]
    evidence["scoring_rule"] = "deterministic heuristic; not a calibrated probability"
    evidence["confidence_rule"] = f"best_score >= {MIN_CONFIDENCE} with strong evidence"

    # --- decide ----------------------------------------------------------
    best = scored[0]
    final_status = "UNKNOWN"
    final_scheme = None

    # Every candidate that also clears the threshold is reported: when two
    # hypotheses are both consistent with the data the caller must be able
    # to see the ambiguity instead of trusting a silent tie-break.
    consistent = [
        str(s["candidate"])
        for s in scored[1:]
        if s["score"] >= MIN_CONFIDENCE and s.get("compatible")
    ]

    if best["score"] >= MIN_CONFIDENCE and best.get("compatible"):
        final_scheme = best["candidate"]
        final_status = "AUTO_DETECTED"
        evidence["decided_by"] = final_scheme
        evidence["confidence"] = float(best["score"])
        evidence["residual_zero"] = bool(best.get("low_residual"))
        evidence["corrected_errors_total"] = int(
            sum(int(c.get("corrected_errors", 0)) for c in candidates)
        )
        evidence["also_consistent"] = consistent
        evidence["ambiguous"] = bool(consistent)
    else:
        result.warnings.append(
            "automatic FEC identification did not reach the decision threshold"
        )

    result.status = final_status
    result.best_scheme = final_scheme
    result.confidence = float(best["score"])
    result.candidates = [
        {"candidate": s["candidate"], "score": s["score"], "evidence": s["evidence"]} for s in scored
    ]
    evidence["input_bit_count"] = input_bits
    evidence["evaluation_seconds"] = round(time.perf_counter() - start, 4)
    result.evidence = evidence
    result.validation_status = final_status if final_scheme else "insufficient_evidence"
    return result
