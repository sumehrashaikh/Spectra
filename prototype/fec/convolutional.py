"""
Convolutional coding: constraint-length 7, rate 1/2, polynomials
G1 = 171 (octal), G2 = 133 (octal) - the industry-standard NASA/CCSDS
code - with hard-decision Viterbi decoding and zero tail bits.
"""

from __future__ import annotations

import numpy as np

NUM_STATES = 64  # 2^(K-1)
CONSTRAINT = 7
POLY_G1 = 0o171  # 1111001
POLY_G2 = 0o133  # 1011011
RATE = 2
NUM_TAIL_BITS = CONSTRAINT - 1


def _next_state(state: int, input_bit: int) -> int:
    return ((state >> 1) | (input_bit << 5)) & (NUM_STATES - 1)


def _output_bits(state: int, input_bit: int) -> tuple[int, int]:
    register = (input_bit << 6) | state  # 7 bits, newest first
    g1 = bin(register & POLY_G1).count("1") & 1
    g2 = bin(register & POLY_G2).count("1") & 1
    return g1, g2


def _build_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_states = np.zeros((NUM_STATES, 2), dtype=np.int32)
    outputs = np.zeros((NUM_STATES, 2, 2), dtype=np.int32)
    for state in range(NUM_STATES):
        for bit in (0, 1):
            next_states[state, bit] = _next_state(state, bit)
            outputs[state, bit] = _output_bits(state, bit)
    return next_states, outputs


def _build_reverse_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predecessor-indexed transition tables for the vectorised Viterbi.

    A rate-1/2 K=7 encoder has exactly two incoming transitions per state
    (one per input bit), so the add-compare-select step can be expressed
    as a component-wise minimum over a ``(NUM_STATES, 2)`` array instead
    of a nested Python loop.  Entry ``j`` of each row is filled in
    (previous state ascending, bit 0 then 1) order so tie-breaking is
    identical to the scalar implementation.
    """
    inc_states = np.zeros((NUM_STATES, 2), dtype=np.int32)
    inc_bits = np.zeros((NUM_STATES, 2), dtype=np.uint8)
    inc_outputs = np.zeros((NUM_STATES, 2, 2), dtype=np.uint8)
    slot = np.zeros(NUM_STATES, dtype=np.int32)
    for state in range(NUM_STATES):
        for bit in (0, 1):
            next_state = int(_NEXT_STATES[state, bit])
            position = int(slot[next_state])
            inc_states[next_state, position] = state
            inc_bits[next_state, position] = bit
            inc_outputs[next_state, position] = _OUTPUTS[state, bit]
            slot[next_state] += 1
    return inc_states, inc_bits, inc_outputs


_NEXT_STATES, _OUTPUTS = _build_tables()
_INC_STATES, _INC_BITS, _INC_OUTPUTS = _build_reverse_tables()
_STATE_INDEX = np.arange(NUM_STATES, dtype=np.int32)


def encode(bits: np.ndarray, tail: bool = True) -> np.ndarray:
    """
    Rate-1/2 convolutional encode.

    With ``tail=True`` (default) ``K-1`` zero tail bits are appended so
    the decoder can finish in the zero state.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        raise ValueError("No bits to encode.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary.")

    stream = np.concatenate([bits, np.zeros(NUM_TAIL_BITS, dtype=np.uint8)]) if tail else bits

    output = np.empty(stream.size * 2, dtype=np.uint8)
    state = 0
    for index, bit in enumerate(stream):
        bit = int(bit)
        g1, g2 = _OUTPUTS[state, bit]
        output[2 * index] = g1
        output[2 * index + 1] = g2
        state = int(_NEXT_STATES[state, bit])
    return output


def decode(
    code_bits: np.ndarray,
    traceback_length: int = 35,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Hard-decision Viterbi decode of a rate-1/2 codeword.

    Assumes a terminated (``tail=True``) encoder, so traceback starts
    from the zero state. Returns (decoded_bits, stats); ``decoded_bits``
    excludes the K-1 tail bits.
    """
    code_bits = np.asarray(code_bits, dtype=np.uint8).reshape(-1)
    if code_bits.size == 0 or code_bits.size % 2 != 0:
        raise ValueError("Encoded stream must contain pairs of code bits.")

    steps = code_bits.size // 2
    received = code_bits.reshape(steps, 2).astype(np.int32)

    # Branch metrics for every step at once: (steps, NUM_STATES, 2).
    branch = (
        received[:, None, None, :] != _INC_OUTPUTS[None, :, :, :]
    ).sum(axis=-1).astype(np.float64)

    path_metric = np.full(NUM_STATES, np.inf, dtype=np.float64)
    path_metric[0] = 0.0
    predecessors = np.full((steps, NUM_STATES), -1, dtype=np.int8)
    predecessor_bits = np.zeros((steps, NUM_STATES), dtype=np.uint8)

    for step in range(steps):
        # Add-compare-select over the two incoming transitions per state.
        candidate = path_metric[_INC_STATES] + branch[step]
        best = candidate.argmin(axis=1)
        path_metric = candidate[_STATE_INDEX, best]
        pred = _INC_STATES[_STATE_INDEX, best]
        unreachable = ~np.isfinite(path_metric)
        pred[unreachable] = -1
        predecessors[step] = pred.astype(np.int8)
        predecessor_bits[step] = _INC_BITS[_STATE_INDEX, best]

    if np.isfinite(path_metric[0]):
        state = 0
    else:
        finite = np.where(np.isfinite(path_metric), path_metric, np.inf)
        state = int(np.argmin(finite))

    bits_full = np.zeros(steps, dtype=np.uint8)
    for step in range(steps - 1, -1, -1):
        bits_full[step] = predecessor_bits[step, state]  # bit of that transition
        state = int(predecessors[step, state])
        if state < 0:
            raise ValueError(
                "Viterbi traceback failed: no surviving predecessor state."
            )

    if steps > NUM_TAIL_BITS:
        decoded = bits_full[:-NUM_TAIL_BITS]
    else:
        decoded = bits_full

    stats = {
        "steps": int(steps),
        "final_path_metric": (
            float(path_metric[0]) if np.isfinite(path_metric[0]) else -1.0
        ),
        "uncorrectable_blocks": 0 if np.isfinite(path_metric[0]) else 1,
    }
    return decoded, stats
