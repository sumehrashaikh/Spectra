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


_NEXT_STATES, _OUTPUTS = _build_tables()


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
    received = code_bits.reshape(steps, 2)

    path_metric = np.full(NUM_STATES, np.inf, dtype=np.float64)
    path_metric[0] = 0.0
    predecessors = np.full((steps, NUM_STATES), -1, dtype=np.int8)

    for step in range(steps):
        r0 = int(received[step, 0])
        r1 = int(received[step, 1])
        new_metric = np.full(NUM_STATES, np.inf, dtype=np.float64)
        pred = np.full(NUM_STATES, -1, dtype=np.int8)

        for previous_state in range(NUM_STATES):
            if not np.isfinite(path_metric[previous_state]):
                continue
            base = path_metric[previous_state]
            for bit in (0, 1):
                next_state = int(_NEXT_STATES[previous_state, bit])
                expected = _OUTPUTS[previous_state, bit]
                branch = (r0 != int(expected[0])) + (r1 != int(expected[1]))
                candidate = base + branch
                if candidate < new_metric[next_state]:
                    new_metric[next_state] = candidate
                    pred[next_state] = previous_state

        path_metric = new_metric
        predecessors[step] = pred

    if np.isfinite(path_metric[0]):
        state = 0
    else:
        finite = np.where(np.isfinite(path_metric), path_metric, np.inf)
        state = int(np.argmin(finite))

    bits_full = np.zeros(steps, dtype=np.uint8)
    for step in range(steps - 1, -1, -1):
        bits_full[step] = (state >> 5) & 1  # bit chosen for the transition
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
