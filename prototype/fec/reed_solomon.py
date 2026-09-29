"""Reed-Solomon over GF(2**8): systematic shortened RS with the
Berlekamp-Massey / Chien / Vandermonde error-correction chain.

The codec exposes the same bit-oriented ``encode`` / ``decode`` interface
as the rest of :mod:`prototype.fec`, so it slots straight into the scheme
registry and the pipeline.

Conventions
-----------
* Symbols are bytes; GF(2**8) uses the primitive polynomial ``0x11D`` and
  generator element ``alpha = 2`` (the CCSDS / DVB / QR-code field).
* Codewords are treated big-endian: index ``0`` is the highest-degree
  coefficient.  Evaluation points are ``alpha**1 .. alpha**(2t)``.
* Blocks are shortened to ``K_DEFAULT`` data bytes with ``NSYM_DEFAULT``
  parity bytes (``t = 4`` correctable symbol errors per block).  Data is
  zero-padded to a whole number of blocks at encode time.

This is a compact, exact, deterministic RS suitable for tests and
demonstrations; it is not tuned for throughput.
"""

from __future__ import annotations

import numpy as np

PRIMITIVE_POLY = 0x11D
FIELD_SIZE = 256
ORDER = 255  # multiplicative order of the field

K_DEFAULT = 32   # data bytes per shortened block
NSYM_DEFAULT = 8  # parity bytes per block -> t = 4 correctable symbols
MAX_BLOCK = 255


# ---------------------------------------------------------------------------
# GF(2**8) arithmetic
# ---------------------------------------------------------------------------

_EXP = np.zeros(ORDER * 2, dtype=np.int32)
_LOG = np.zeros(FIELD_SIZE, dtype=np.int32)
_x = 1
for _i in range(ORDER):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIMITIVE_POLY
for _i in range(ORDER, ORDER * 2):
    _EXP[_i] = _EXP[_i - ORDER]


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_EXP[int(_LOG[a]) + int(_LOG[b])])


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if a == 0:
        return 0
    return int(_EXP[(int(_LOG[a]) - int(_LOG[b])) % ORDER])


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(256) inverse of zero")
    return int(_EXP[ORDER - int(_LOG[a])])


def gf_pow(a: int, power: int) -> int:
    if a == 0:
        return 0
    return int(_EXP[(int(_LOG[a]) * power) % ORDER])


def alpha_pow(power: int) -> int:
    """``alpha**power`` for any integer power (wrap-around handled)."""
    return int(_EXP[power % ORDER])


def _poly_eval(poly, x: int) -> int:
    """Horner evaluation of a big-endian polynomial at ``x``."""
    y = 0
    for coef in poly:
        y = gf_mul(y, x) ^ int(coef)
    return y


# ---------------------------------------------------------------------------
# Generator polynomial
# ---------------------------------------------------------------------------

def _generator_poly(nsym: int) -> list[int]:
    """g(x) = prod_{i=1..nsym} (x - alpha**i), big-endian, monic."""
    g = [1]
    for i in range(1, nsym + 1):
        root = alpha_pow(i)
        new = [0] * (len(g) + 1)
        for j, coef in enumerate(g):
            new[j] ^= coef
            new[j + 1] ^= gf_mul(coef, root)
        g = new
    return g


def _gf_solve(matrix: list[list[int]], rhs: list[int]) -> list[int]:
    """Solve ``matrix @ x = rhs`` over GF(256) via Gauss-Jordan.

    Raises :class:`ValueError` when the system is singular.
    """
    n = len(matrix)
    aug = [list(row) + [int(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("singular syndrome system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = gf_inv(aug[col][col])
        aug[col] = [gf_mul(v, inv) for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a ^ gf_mul(factor, b) for a, b in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def _berlekamp_massey(syndromes: list[int], nsym: int) -> tuple[list[int], int]:
    """Return the error-locator polynomial (little-endian, C[0]=1) and L.

    ``syndromes[i]`` is ``S_{i+1}`` (i.e. ``r(alpha**(i+1))``).
    """
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for n_idx in range(nsym):
        d = int(syndromes[n_idx])
        for i in range(1, L + 1):
            ci = C[i] if i < len(C) else 0
            d ^= gf_mul(ci, int(syndromes[n_idx - i]))
        if d == 0:
            m += 1
            continue
        T = list(C)
        coef = gf_div(d, b)
        shifted = [0] * (len(B) + m)
        for i, bc in enumerate(B):
            shifted[i + m] = gf_mul(int(bc), coef)
        new_len = max(len(C), len(shifted))
        C = [
            (C[i] if i < len(C) else 0) ^ (shifted[i] if i < len(shifted) else 0)
            for i in range(new_len)
        ]
        if 2 * L <= n_idx:
            L = n_idx + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1
    return C, L


def _locator_eval(locator: list[int], x: int) -> int:
    """Evaluate the little-endian locator ``sum locator[i] * x**i``."""
    y = 0
    for coef in reversed(locator):
        y = gf_mul(y, x) ^ int(coef)
    return y


def _syndromes(codeword: list[int], nsym: int) -> list[int]:
    return [_poly_eval(codeword, alpha_pow(j)) for j in range(1, nsym + 1)]


def _correct_block(codeword: list[int], nsym: int) -> tuple[list[int], int]:
    """Correct up to ``nsym // 2`` symbol errors; returns (fixed, count).

    Raises :class:`ValueError` when the errors exceed the code's capacity.
    """
    synd = _syndromes(codeword, nsym)
    if not any(synd):
        return codeword, 0

    locator, L = _berlekamp_massey(synd, nsym)
    if L == 0 or 2 * L > nsym:
        raise ValueError("too many errors to correct")

    n = len(codeword)
    positions: list[int] = []
    for p in range(n):
        # error at index p -> X = alpha**(n-1-p); Chien tests alpha**-(n-1-p)
        root = alpha_pow(-(n - 1 - p))
        if _locator_eval(locator, root) == 0:
            positions.append(p)
    if len(positions) != L:
        raise ValueError("Chien search found an inconsistent number of errors")

    # Magnitudes: S_j = sum_i e_i * X_i**j  for j = 1..L (Vandermonde solve).
    X = [alpha_pow(n - 1 - p) for p in positions]
    matrix = [[gf_pow(X[i], j) for i in range(L)] for j in range(1, L + 1)]
    magnitudes = _gf_solve(matrix, synd[:L])

    corrected = list(codeword)
    for p, mag in zip(positions, magnitudes):
        corrected[p] ^= int(mag)

    if any(_syndromes(corrected, nsym)):
        raise ValueError("residual syndromes after correction")
    return corrected, sum(1 for m in magnitudes if m != 0)


# ---------------------------------------------------------------------------
# Bits <-> bytes helpers
# ---------------------------------------------------------------------------

def _bits_to_bytes(bits: np.ndarray) -> list[int]:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary.")
    out = []
    for i in range(0, bits.size, 8):
        chunk = bits[i : i + 8]
        value = 0
        for b in chunk:
            value = (value << 1) | int(b)
        out.append(value)
    return out


def _bytes_to_bits(data: list[int]) -> np.ndarray:
    bits = np.empty(len(data) * 8, dtype=np.uint8)
    for i, byte in enumerate(data):
        for j in range(8):
            bits[i * 8 + j] = (int(byte) >> (7 - j)) & 1
    return bits


# ---------------------------------------------------------------------------
# Public codec
# ---------------------------------------------------------------------------

def encode(
    bits: np.ndarray,
    nsym: int = NSYM_DEFAULT,
    k_data: int = K_DEFAULT,
) -> np.ndarray:
    """Systematic shortened Reed-Solomon encode of ``bits``.

    Data is zero-padded to a whole number of ``k_data``-byte blocks; each
    block becomes ``k_data + nsym`` bytes (data followed by parity).
    """
    if nsym < 1 or k_data < 1 or k_data + nsym > MAX_BLOCK:
        raise ValueError("invalid Reed-Solomon block geometry")
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to encode.")
    arr = np.concatenate(
        [arr, np.zeros((-(arr.size) % (k_data * 8)) % (k_data * 8), dtype=np.uint8)]
    )
    gen = _generator_poly(nsym)
    n = k_data + nsym

    payload = _bits_to_bytes(arr)
    out_bytes: list[int] = []
    for start in range(0, len(payload), k_data):
        data = payload[start : start + k_data]
        padded = [0] * (n)
        padded[:k_data] = data
        for i in range(k_data):
            coef = padded[i]
            if coef != 0:
                for j in range(len(gen)):
                    padded[i + j] ^= gf_mul(gen[j], coef)
        out_bytes.extend(data)
        out_bytes.extend(padded[k_data:n])
    return _bytes_to_bits(out_bytes)


def decode(
    bits: np.ndarray,
    nsym: int = NSYM_DEFAULT,
    k_data: int = K_DEFAULT,
) -> tuple[np.ndarray, dict[str, int]]:
    """Decode shortened RS blocks, correcting up to ``nsym // 2`` symbol errors.

    Uncorrectable blocks are returned unchanged (parity stripped) and
    counted in ``uncorrectable_blocks`` instead of raising, so the pipeline
    keeps flowing on noisy input.
    """
    if nsym < 1 or k_data < 1 or k_data + nsym > MAX_BLOCK:
        raise ValueError("invalid Reed-Solomon block geometry")
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        raise ValueError("No bits to decode.")
    if arr.size % 8 != 0:
        raise ValueError("Reed-Solomon decode requires a whole number of bytes.")
    raw = _bits_to_bytes(arr)
    n = k_data + nsym
    if len(raw) % n != 0:
        raise ValueError(
            f"Reed-Solomon decode requires blocks of {n} bytes, got {len(raw)}."
        )

    corrected_errors = 0
    uncorrectable = 0
    out_bytes: list[int] = []
    for start in range(0, len(raw), n):
        block = raw[start : start + n]
        try:
            fixed, count = _correct_block(block, nsym)
            out_bytes.extend(fixed[:k_data])
            corrected_errors += count
        except ValueError:
            out_bytes.extend(block[:k_data])
            uncorrectable += 1

    stats = {
        "blocks": len(raw) // n,
        "corrected_errors": int(corrected_errors),
        "uncorrectable_blocks": int(uncorrectable),
    }
    return _bytes_to_bits(out_bytes), stats
