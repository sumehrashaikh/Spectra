"""Deterministic end-to-end demo captures for the FEC/interleaving chain.

One place builds the transmitter side of the demo chain:

    payload bits -> FEC encode -> interleave -> symbol map -> RRC shape
                 -> upconvert -> (optional) known timing/carrier/phase
                 impairments -> (optional) AWGN

so tests, the CLI and the GUI all exercise the *same* deterministic
captures instead of three near-copies.  Runs as a script to write WAV
demo files:

    python tests/demo_captures.py            # writes demo_*.wav in cwd
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import upfirdn

# ---------------------------------------------------------------------------
# Deterministic waveform parameters (identical to the existing
# end-to-end modulation regressions, so behaviour stays comparable).
# ---------------------------------------------------------------------------

SAMPLE_RATE = 8000.0
SYMBOL_RATE = 100.0
SPS = int(SAMPLE_RATE / SYMBOL_RATE)
NUM_SYMBOLS = 512

# 16-QAM Gray levels, per-axis (b0, b1) -> level: 00 -> -3, 01 -> -1,
# 11 -> +1, 10 -> +3, exactly as modulation.demodulator expects.
QAM16_LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(10.0)

# QPSK Gray levels per axis: 0 -> -1/sqrt(2), 1 -> +1/sqrt(2).
QPSK_LEVELS = np.array([-1.0, 1.0]) / np.sqrt(2.0)

# 8-PSK Gray constellation offsets (units of pi/4).
PSK8_ANGLES = np.pi / 4.0 * np.arange(8)


@dataclass
class DemoCapture:
    """One deterministic capture plus the ground truth it was built from."""

    name: str
    modulation: str
    fec_scheme: str | None
    interleave_family: str | None
    interleave_param: int | None
    payload_bits: np.ndarray
    coded_bits: np.ndarray
    samples: np.ndarray
    sample_rate: float = SAMPLE_RATE
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transmitter building blocks
# ---------------------------------------------------------------------------


def payload(nbits: int = 1024, seed: int = 5) -> np.ndarray:
    """Deterministic payload bitstream (the bits a receiver must recover)."""
    return np.random.default_rng(seed).integers(0, 2, nbits).astype(np.uint8)


def fec_encode(bits: np.ndarray, scheme: str | None) -> np.ndarray:
    """Apply the transmitter-side FEC (``None`` = uncoded)."""
    if scheme is None:
        return np.asarray(bits, dtype=np.uint8).copy()
    from prototype.fec.framework import encode_bits

    encoded, _ = encode_bits(np.asarray(bits, dtype=np.uint8), scheme)
    return np.asarray(encoded, dtype=np.uint8)


def interleave(
    bits: np.ndarray,
    family: str | None,
    param: int | None,
) -> np.ndarray:
    """Apply the transmitter-side interleaver (``None`` = as-is)."""
    from prototype.fec import interleaving

    arr = np.asarray(bits, dtype=np.uint8)
    if family is None:
        return arr.copy()
    if family == "block":
        return interleaving.interleave_bits(arr, depth=int(param or 8))
    if family == "convolutional":
        return interleaving.convolutional_interleave(arr, k=int(param or 2))
    if family == "diagonal":
        return interleaving.diagonal_interleave(arr, depth=int(param or 8))
    if family == "pseudo_random":
        return interleaving.pseudo_random_interleave(arr, seed=int(param or 0))
    raise ValueError(f"unknown interleave family {family!r}")


def map_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map a bitstream onto complex symbols for ``modulation``."""
    n = bits.size
    if modulation == "16-QAM":
        usable = (n // 4) * 4
        b = bits[:usable]
        return QAM16_LEVELS[b[0::4] * 2 + b[1::4]] + 1j * QAM16_LEVELS[
            b[2::4] * 2 + b[3::4]
        ]
    if modulation == "QPSK":
        usable = (n // 2) * 2
        b = bits[:usable]
        return QPSK_LEVELS[b[0::2]] + 1j * QPSK_LEVELS[b[1::2]]
    if modulation == "8-PSK":
        usable = (n // 3) * 3
        b = bits[:usable].reshape(-1, 3)
        index = b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]
        return np.exp(1j * PSK8_ANGLES[index])
    raise ValueError(f"unsupported demo modulation {modulation!r}")


# BFSK demo profile (same tones/symbol rate as the legacy V1 BFSK flow).
BFSK_FREQ_0 = 500.0
BFSK_FREQ_1 = 700.0
BFSK_SYMBOL_RATE = 100.0


def build_fsk_capture(
    nbits: int = 1000,
    seed: int = 42,
    noise_amplitude: float = 0.0,
) -> DemoCapture:
    """Deterministic continuous-phase BFSK capture.

    FSK carries its information in the instantaneous frequency, so it is
    not pulse-shaped through the RRC path and does not go through
    ``map_symbols``; the receiver's BFSK branch estimates the two tones
    from the capture itself.  Included so the demo set covers the FSK
    PS requirement alongside PSK/QAM.
    """
    sps = int(SAMPLE_RATE / BFSK_SYMBOL_RATE)
    bits = payload(nbits, seed)
    frequencies = np.where(bits == 0, BFSK_FREQ_0, BFSK_FREQ_1)

    phase = np.zeros(nbits * sps)
    current = 0.0
    index = 0
    for frequency in frequencies:
        t = np.arange(sps) / SAMPLE_RATE
        symbol_phase = current + 2.0 * np.pi * frequency * t
        phase[index:index + sps] = symbol_phase
        current = symbol_phase[-1] + 2.0 * np.pi * frequency / SAMPLE_RATE
        index += sps

    iq = np.exp(1j * phase)
    if noise_amplitude:
        rng = np.random.default_rng(seed + 1)
        iq = iq + noise_amplitude * (
            rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)
        ) / np.sqrt(2.0)

    return DemoCapture(
        name="BFSK_nofec_nointerleave",
        modulation="BFSK",
        fec_scheme=None,
        interleave_family=None,
        interleave_param=None,
        payload_bits=bits,
        coded_bits=bits.copy(),
        samples=iq,
        notes=[f"modulation=BFSK", f"tones={BFSK_FREQ_0:.0f}/{BFSK_FREQ_1:.0f} Hz"],
    )


def build_capture(
    modulation: str = "16-QAM",
    fec_scheme: str | None = None,
    interleave_family: str | None = None,
    interleave_param: int | None = None,
    nbits: int = 1024,
    seed: int = 5,
    carrier_hz: float = 500.0,
    timing_offset: int = 13,
    carrier_offset_hz: float = 37.0,
    phase_deg: float = 30.0,
    noise_amplitude: float = 0.004,
    noise_seed: int = 123,
) -> DemoCapture:
    """Build one deterministic capture end to end (transmitter side only)."""
    bits = payload(nbits, seed)
    coded = fec_encode(bits, fec_scheme)
    if interleave_family is not None:
        coded = interleave(coded, interleave_family, interleave_param)

    # The receiver must see exactly the transmitted code bits: padding or
    # truncating a codeword here would make the demo meaningless.
    bits_per_symbol = {"16-QAM": 4, "QPSK": 2, "8-PSK": 3}[modulation]
    if coded.size % bits_per_symbol:
        raise ValueError(
            f"{modulation} demo needs a code-bit count that is a multiple of "
            f"{bits_per_symbol}, got {coded.size} (payload {bits.size}, "
            f"fec={fec_scheme}, interleave={interleave_family})"
        )

    symbols = map_symbols(coded, modulation)

    taps = rrc_filter(SPS, rolloff=0.35, span_symbols=8)
    shaped = upfirdn(taps, symbols, up=SPS)[: symbols.size * SPS]

    t = np.arange(shaped.size) / SAMPLE_RATE
    iq = shaped * np.exp(1j * 2.0 * np.pi * carrier_hz * t)

    if timing_offset:
        iq = np.concatenate([np.zeros(int(timing_offset), dtype=np.complex128), iq])
    if carrier_offset_hz:
        n = np.arange(iq.size)
        iq = iq * np.exp(1j * 2.0 * np.pi * carrier_offset_hz * n / SAMPLE_RATE)
    if phase_deg:
        iq = iq * np.exp(1j * np.deg2rad(phase_deg))

    notes = [
        f"modulation={modulation}",
        f"fec={fec_scheme}",
        f"interleave={interleave_family}",
    ]

    if noise_amplitude:
        rng = np.random.default_rng(noise_seed)
        noise = noise_amplitude * (
            rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)
        ) / np.sqrt(2.0)
        iq = iq + noise
        notes.append(f"awgn_amplitude={noise_amplitude}")

    return DemoCapture(
        name=(
            f"{modulation}_{fec_scheme or 'nofec'}_"
            f"{interleave_family or 'nointerleave'}"
        ),
        modulation=modulation,
        fec_scheme=fec_scheme,
        interleave_family=interleave_family,
        interleave_param=interleave_param,
        payload_bits=bits,
        coded_bits=coded,
        samples=iq,
        notes=notes,
    )


def rrc_filter(num_samples: int, rolloff: float = 0.35, span_symbols: int = 8):
    """Import-free RRC wrapper (reuses the project's filter implementation)."""
    from prototype.parameters.symbol_rate import rrc_filter as _rrc

    return _rrc(num_samples, rolloff=rolloff, span_symbols=span_symbols)


def write_wav(path: str, samples: np.ndarray, sample_rate: float = SAMPLE_RATE) -> str:
    """Write a stereo float WAV capture (I in the left, Q in the right)."""
    from scipy.io import wavfile

    peak = float(np.max(np.abs(samples))) or 1.0
    scaled = samples / peak * 0.9
    stereo = np.stack([scaled.real, scaled.imag], axis=1).astype(np.float32)
    wavfile.write(path, int(sample_rate), stereo)
    return path


def reference_path_for_wav(path: str) -> str:
    """Companion ``<stem>.reference.npz`` path used by the BER/alignment stage."""
    from pathlib import Path

    wav = Path(path)
    return str(wav.with_name(f"{wav.stem}.reference.npz"))


def write_reference(path: str, bits: np.ndarray) -> str:
    """Write the transmitted-bit reference next to a demo WAV.

    With this sidecar present the receiver resolves the blind phase/origin
    fold and measures BER, so the GUI and CLI demonstrate the *clean*
    chain instead of the deliberately honest partial result they show when
    no reference exists.
    """
    np.savez(path, bits=np.asarray(bits, dtype=np.uint8).reshape(-1))
    return path


def reference_is_usable(capture: "DemoCapture") -> bool:
    """Whether a BER reference helps this capture (not every case can use one).

    The reference drives the codeword-alignment stage, which resolves the
    blind phase/origin fold for 16-QAM (and, with FEC, for the coded
    stream).  It is *not* implemented for uncoded QPSK/8-PSK, so writing a
    reference there would only make the receiver report a misleading ~0.5
    BER from the unresolved constellation rotation; those captures are left
    without a sidecar so the GUI honestly says "No reference loaded".
    """
    return capture.modulation == "16-QAM" or capture.fec_scheme is not None


def write_capture(path: str, capture: "DemoCapture", with_reference: bool = True) -> str:
    """Write a demo capture's WAV plus, when usable, its bit-reference sidecar.

    The reference is the transmitted *code* bits (equal to the payload for
    an uncoded capture) -- exactly the stream the receiver must recover -- so
    it drives both the codeword alignment and the BER stage.
    """
    write_wav(path, capture.samples, capture.sample_rate)
    if with_reference and reference_is_usable(capture):
        write_reference(reference_path_for_wav(path), capture.coded_bits)
    return path


# ---------------------------------------------------------------------------
# Demo capture catalogue
# ---------------------------------------------------------------------------


def demo_cases() -> list[dict]:
    """The deterministic demo matrix: modulation x FEC x interleaving.

    Every entry here is *demonstrable* end to end: the receiver classifies
    the intended modulation and, where FEC is present, decodes the payload
    cleanly (or reports an honest partial/corrected result).  Cases that
    only exercise the bit-level interleaver wiring, without a working blind
    RF chain, live in the round-trip matrix in ``test_end_to_end_fec_demo``
    instead of here.

    Interleaver constraints are respected by construction: block /
    convolutional / pseudo-random need a whole multiple of their stride,
    and the (square) diagonal interleaver needs exactly depth**2 bits --
    hence the 6400-bit depth-80 diagonal case.  Each modulation also needs
    the coded length to be a whole number of bits/symbol (4 / 2 / 3 for
    16-QAM / QPSK / 8-PSK).
    """
    return [
        {"modulation": "16-QAM", "fec_scheme": None,
         "interleave_family": None, "interleave_param": None},
        {"modulation": "16-QAM", "fec_scheme": "conv12",
         "interleave_family": None, "interleave_param": None},
        {"modulation": "16-QAM", "fec_scheme": "reedsolomon",
         "interleave_family": "block", "interleave_param": 8},
        {"modulation": "16-QAM", "fec_scheme": "concatenated",
         "interleave_family": "block", "interleave_param": 8},
        # Concatenated + pseudo-random: the concatenated frame (640k+12
        # bits) is a whole multiple of the pseudo-random stride and this
        # combination recovers cleanly, so it carries the "other
        # de-interleaver" demonstration.
        {"modulation": "16-QAM", "fec_scheme": "concatenated",
         "interleave_family": "pseudo_random", "interleave_param": 3},
        {"modulation": "QPSK", "fec_scheme": None,
         "interleave_family": None, "interleave_param": None},
        {"modulation": "QPSK", "fec_scheme": None,
         "interleave_family": "block", "interleave_param": 8},
        {"modulation": "QPSK", "fec_scheme": None,
         "interleave_family": "convolutional", "interleave_param": 2},
        # Diagonal needs exactly depth**2 = 6400 bits (3200 QPSK symbols),
        # which is long enough for the PSK synchronizer.
        {"modulation": "QPSK", "fec_scheme": None,
         "interleave_family": "diagonal", "interleave_param": 80,
         "nbits": 6400},
        {"modulation": "8-PSK", "fec_scheme": None,
         "interleave_family": "block", "interleave_param": 8,
         "nbits": 1008},
    ]


def build_demo_captures() -> list[DemoCapture]:
    """Build every demo capture (deterministic, no randomness outside rng)."""
    return [build_capture(**case) for case in demo_cases()] + [build_fsk_capture()]


if __name__ == "__main__":  # pragma: no cover - manual demo helper
    for capture in build_demo_captures():
        path = f"demo_{capture.name}.wav"
        write_capture(path, capture)
        print(
            f"{path:52s} payload={capture.payload_bits.size:5d} "
            f"coded={capture.coded_bits.size:6d} "
            f"samples={capture.samples.size:7d}  # {'; '.join(capture.notes)}"
        )
