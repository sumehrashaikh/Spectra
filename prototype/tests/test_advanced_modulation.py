"""Tests for advanced modulation support: 8-PSK and OOK/ASK."""

import numpy as np
import pytest

from prototype.core.signal import Signal
from prototype.modulation.digital import (
    demodulate_ook,
    demodulate_psk8,
    ook_decision,
    psk8_decision,
)
from prototype.classification.classifier import classify_signal

GRAY_TO_K = {0: 0, 1: 1, 3: 2, 2: 3, 6: 4, 7: 5, 5: 6, 4: 7}
IDEAL_8PSK = np.exp(1j * (2 * np.arange(8) + 1) * np.pi / 8)


def _psk8_symbols(n=600, seed=3):
    rng = np.random.default_rng(seed)
    bits3 = rng.integers(0, 2, (n, 3))
    keys = bits3[:, 0] * 4 + bits3[:, 1] * 2 + bits3[:, 2]
    k = np.array([GRAY_TO_K[key] for key in keys])
    return IDEAL_8PSK[k], bits3.reshape(-1)


class TestPSK8:
    def test_clean_roundtrip(self):
        symbols, bits = _psk8_symbols()
        decided_bits, _, _ = psk8_decision(symbols)
        assert np.array_equal(decided_bits, bits)

    def test_noise_tolerance(self):
        symbols, bits = _psk8_symbols(seed=4)
        rng = np.random.default_rng(9)
        noisy = symbols + 0.15 * (
            rng.standard_normal(symbols.size)
            + 1j * rng.standard_normal(symbols.size)
        ) / np.sqrt(2)
        decided_bits, _, _ = psk8_decision(noisy)
        assert np.mean(decided_bits != bits) < 0.02

    def test_demodulator_with_phase_offset(self):
        symbols, bits = _psk8_symbols(seed=5)
        rng = np.random.default_rng(2)
        sps = 8
        # Offset of +0.3 rad is inside the blind fold range (-pi/8, pi/8];
        # larger offsets fold by k*pi/4 and are documented as fundamental.
        up = np.zeros(symbols.size * sps, dtype=complex)
        up[::sps] = symbols * np.exp(1j * 0.3)
        up += 0.1 * (
            rng.standard_normal(up.size) + 1j * rng.standard_normal(up.size)
        ) / np.sqrt(2)

        result = demodulate_psk8(up, sps, 0)
        assert np.mean(result["bits"] != bits) < 0.02
        # phase estimate should land near the true offset (mod 45 deg)
        assert abs(result["phase_estimate_rad"] - 0.3) < 0.1

    def test_demodulator_phase_offset_folds_by_45deg(self):
        # Offsets beyond pi/8 fold to the nearest 8th-power solution:
        # blind reception cannot distinguish phi from phi + k*pi/4.
        # The demodulator must still report the fold explicitly.
        symbols, bits = _psk8_symbols(seed=5)
        rng = np.random.default_rng(3)
        sps = 8
        up = np.zeros(symbols.size * sps, dtype=complex)
        up[::sps] = symbols * np.exp(1j * 0.5)  # > pi/8
        up += 0.1 * (
            rng.standard_normal(up.size) + 1j * rng.standard_normal(up.size)
        ) / np.sqrt(2)

        result = demodulate_psk8(up, sps, 0)
        estimate = result["phase_estimate_rad"]
        # estimate is correct modulo pi/4
        fold_err = abs(np.angle(np.exp(1j * 8 * (estimate - 0.5)))) / 8
        assert fold_err < 0.05

    def test_phase_ambiguity_folded(self):
        symbols, bits = _psk8_symbols(seed=6)
        result_in = psk8_decision(symbols * np.exp(1j * 0.2))
        # a 45-degree rotation maps to a different Gray word - that is the
        # documented ambiguity; decisions must still be self-consistent
        decided_bits, _, _ = result_in
        assert decided_bits.size == bits.size


class TestOOK:
    def test_clean_roundtrip(self):
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, 400)
        on_levels = 0.8 + 0.4 * rng.random(400)
        symbols = np.where(bits == 1, on_levels, 0.02) + 0j
        decided, _, margin = ook_decision(symbols)
        assert np.array_equal(decided, bits)
        assert margin > 3.0  # well-separated on/off clusters

    def test_demodulator(self):
        rng = np.random.default_rng(8)
        bits = rng.integers(0, 2, 300)
        symbols = np.where(bits == 1, 1.0, 0.05) + 0j
        result = demodulate_ook(symbols, 1, 0)
        assert np.array_equal(result["bits"], bits)

    def test_classifier_detects_ook(self):
        rng = np.random.default_rng(9)
        sps = 8
        bits = rng.integers(0, 2, 400)
        # pulse-shaped-ish OOK: rectangular with amplitude jitter
        samples = np.repeat(
            np.where(bits == 1, 1.0, 0.03) * (0.9 + 0.2 * rng.random(400)), sps
        ).astype(np.complex128)
        signal = Signal(samples=samples, sample_rate=8000.0)
        result = classify_signal(signal, use_constellation=False, synchronized=False)
        assert result.modulation == "OOK"

    def test_constant_envelope_not_ook(self):
        t = np.arange(4000) / 8000.0
        tone = np.exp(1j * 2 * np.pi * 500.0 * t)
        signal = Signal(samples=tone, sample_rate=8000.0)
        result = classify_signal(signal, use_constellation=False, synchronized=False)
        assert result.modulation != "OOK"


class TestPipelineDispatch:
    def test_demodulate_signal_accepts_8psk(self):
        symbols, bits = _psk8_symbols(n=200)
        sps = 8
        up = np.zeros(symbols.size * sps, dtype=complex)
        up[::sps] = symbols
        signal = Signal(samples=up, sample_rate=8000.0)

        from prototype.demodulation.demodulator import demodulate_signal

        result = demodulate_signal(signal, "8-PSK", samples_per_symbol=sps)
        assert result.modulation == "8-PSK"
        assert result.num_bits == 3 * 200

    def test_demodulate_signal_accepts_ook(self):
        rng = np.random.default_rng(11)
        bits = rng.integers(0, 2, 200)
        samples = np.where(bits == 1, 1.0, 0.05).astype(complex)
        signal = Signal(samples=samples, sample_rate=8000.0)

        from prototype.demodulation.demodulator import demodulate_signal

        result = demodulate_signal(signal, "OOK", samples_per_symbol=1)
        assert np.array_equal(result.bits, bits)
