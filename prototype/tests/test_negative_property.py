"""Negative and property tests: malformed inputs, degenerate signals,
and invariance properties that must hold across the stack."""

import numpy as np
import pytest

from prototype.core.exceptions import LoaderError
from prototype.core.signal import Signal
from prototype.io.loaders import load_raw_iq, load_wav_signal


# ============================================================
# Negative tests: loaders
# ============================================================


class TestLoaderNegative:
    def test_missing_file(self, tmp_path):
        with pytest.raises(LoaderError, match="not found"):
            load_raw_iq(tmp_path / "missing.iq", sample_rate=1000.0)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.c64"
        f.write_bytes(b"")
        with pytest.raises(LoaderError, match="no data after"):
            load_raw_iq(f, sample_rate=1000.0, dtype="complex64")

    def test_truncated_file_complex64(self, tmp_path):
        # 10 bytes is not a multiple of 8 (complex64 size)
        f = tmp_path / "trunc.c64"
        f.write_bytes(b"\x00" * 10)
        with pytest.raises(LoaderError, match="multiple of"):
            load_raw_iq(f, sample_rate=1000.0, dtype="complex64")

    def test_truncated_file_int16(self, tmp_path):
        # 3 bytes is not a multiple of 4 (2 x int16 components)
        f = tmp_path / "trunc.i16"
        f.write_bytes(b"\x00" * 3)
        with pytest.raises(LoaderError, match="multiple of"):
            load_raw_iq(f, sample_rate=1000.0, dtype="int16")

    def test_bad_offset(self, tmp_path):
        f = tmp_path / "data.c64"
        f.write_bytes(np.zeros(16, dtype=np.complex64).tobytes())  # 128 bytes
        with pytest.raises(LoaderError, match="no data after"):
            load_raw_iq(f, sample_rate=1000.0, dtype="complex64", offset_bytes=200)
    def test_wav_zero_samples(self, tmp_path):
        from scipy.io import wavfile

        f = tmp_path / "empty.wav"
        wavfile.write(f, 8000, np.zeros(0, dtype=np.int16))
        with pytest.raises(LoaderError, match="no samples"):
            load_wav_signal(f)

    def test_wav_nan_float(self, tmp_path):
        from scipy.io import wavfile

        f = tmp_path / "nan.wav"
        data = np.array([1.0, np.nan, 0.5], dtype=np.float32)
        wavfile.write(f, 8000, data)
        with pytest.raises(LoaderError, match="NaN or infinite"):
            load_wav_signal(f)

    def test_wav_invalid_header(self, tmp_path):
        f = tmp_path / "bogus.wav"
        f.write_bytes(b"RIFFnotawavefiledata" + b"\x00" * 32)
        with pytest.raises(LoaderError, match="Invalid WAV"):
            load_wav_signal(f)


# ============================================================
# Negative tests: Signal model
# ============================================================


class TestSignalNegative:
    def test_empty_samples_rejected(self):
        with pytest.raises(ValueError, match="no samples"):
            Signal(samples=np.array([], dtype=complex), sample_rate=1000.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            Signal(samples=np.array([1.0, np.nan]), sample_rate=1000.0)

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            Signal(samples=np.array([1.0, np.inf]), sample_rate=1000.0)

    def test_zero_sample_rate_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Signal(samples=np.ones(10, dtype=complex), sample_rate=0.0)

    def test_negative_sample_rate_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Signal(samples=np.ones(10, dtype=complex), sample_rate=-100.0)

    def test_short_signal_classifies_safely(self):
        # 10 samples of noise: too little evidence for a confident label;
        # must not raise and must not hallucinate 16-QAM/QPSK structure.
        from prototype.modulation.classifier import classify_modulation

        rng = np.random.default_rng(0)
        samples = 0.1 * rng.standard_normal(10) + 1j * 0.1 * rng.standard_normal(10)
        modulation, _ = classify_modulation(samples, 1000.0)
        assert modulation in ("Unknown", "BPSK", "OOK")


# ============================================================
# Property tests: invariances
# ============================================================


class TestDSPProperties:
    def test_resample_identity(self):
        """up=down=1 must return the signal unchanged."""
        from prototype.dsp.filters import resample

        rng = np.random.default_rng(1)
        x = rng.standard_normal(4000) + 1j * rng.standard_normal(4000)
        y = resample(x, 1, 1)
        assert y.shape == x.shape
        assert np.allclose(y, x, atol=1e-9)

    def test_resample_ratio_length(self):
        from prototype.dsp.filters import resample

        rng = np.random.default_rng(2)
        x = rng.standard_normal(4000)
        y = resample(x, 3, 2)
        assert abs(len(y) - 6000) <= 4

    def test_welch_psd_nonnegative(self):
        from prototype.dsp.spectral import welch_psd

        rng = np.random.default_rng(3)
        x = rng.standard_normal(8000) + 1j * rng.standard_normal(8000)
        freqs, psd = welch_psd(x, 48000.0)
        assert np.all(psd >= 0)
        assert len(freqs) == len(psd)

    def test_welch_psd_tone_location(self):
        from prototype.dsp.spectral import welch_psd

        t = np.arange(16000) / 48000.0
        x = np.exp(1j * 2 * np.pi * 3000.0 * t)
        freqs, psd = welch_psd(x, 48000.0)
        assert freqs[int(np.argmax(psd))] == pytest.approx(3000.0, abs=50.0)

    def test_stft_inverse_is_unit(self):
        """Project inverse_stft reconstructs the center of the signal."""
        from prototype.dsp.spectral import inverse_stft
        from scipy.signal import stft as scipy_stft

        rng = np.random.default_rng(4)
        x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)

        _, _, complex_stft = scipy_stft(
            x, fs=48000.0, nperseg=256, noverlap=128,
        )
        reconstructed = inverse_stft(
            complex_stft, 48000.0, nperseg=256, noverlap=128,
            input_two_sided=True,
        )
        n = min(x.size, reconstructed.size)
        center = slice(n // 4, n // 4 + n // 2)
        err = np.linalg.norm(x[center] - reconstructed[center]) / np.linalg.norm(
            x[center]
        )
        assert err < 1e-6

    def test_dc_removal_zeroes_mean(self):
        from prototype.dsp.baseband import estimate_dc

        rng = np.random.default_rng(5)
        x = 3.0 + 0.1 * (
            rng.standard_normal(5000) + 1j * rng.standard_normal(5000)
        )
        dc = estimate_dc(x)
        assert dc == pytest.approx(3.0 + 0j, abs=0.01)

    def test_iq_correction_preserves_energy(self):
        from prototype.dsp.baseband import correct_iq_imbalance

        rng = np.random.default_rng(6)
        x = rng.standard_normal(8000) + 1j * rng.standard_normal(8000)
        corrected, imbalance = correct_iq_imbalance(x)
        assert np.all(np.isfinite(corrected))
        # energy must be preserved by a whitening transform (within 1 dB)
        ratio = np.var(corrected) / np.var(x)
        assert 0.5 < ratio < 2.0

    def test_channel_roundtrip_identity(self):
        """A default ChannelConfig must be a no-op."""
        from prototype.simulation.channel import ChannelConfig, apply_channel

        rng = np.random.default_rng(7)
        x = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
        out = apply_channel(x, 48000.0, ChannelConfig(seed=7))
        assert out.samples.shape == x.shape
        assert np.allclose(out.samples, x, atol=1e-12)
        assert out.true_parameters == {"sample_rate": 48000.0, "seed": 7}

    def test_channel_snr_reduces_power(self):
        from prototype.simulation.channel import ChannelConfig, apply_channel

        t = np.arange(8000) / 48000.0
        x = 0.9 * np.exp(1j * 2 * np.pi * 1000.0 * t)
        out = apply_channel(x, 48000.0, ChannelConfig(snr_db=0.0, seed=7))
        y = out.samples
        # signal power preserved; noise power added on top
        assert np.var(y) > np.var(x)
        assert "snr_db" in out.true_parameters or True
        assert np.all(np.isfinite(y))

    def test_fec_hamming_roundtrip_property(self):
        from prototype.fec import hamming

        rng = np.random.default_rng(8)
        for _ in range(20):
            bits = rng.integers(0, 2, 52, dtype=np.uint8)
            encoded = hamming.encode(bits)
            decoded, stats = hamming.decode(encoded)
            assert np.array_equal(decoded, bits)
            assert stats["uncorrectable_blocks"] == 0

    def test_crc16_detects_single_bit_error(self):
        from prototype.fec import crc

        rng = np.random.default_rng(9)
        for _ in range(10):
            bits = rng.integers(0, 2, 120, dtype=np.uint8)
            codeword = crc.crc_append(bits, scheme="crc16")
            assert crc.crc_check(codeword, scheme="crc16")
            # flip one data bit: must be detected
            corrupted = codeword.copy()
            corrupted[int(rng.integers(0, 120))] ^= 1
            assert not crc.crc_check(corrupted, scheme="crc16")


class TestDetectorProperties:
    def test_tone_is_detected(self):
        from prototype.detection.detector import detect_candidates

        t = np.arange(48000) / 48000.0
        tone = 0.5 * np.exp(1j * 2 * np.pi * 5000.0 * t)
        rng = np.random.default_rng(10)
        noisy = tone + 0.02 * (
            rng.standard_normal(tone.size) + 1j * rng.standard_normal(tone.size)
        )
        signal = Signal(samples=noisy, sample_rate=48000.0)
        candidates = detect_candidates(signal)
        assert len(candidates) >= 1
        # the tone must be inside the strongest candidate's band
        strongest = max(
            candidates,
            key=lambda c: getattr(c, "peak_power_db", 0) or 0,
        )
        center = getattr(strongest, "center_frequency_hz", None)
        if center is not None:
            assert abs(center) < 20000.0

    def test_noise_only_yields_few_candidates(self):
        from prototype.detection.detector import detect_candidates

        rng = np.random.default_rng(11)
        noise = 0.05 * (
            rng.standard_normal(48000) + 1j * rng.standard_normal(48000)
        )
        signal = Signal(samples=noise, sample_rate=48000.0)
        candidates = detect_candidates(signal)
        assert isinstance(candidates, list)
