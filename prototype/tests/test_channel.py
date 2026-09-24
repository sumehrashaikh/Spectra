"""Tests for the channel simulator (simulation/channel.py)."""

import numpy as np
import pytest

from prototype.simulation.channel import ChannelConfig, apply_channel


def _tone(f=500.0, n=8000, fs=8000.0):
    t = np.arange(n) / fs
    return np.exp(1j * 2 * np.pi * f * t)


class TestImpairments:
    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            apply_channel(np.array([]), 8000.0, ChannelConfig())

    def test_noop_identity(self):
        x = _tone()
        out = apply_channel(x, 8000.0, ChannelConfig())
        assert np.allclose(out.samples, x)
        assert out.true_parameters["sample_rate"] == 8000.0

    def test_frequency_offset_truth(self):
        x = _tone()
        out = apply_channel(x, 8000.0, ChannelConfig(frequency_offset_hz=37.0))
        assert out.true_parameters["frequency_offset_hz"] == 37.0
        # measured spectral peak moves to 537 Hz
        from prototype.dsp.spectral import welch_psd

        freqs, psd = welch_psd(out.samples, 8000.0)
        assert abs(freqs[np.argmax(psd)] - 537.0) < 20.0

    def test_phase_offset_truth(self):
        x = _tone()
        out = apply_channel(x, 8000.0, ChannelConfig(phase_offset_rad=0.5))
        assert np.allclose(out.samples, x * np.exp(1j * 0.5))

    def test_timing_offset_introduces_delay(self):
        x = _tone()
        out = apply_channel(x, 8000.0, ChannelConfig(timing_offset_samples=13))
        # first 13 samples are zeroed by the delay
        assert np.allclose(out.samples[:13], 0.0, atol=1e-12)
        assert out.true_parameters["timing_offset_samples"] == 13

    def test_awgn_measured_snr(self):
        x = _tone()
        out = apply_channel(x, 8000.0, ChannelConfig(snr_db=20.0, seed=1))
        assert out.true_parameters["target_snr_db"] == 20.0
        measured = out.true_parameters["measured_snr_db"]
        assert measured == pytest.approx(20.0, abs=0.5)

    def test_multipath_convolution(self):
        x = _tone()
        taps = ((0, 1.0), (5, 0.5))
        out = apply_channel(x, 8000.0, ChannelConfig(multipath_taps=taps))
        # output at sample n should be x[n] + 0.5*x[n-5]
        expected = x + 0.5 * np.concatenate([np.zeros(5), x[:-5]])
        assert np.allclose(out.samples, expected)

    def test_iq_imbalance_truth(self):
        x = _tone()
        out = apply_channel(
            x, 8000.0,
            ChannelConfig(iq_gain_imbalance_db=3.0, iq_phase_imbalance_deg=10.0),
        )
        assert out.true_parameters["iq_gain_imbalance_db"] == 3.0
        assert out.true_parameters["iq_phase_imbalance_deg"] == 10.0

    def test_interferer_present(self):
        x = _tone(500.0, 16384)
        out = apply_channel(
            x, 8000.0,
            ChannelConfig(interferer_frequency_hz=1500.0, interferer_amplitude=0.5),
        )
        assert out.true_parameters["interferer_frequency_hz"] == 1500.0
        from prototype.dsp.spectral import welch_psd

        freqs, psd = welch_psd(out.samples, 8000.0)
        # both 500 and 1500 Hz components should be prominent
        peak_500 = psd[np.argmin(np.abs(freqs - 500.0))]
        peak_1500 = psd[np.argmin(np.abs(freqs - 1500.0))]
        assert peak_1500 > 0.1 * peak_500

    def test_clipping(self):
        x = 10.0 * _tone()
        out = apply_channel(x, 8000.0, ChannelConfig(clip_level=1.0))
        assert out.true_parameters["clip_level"] == 1.0
        assert np.max(np.abs(out.samples)) == pytest.approx(1.0)

    def test_quantization(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, 4096) + 1j * rng.uniform(-1, 1, 4096)
        out = apply_channel(x, 8000.0, ChannelConfig(quantization_bits=4))
        # quantized I values are multiples of 1/8
        scaled = out.samples.real * 8
        assert np.allclose(scaled, np.round(scaled), atol=1e-9)

    def test_impulsive_noise_counted(self):
        x = _tone()
        out = apply_channel(
            x, 8000.0,
            ChannelConfig(impulsive_noise_probability=0.1,
                          impulsive_noise_amplitude=1.0, seed=9),
        )
        assert "impulsive_hits" in out.true_parameters

    def test_doppler(self):
        x = _tone()
        out = apply_channel(x, 8000.0, ChannelConfig(doppler_rate_hz_per_s=10.0))
        assert out.true_parameters["doppler_rate_hz_per_s"] == 10.0

    def test_reproducibility_same_seed(self):
        x = _tone()
        config = ChannelConfig(snr_db=10.0, seed=123)
        a = apply_channel(x, 8000.0, config).samples
        b = apply_channel(x, 8000.0, config).samples
        assert np.array_equal(a, b)

    def test_different_seed_different_noise(self):
        x = _tone()
        a = apply_channel(x, 8000.0, ChannelConfig(snr_db=10.0, seed=1)).samples
        b = apply_channel(x, 8000.0, ChannelConfig(snr_db=10.0, seed=2)).samples
        assert not np.array_equal(a, b)

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            ChannelConfig(fading="superposition")
        with pytest.raises(ValueError):
            ChannelConfig(fractional_timing_offset=0.9)
