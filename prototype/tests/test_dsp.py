"""Tests for the DSP engine (dsp/): spectral, filters, baseband, correlation."""

import numpy as np
import pytest

from prototype.dsp import baseband, correlation, filters, spectral
from prototype.dsp.spectral import next_power_of_two


@pytest.fixture
def tone_500():
    t = np.arange(8192) / 8000.0
    return np.exp(1j * 2 * np.pi * 500.0 * t)


class TestSpectral:
    def test_next_power_of_two(self):
        assert next_power_of_two(1) == 1
        assert next_power_of_two(1000) == 1024
        assert next_power_of_two(1024) == 1024

    def test_welch_psd_peak_location(self, tone_500):
        freqs, psd = spectral.welch_psd(tone_500, 8000)
        peak = freqs[np.argmax(psd)]
        assert abs(peak - 500.0) < 20.0

    def test_welch_psd_two_sided_for_complex(self, tone_500):
        freqs, psd = spectral.welch_psd(tone_500, 8000)
        assert freqs.min() < 0 < freqs.max()
        assert freqs.size == psd.size

    def test_welch_psd_real_input_one_sided(self):
        t = np.arange(8192) / 8000.0
        real = np.cos(2 * np.pi * 300.0 * t)
        freqs, psd = spectral.welch_psd(real, 8000)
        assert freqs.min() >= 0

    def test_psd_db(self, tone_500):
        _, psd = spectral.welch_psd(tone_500, 8000)
        psd_db = spectral.psd_db(psd)
        assert psd_db.shape == psd.shape
        assert np.all(np.isfinite(psd_db))

    def test_spectrogram_shape(self, tone_500):
        freqs, times, sxx = spectral.spectrogram_stft(tone_500, 8000, nperseg=256)
        assert sxx.shape[0] == freqs.size
        assert sxx.shape[1] == times.size
        peak_bin = np.unravel_index(np.argmax(sxx), sxx.shape)
        assert abs(freqs[peak_bin[0]] - 500.0) < 100.0

    def test_stft_inverse_roundtrip(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4096) + 0j
        # recompute the complex STFT for inversion
        from scipy.signal import stft as _stft

        _, _, complex_stft = _stft(x, fs=8000, nperseg=256, noverlap=128)
        reconstructed = spectral.inverse_stft(
            complex_stft, 8000, nperseg=256, noverlap=128,
            input_two_sided=True,
        )
        n = min(x.size, reconstructed.size)
        # Hann with 50% overlap satisfies COLA; reconstruction is exact
        center = slice(n // 4, n // 4 + 1024)
        assert np.allclose(
            reconstructed[center], x[center], atol=1e-9
        )

    def test_short_signal(self):
        with pytest.raises(ValueError):
            spectral.welch_psd(np.array([1.0]), 8000)


class TestFilters:
    def test_fir_lowpass_attenuates_high(self):
        t = np.arange(8192) / 8000.0
        low = np.exp(1j * 2 * np.pi * 200.0 * t)
        high = np.exp(1j * 2 * np.pi * 3000.0 * t)
        taps = filters.fir_lowpass(129, 800.0, 8000.0)
        out_low = filters.apply_fir(low, taps)
        out_high = filters.apply_fir(high, taps)
        # Compare power in the steady-state interior.
        p_low = np.mean(np.abs(out_low[1000:-1000]) ** 2)
        p_high = np.mean(np.abs(out_high[1000:-1000]) ** 2)
        assert 10 * np.log10(p_high / p_low) < -40

    def test_fir_bandpass(self):
        t = np.arange(8192) / 8000.0
        in_band = np.exp(1j * 2 * np.pi * 1000.0 * t)
        out_band = np.exp(1j * 2 * np.pi * 3000.0 * t)
        taps = filters.fir_bandpass(129, 800.0, 1500.0, 8000.0)
        p_in = np.mean(np.abs(filters.apply_fir(in_band, taps)[1000:-1000]) ** 2)
        p_out = np.mean(np.abs(filters.apply_fir(out_band, taps)[1000:-1000]) ** 2)
        assert 10 * np.log10(p_out / p_in) < -40

    def test_iir_sos_zero_phase(self, tone_500):
        sos = filters.iir_sos(4, 600.0, 8000.0, "lowpass")
        out = filters.apply_sos(tone_500, sos, zero_phase=True)
        # zero-phase filtering of a pure tone preserves amplitude & phase
        interior = slice(200, -200)
        assert np.allclose(
            np.angle(out[interior] * np.conj(tone_500[interior])), 0.0, atol=1e-4
        )

    def test_iir_types(self):
        for ftype in ("butter", "cheby1", "cheby2", "ellip"):
            sos = filters.iir_sos(4, 500.0, 8000.0, "lowpass", ftype=ftype)
            assert sos.ndim == 2

    def test_resample(self):
        t = np.arange(8000) / 8000.0
        x = np.exp(1j * 2 * np.pi * 500.0 * t)
        up, down = 3, 2  # 8000 -> 12000
        resampled = filters.resample(x, up, down)
        assert abs(resampled.size * 2 / 3 - x.size) <= 2
        freqs, psd = spectral.welch_psd(resampled, 12000.0)
        assert abs(freqs[np.argmax(psd)] - 500.0) < 20.0

    def test_resample_to_rate(self):
        t = np.arange(8000) / 8000.0
        x = np.exp(1j * 2 * np.pi * 500.0 * t)
        out, achieved = filters.resample_to_rate(x, 8000, 16000)
        assert abs(achieved - 16000) < 1.0
        assert abs(out.size - 2 * x.size) <= 2

    def test_invalid_cutoff(self):
        with pytest.raises(ValueError):
            filters.fir_lowpass(65, 5000.0, 8000.0)  # above Nyquist
        with pytest.raises(ValueError):
            filters.fir_bandpass(65, 900.0, 800.0, 8000.0)  # inverted


class TestBaseband:
    def test_analytic_signal(self):
        t = np.arange(8000) / 8000.0
        real = np.cos(2 * np.pi * 500.0 * t)
        analytic = baseband.analytic_signal(real)
        assert np.allclose(np.abs(analytic), 1.0, atol=0.02)

    def test_instantaneous_frequency(self, tone_500):
        inst = baseband.instantaneous_frequency(tone_500, 8000)
        assert abs(np.median(inst) - 500.0) < 1.0

    def test_mix_downconvert(self):
        t = np.arange(8000) / 8000.0
        at_500 = np.exp(1j * 2 * np.pi * 500.0 * t)
        at_1200 = np.exp(1j * 2 * np.pi * 1200.0 * t)
        combined = at_500 + at_1200
        baseband_signal = baseband.mix(combined, 500.0, 8000.0)
        freqs, psd = spectral.welch_psd(baseband_signal, 8000)
        peak = freqs[np.argmax(psd)]
        # after removing 500 Hz, the 1200 Hz component sits at 700 Hz
        assert abs(peak - 700.0) < 20.0

    def test_iq_imbalance_correction(self):
        rng = np.random.default_rng(5)
        # QPSK-like random symbols
        bits = rng.integers(0, 2, (4096, 2))
        const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        symbols = const[bits[:, 0] * 2 + bits[:, 1]]
        # apply known gain/phase imbalance
        gain = 10 ** (3.0 / 20.0)
        theta = np.deg2rad(10.0)
        i = gain * np.real(symbols)
        q = np.real(symbols) * np.sin(theta) + np.imag(symbols) * np.cos(theta)
        impaired = i + 1j * q
        corrected, imbalance = baseband.correct_iq_imbalance(impaired)
        # after correction, I and Q should be nearly uncorrelated
        corr = np.corrcoef(np.real(corrected), np.imag(corrected))[0, 1]
        assert abs(corr) < 0.05
        assert imbalance.gain_imbalance_db == pytest.approx(3.0, abs=1.0)
        assert imbalance.phase_imbalance_deg == pytest.approx(10.0, abs=3.0)

    def test_clipping_detection(self):
        x = np.array([0.5, 1.0, 1.5, 0.2, -1.2])
        report = baseband.detect_clipping(x, level=1.0)
        assert report["clipped_samples"] == 3
        assert report["peak_amplitude"] == pytest.approx(1.5)


class TestCorrelation:
    def test_xcorr_peak_at_known_lag(self):
        rng = np.random.default_rng(1)
        template = rng.standard_normal(64) + 0j
        signal = np.zeros(512, dtype=complex)
        signal[100:164] = template * 2.0  # amplitude change
        lags, values = correlation.xcorr(signal, template)
        assert lags[np.argmax(np.abs(values))] == 100

    def test_normalized_xcorr_amplitude_invariant(self):
        rng = np.random.default_rng(1)
        template = rng.standard_normal(64) + 0j
        signal = np.zeros(512, dtype=complex)
        signal[100:164] = template * 7.0
        _, values = correlation.normalized_xcorr(signal, template)
        assert values.max() == pytest.approx(1.0, abs=0.02)

    def test_autocorrelation_zero_lag(self):
        t = np.arange(4096) / 8000.0
        x = np.exp(1j * 2 * np.pi * 500.0 * t)
        ac = correlation.autocorrelation(x, 16)
        assert ac[0] == pytest.approx(1.0, rel=1e-6)

    def test_find_sync_word(self):
        rng = np.random.default_rng(2)
        preamble = np.exp(1j * 2 * np.pi * 100.0 * np.arange(32) / 8000.0)
        signal = rng.standard_normal(600) + 1j * rng.standard_normal(600)
        signal = signal * 0.3
        signal[200:232] += preamble
        matches = correlation.find_sync_word(signal, preamble, threshold=0.8)
        assert matches, "expected the planted preamble to be found"
        assert matches[0]["index"] == pytest.approx(200, abs=3)

    def test_sync_word_absent(self):
        rng = np.random.default_rng(2)
        preamble = rng.standard_normal(32) + 0j
        signal = rng.standard_normal(600) + 1j * rng.standard_normal(600)
        matches = correlation.find_sync_word(signal, preamble, threshold=0.95)
        assert matches == []
