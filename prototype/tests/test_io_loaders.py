"""Tests for the acquisition layer (io/loaders.py)."""

import numpy as np
import pytest

from prototype.core.exceptions import (
    LoaderError,
    SidecarError,
    UnsupportedFormatError,
)
from prototype.io.loaders import (
    load_iq_pair,
    load_raw_iq,
    load_signal,
    load_sidecar,
    load_wav_signal,
    save_iq,
    sidecar_path_for,
    stream_raw_iq,
    write_sidecar,
)


def _synth(num_samples=4096, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples) / 8000.0
    return (
        0.5 * np.exp(1j * 2 * np.pi * 500.0 * t)
        + 0.05 * rng.standard_normal(num_samples)
    )


class TestRawIq:
    def test_complex64_roundtrip(self, tmp_path):
        samples = _synth()
        path = tmp_path / "capture.c64"
        save_iq(path, samples, dtype="complex64")
        loaded = load_raw_iq(path, sample_rate=8000, dtype="complex64")
        assert loaded.sample_rate == 8000
        assert loaded.num_samples == samples.size
        # complex32 component precision: ~1e-7
        assert np.allclose(loaded.samples, samples, atol=1e-6)
        assert loaded.metadata["capture"]["dtype"].endswith("complex64")

    def test_int16_roundtrip(self, tmp_path):
        samples = _synth(2048)
        path = tmp_path / "capture.i16"
        save_iq(path, samples, dtype="int16")
        loaded = load_raw_iq(path, sample_rate=8000, dtype="int16")
        assert loaded.num_samples == samples.size
        # quantization: |error| <= 1/2 LSB per component (1/32768)
        # combined with save-side rounding -> worst case ~1.5 LSB complex
        assert np.allclose(loaded.samples, samples, atol=6e-5)

    def test_int8_uint8_roundtrip(self, tmp_path):
        samples = _synth(1024)
        s8 = tmp_path / "c.i8"
        u8 = tmp_path / "c.u8"
        save_iq(s8, samples, dtype="int8")
        save_iq(u8, samples, dtype="uint8")
        l8 = load_raw_iq(s8, sample_rate=8000, dtype="int8")
        lu8 = load_raw_iq(u8, sample_rate=8000, dtype="uint8")
        assert l8.num_samples == samples.size
        assert lu8.num_samples == samples.size
        assert np.allclose(l8.samples, samples, atol=0.01)
        assert np.allclose(lu8.samples, samples, atol=0.02)

    def test_endianness(self, tmp_path):
        samples = _synth(512)
        # float32 interleaved, big-endian
        interleaved = np.empty(2 * samples.size, dtype=">f4")
        interleaved[0::2] = np.real(samples).astype(np.float32)
        interleaved[1::2] = np.imag(samples).astype(np.float32)
        path = tmp_path / "be.f32"
        interleaved.tofile(path)
        loaded = load_raw_iq(path, sample_rate=8000, dtype="float32",
                             endianness="big")
        assert np.allclose(loaded.samples, samples, atol=1e-6)

    def test_qi_order(self, tmp_path):
        samples = _synth(256)
        path = tmp_path / "qi.c64"
        save_iq(path, samples, dtype="complex64")
        # Re-save as Q-first by swapping components.
        interleaved = np.empty(2 * samples.size, dtype=np.float32)
        interleaved[0::2] = np.imag(samples).astype(np.float32)
        interleaved[1::2] = np.real(samples).astype(np.float32)
        interleaved.tofile(path)
        loaded = load_raw_iq(path, sample_rate=8000, dtype="float32",
                             iq_order="qi")
        assert np.allclose(loaded.samples, samples, atol=1e-6)

    def test_offset_bytes_skips_header(self, tmp_path):
        samples = _synth(256)
        path = tmp_path / "hdr.c64"
        save_iq(path, samples, dtype="complex64")
        with open(path, "ab") as handle:
            pass
        header = b"HEADER" + b"\x00" * 58
        with open(path, "rb") as original:
            data = original.read()
        with open(path, "wb") as handle:
            handle.write(header)
            handle.write(data)
        loaded = load_raw_iq(path, sample_rate=8000, dtype="complex64",
                             offset_bytes=64)
        assert np.allclose(loaded.samples, samples, atol=1e-6)

    def test_max_samples_truncates(self, tmp_path):
        samples = _synth(4096)
        path = tmp_path / "trunc.c64"
        save_iq(path, samples, dtype="complex64")
        loaded = load_raw_iq(path, sample_rate=8000, dtype="complex64",
                             max_samples=1000)
        assert loaded.num_samples == 1000

    def test_missing_file(self, tmp_path):
        with pytest.raises(LoaderError):
            load_raw_iq(tmp_path / "nope.c64", sample_rate=8000)

    def test_truncated_file_raises(self, tmp_path):
        samples = _synth(256)
        path = tmp_path / "trunc2.c64"
        save_iq(path, samples, dtype="complex64")
        with open(path, "r+b") as handle:
            handle.truncate(3 * 8 + 4)  # not a multiple of 8
        with pytest.raises(LoaderError):
            load_raw_iq(path, sample_rate=8000, dtype="complex64")

    def test_unsupported_dtype(self, tmp_path):
        missing = tmp_path / "x.c64"
        # existing file so the dtype check is what fails, not the path
        missing.write_bytes(b"\x00" * 64)
        with pytest.raises(UnsupportedFormatError):
            load_raw_iq(missing, sample_rate=8000, dtype="complex256")

    def test_nan_rejected(self, tmp_path):
        samples = np.array([1 + 1j, np.nan + 0j, 0.5 - 0.5j])
        path = tmp_path / "nan.c64"
        # bypass save_iq validation deliberately
        samples.astype(np.complex64).tofile(path)
        with pytest.raises(LoaderError):
            load_raw_iq(path, sample_rate=8000, dtype="complex64")

    def test_streaming_chunks(self, tmp_path):
        samples = _synth(5000)
        path = tmp_path / "stream.c64"
        save_iq(path, samples, dtype="complex64")
        chunks = list(stream_raw_iq(path, chunk_samples=1000, dtype="complex64"))
        joined = np.concatenate(chunks)
        assert np.allclose(joined, samples, atol=1e-6)
        assert all(chunk.size <= 1000 for chunk in chunks)


class TestIqPair:
    def test_pair_loading(self, tmp_path):
        samples = _synth(512)
        i_path = tmp_path / "i.f32"
        q_path = tmp_path / "q.f32"
        np.real(samples).astype(np.float32).tofile(i_path)
        np.imag(samples).astype(np.float32).tofile(q_path)
        loaded = load_iq_pair(i_path, q_path, sample_rate=8000, dtype="float32")
        assert np.allclose(loaded.samples, samples, atol=1e-6)

    def test_size_mismatch_rejected(self, tmp_path):
        samples = _synth(512)
        i_path = tmp_path / "i2.f32"
        q_path = tmp_path / "q2.f32"
        np.real(samples).astype(np.float32).tofile(i_path)
        np.imag(samples)[:100].astype(np.float32).tofile(q_path)
        with pytest.raises(LoaderError):
            load_iq_pair(i_path, q_path, sample_rate=8000)


class TestSidecar:
    def test_write_and_load(self, tmp_path):
        capture = tmp_path / "cap.c64"
        capture.write_bytes(b"\x00" * 64)
        sidecar = write_sidecar(capture, {
            "sample_rate": 1000,
            "dtype": "complex64",
            "center_frequency_hz": 433.5e6,
        })
        assert sidecar.name == "cap.meta.json"
        data = load_sidecar(capture)
        assert data["sample_rate"] == 1000

    def test_load_signal_uses_sidecar(self, tmp_path):
        samples = _synth(512)
        capture = tmp_path / "auto.c64"
        save_iq(capture, samples, dtype="complex64")
        write_sidecar(capture, {"sample_rate": 8000, "dtype": "complex64"})
        loaded = load_signal(capture)
        assert loaded.sample_rate == 8000
        assert np.allclose(loaded.samples, samples, atol=1e-6)

    def test_raw_without_rate_fails(self, tmp_path):
        samples = _synth(64)
        capture = tmp_path / "norate.c64"
        save_iq(capture, samples, dtype="complex64")
        with pytest.raises(LoaderError):
            load_signal(capture)

    def test_invalid_sidecar_json(self, tmp_path):
        capture = tmp_path / "bad.c64"
        capture.write_bytes(b"\x00" * 16)
        sidecar_path_for(capture).write_text("{not json", encoding="utf-8")
        with pytest.raises(SidecarError):
            load_sidecar(capture)

    def test_missing_sidecar(self, tmp_path):
        with pytest.raises(SidecarError):
            load_sidecar(tmp_path / "absent.c64")


class TestWav:
    def test_stereo_wav_iq(self, tmp_path):
        from scipy.io import wavfile

        samples = _synth(2048)
        rate = 8000
        interleaved = np.empty((samples.size, 2), dtype=np.float32)
        interleaved[:, 0] = np.real(samples)
        interleaved[:, 1] = np.imag(samples)
        wav_path = tmp_path / "iq.wav"
        wavfile.write(wav_path, rate, interleaved)

        loaded = load_wav_signal(wav_path)
        assert loaded.sample_rate == rate
        assert np.allclose(loaded.samples, samples, atol=1e-6)

    def test_int16_wav(self, tmp_path):
        from scipy.io import wavfile

        rate = 8000
        rng = np.random.default_rng(3)
        data = (rng.standard_normal((4096, 2)) * 8000).astype(np.int16)
        wav_path = tmp_path / "int16.wav"
        wavfile.write(wav_path, rate, data)

        loaded = load_wav_signal(wav_path)
        assert loaded.sample_rate == rate
        assert loaded.num_samples == 4096
        assert np.isfinite(loaded.samples).all()

    def test_missing_wav(self, tmp_path):
        with pytest.raises(LoaderError):
            load_wav_signal(tmp_path / "none.wav")
