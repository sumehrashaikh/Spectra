"""Tests for the FEC package (fec/)."""

import numpy as np
import pytest

from prototype.core.exceptions import FECError
from prototype.fec import crc, hamming, repetition, convolutional
from prototype.fec import (
    decode_bits,
    encode_bits,
    list_schemes,
    describe_scheme,
)
from prototype.fec.interleaving import deinterleave_bits, interleave_bits


@pytest.fixture
def bits_400():
    rng = np.random.default_rng(7)
    return rng.integers(0, 2, 400).astype(np.uint8)


class TestCrc:
    def test_crc16_roundtrip(self, bits_400):
        appended = crc.crc_append(bits_400, "crc16")
        assert appended.size == bits_400.size + 16
        assert crc.crc_check(appended, "crc16")

    def test_crc16_detects_error(self, bits_400):
        appended = crc.crc_append(bits_400, "crc16")
        corrupt = appended.copy()
        corrupt[100] ^= 1
        assert not crc.crc_check(corrupt, "crc16")

    def test_crc32_roundtrip(self, bits_400):
        appended = crc.crc_append(bits_400, "crc32")
        assert appended.size == bits_400.size + 32
        assert crc.crc_check(appended, "crc32")

    def test_crc32_detects_error(self, bits_400):
        appended = crc.crc_append(bits_400, "crc32")
        corrupt = appended.copy()
        corrupt[-1] ^= 1
        assert not crc.crc_check(corrupt, "crc32")

    def test_bits_bytes_roundtrip(self):
        bits = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        data = crc.bits_to_bytes(bits)
        assert np.array_equal(crc.bytes_to_bits(data, 8), bits)

    def test_unknown_scheme(self, bits_400):
        with pytest.raises(ValueError):
            crc.crc_append(bits_400, "crc7")

    def test_too_short_for_check(self):
        with pytest.raises(ValueError):
            crc.crc_check(np.array([1, 0], dtype=np.uint8), "crc16")


class TestHamming:
    def test_generator_parity_check_consistency(self):
        # every codeword must satisfy H @ c = 0
        for data in np.eye(4, dtype=np.int8):
            codeword = (data @ hamming.G) % 2
            syndrome = (hamming.H @ codeword) % 2
            assert not syndrome.any()

    def test_clean_roundtrip(self, bits_400):
        encoded = hamming.encode(bits_400)
        assert encoded.size == int(np.ceil(400 / 4)) * 7
        decoded, stats = hamming.decode(encoded)
        assert np.array_equal(decoded, bits_400)
        assert stats["corrected_errors"] == 0

    def test_single_error_corrected(self, bits_400):
        encoded = hamming.encode(bits_400)
        corrupted = encoded.copy()
        corrupted[0] ^= 1
        decoded, stats = hamming.decode(corrupted)
        assert np.array_equal(decoded, bits_400)
        assert stats["corrected_errors"] == 1

    def test_every_single_bit_error_corrected(self):
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        encoded = hamming.encode(data)
        for position in range(encoded.size):
            corrupted = encoded.copy()
            corrupted[position] ^= 1
            decoded, _ = hamming.decode(corrupted)
            assert np.array_equal(decoded, data), f"position {position}"

    def test_non_multiple_of_seven_rejected(self):
        with pytest.raises(ValueError):
            hamming.decode(np.zeros(10, dtype=np.uint8))

    def test_nonbinary_rejected(self, bits_400):
        with pytest.raises(ValueError):
            hamming.encode(bits_400 + 2)


class TestRepetition:
    def test_roundtrip(self, bits_400):
        encoded = repetition.encode(bits_400, repetitions=3)
        assert encoded.size == 1200
        decoded, _ = repetition.decode(encoded, repetitions=3)
        assert np.array_equal(decoded, bits_400)

    def test_majority_vote(self, bits_400):
        encoded = repetition.encode(bits_400, repetitions=3)
        corrupted = encoded.copy()
        corrupted[0] ^= 1  # one of three -> majority still correct
        decoded, _ = repetition.decode(corrupted, repetitions=3)
        assert np.array_equal(decoded, bits_400)


class TestConvolutional:
    def test_clean_roundtrip(self, bits_400):
        encoded = convolutional.encode(bits_400, tail=True)
        assert encoded.size == (bits_400.size + 6) * 2
        decoded, stats = convolutional.decode(encoded)
        assert np.array_equal(decoded, bits_400)
        assert stats["final_path_metric"] == 0.0

    def test_error_correction(self, bits_400):
        rng = np.random.default_rng(11)
        encoded = convolutional.encode(bits_400)
        for trial in range(5):
            corrupted = encoded.copy()
            for index in rng.integers(0, encoded.size, 16):
                corrupted[index] ^= 1
            decoded, _ = convolutional.decode(corrupted)
            assert np.array_equal(decoded, bits_400), f"trial {trial}"

    def test_odd_input_rejected(self):
        with pytest.raises(ValueError):
            convolutional.decode(np.zeros(7, dtype=np.uint8))

    def test_termination_to_zero_state(self, bits_400):
        encoded = convolutional.encode(np.concatenate([bits_400[:64]]))
        # with tail bits, the encoder must end in state 0
        state = 0
        stream = np.concatenate([bits_400[:64], np.zeros(6, dtype=np.uint8)])
        for bit in stream:
            state = convolutional._next_state(state, int(bit))
        assert state == 0


class TestInterleaving:
    def test_roundtrip(self, bits_400):
        interleaved = interleave_bits(bits_400, depth=8)
        assert interleaved.size == 400
        recovered = deinterleave_bits(interleaved, depth=8, original_size=400)
        assert np.array_equal(recovered, bits_400)

    def test_roundtrip_nonmultiple(self):
        bits = np.arange(1, 38, dtype=np.uint8) % 2
        recovered = deinterleave_bits(
            interleave_bits(bits, depth=4), depth=4, original_size=bits.size
        )
        assert np.array_equal(recovered, bits)

    def test_identity_for_depth_one(self, bits_400):
        assert np.array_equal(interleave_bits(bits_400, depth=1), bits_400)

    def test_changes_order(self, bits_400):
        interleaved = interleave_bits(bits_400, depth=8)
        assert not np.array_equal(interleaved, bits_400)


class TestFramework:
    def test_registry(self):
        schemes = sorted(list_schemes())
        assert "hamming74" in schemes
        assert "conv12" in schemes
        assert "repetition3" in schemes

    def test_describe(self):
        assert "Hamming" in describe_scheme("hamming74")

    def test_encode_decode_roundtrip_all_schemes(self, bits_400):
        for scheme in list_schemes():
            encoded, encode_result = encode_bits(bits_400, scheme)
            decoded, decode_result = decode_bits(encoded, scheme)
            assert np.array_equal(
                decoded[: bits_400.size], bits_400
            ) or np.array_equal(decoded, bits_400), scheme
            assert decode_result.corrected_errors >= 0

    def test_unknown_scheme_rejected(self, bits_400):
        with pytest.raises(FECError):
            encode_bits(bits_400, "turbo4")
        with pytest.raises(FECError):
            decode_bits(bits_400, "turbo4")
