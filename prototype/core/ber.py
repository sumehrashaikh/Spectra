"""Reference-bit loading and BER validation for reproducible test signals."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REFERENCE_SUFFIX = ".reference.npz"


@dataclass(frozen=True)
class BERValidation:
    """Result of comparing recovered BPSK bits with a saved reference."""

    reference_path: Path
    recovered_bit_count: int
    reference_bit_count: int
    compared_bit_count: int
    length_match: bool
    direct_bit_errors: int
    inverted_bit_errors: int
    bit_errors: int
    ber: float
    polarity_inverted: bool

    @property
    def polarity_label(self):
        return "inverted (180-degree ambiguity resolved)" if self.polarity_inverted else "direct"


def reference_path_for_wav(wav_path):
    """Return the standard saved-reference path associated with a WAV file."""
    path = Path(wav_path)
    return path.with_name(f"{path.stem}{REFERENCE_SUFFIX}")


def load_transmitted_bits(wav_path):
    """Load the generator's companion bit reference, if one is available."""
    reference_path = reference_path_for_wav(wav_path)
    if not reference_path.is_file():
        return None, reference_path

    try:
        with np.load(reference_path, allow_pickle=False) as reference:
            if "bits" not in reference:
                raise ValueError("reference is missing its 'bits' array")
            bits = np.asarray(reference["bits"], dtype=np.uint8).reshape(-1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read BER reference '{reference_path.name}': {exc}") from exc

    if bits.size == 0 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError(f"BER reference '{reference_path.name}' must contain non-empty binary bits.")

    return bits, reference_path


def validate_bpsk_bits(recovered_bits, reference_bits, reference_path):
    """Calculate BER, testing both BPSK polarities against the reference.

    A blind BPSK receiver cannot distinguish a constellation from its 180-degree
    rotation.  Therefore both local hard-decision polarities are evaluated and
    the lower-error polarity is reported explicitly.
    """
    recovered = np.asarray(recovered_bits, dtype=np.uint8).reshape(-1)
    reference = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
    if recovered.size == 0 or reference.size == 0:
        raise ValueError("BER validation requires non-empty recovered and reference bit streams.")
    if not np.all((recovered == 0) | (recovered == 1)):
        raise ValueError("Recovered BPSK decisions must be binary.")

    compared = min(recovered.size, reference.size)
    recovered = recovered[:compared]
    reference = reference[:compared]
    direct_errors = int(np.count_nonzero(recovered != reference))
    inverted_errors = int(np.count_nonzero((recovered ^ 1) != reference))
    polarity_inverted = inverted_errors < direct_errors
    bit_errors = inverted_errors if polarity_inverted else direct_errors

    return BERValidation(
        reference_path=Path(reference_path),
        recovered_bit_count=int(len(recovered_bits)),
        reference_bit_count=int(len(reference_bits)),
        compared_bit_count=compared,
        length_match=len(recovered_bits) == len(reference_bits),
        direct_bit_errors=direct_errors,
        inverted_bit_errors=inverted_errors,
        bit_errors=bit_errors,
        ber=float(bit_errors / compared),
        polarity_inverted=polarity_inverted,
    )
