from dataclasses import dataclass
from typing import Any

import numpy as np

from prototype.core.signal import Signal

from prototype.modulation.demodulator import (
    demodulate_qpsk,
    demodulate_qam16,
    demodulate_bfsk,
    qpsk_decision,
    refine_qam16_phase,
    calculate_ber as v1_calculate_ber,
)


@dataclass
class DemodulationResult:
    """
    Result produced by the V2 demodulation layer.
    """

    modulation: str
    bits: np.ndarray
    num_bits: int
    num_symbols: int
    decision_margin: float
    metadata: dict[str, Any]


def calculate_ber(
    recovered_bits,
    reference_bits,
) -> dict[str, Any]:
    """
    Calculate BER using the existing V1 BER implementation.
    """

    return v1_calculate_ber(
        recovered_bits,
        reference_bits,
    )


def _validate_signal(signal: Signal):
    """
    Validate the input Signal object.
    """

    if not isinstance(signal, Signal):
        raise TypeError(
            "demodulation expects a Signal object."
        )


def _validate_sps(samples_per_symbol):
    """
    Validate samples-per-symbol value.
    """

    if samples_per_symbol <= 0:
        raise ValueError(
            "samples_per_symbol must be positive."
        )


def demodulate_bpsk(
    samples,
) -> dict[str, Any]:
    """
    Hard-decision BPSK demodulation.

    Mapping:

        +I -> 1
        -I -> 0

    The input is expected to contain recovered
    symbol-rate samples.
    """

    samples = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if len(samples) == 0:
        raise ValueError(
            "No symbols supplied."
        )

    bits = (
        samples.real >= 0
    ).astype(np.uint8)

    decision_symbols = np.where(
        bits == 1,
        1.0,
        -1.0,
    ).astype(np.complex128)

    margins = np.abs(
        samples.real
    )

    return {
        "bits": bits,
        "decision_symbols": decision_symbols,
        "decision_margin": float(
            np.mean(margins)
        ),
        "num_symbols": int(
            len(bits)
        ),
        "num_bits": int(
            len(bits)
        ),
    }


def _qpsk_decide_with_rotation(
    symbols,
    rotation_rad,
):
    """
    Apply a known QPSK phase rotation and make
    Gray-coded hard decisions.

    This is used to resolve the 90-degree QPSK
    phase ambiguity left by blind carrier recovery.
    """

    rotated = (
        np.asarray(
            symbols,
            dtype=np.complex128,
        )
        * np.exp(-1j * rotation_rad)
    )

    bits, decision_symbols, margin = (
        qpsk_decision(rotated)
    )

    return (
        bits,
        decision_symbols,
        margin,
        rotated,
    )


def demodulate_qpsk_synchronized(
    samples,
) -> dict[str, Any]:
    """
    Demodulate already synchronized QPSK symbols.

    Carrier and timing synchronization, including any known
    preamble-based phase correction, have already been performed.

    Therefore this function performs only the QPSK symbol decision.
    It does NOT attempt another blind phase-ambiguity resolution.
    """

    symbols = np.asarray(
        samples,
        dtype=np.complex128,
    )

    if len(symbols) == 0:
        raise ValueError(
            "No synchronized QPSK symbols supplied."
        )

    bits, decision_symbols, margin = qpsk_decision(
        symbols
    )

    return {
        "bits": bits,
        "decision_symbols": decision_symbols,
        "decision_margin": float(margin),
        "num_symbols": int(len(symbols)),
        "num_bits": int(len(bits)),
        "phase_ambiguity_rotation_rad": 0.0,
        "phase_ambiguity_rotation_deg": 0.0,
        "corrected_symbols": symbols,
    }

def demodulate_signal(
    signal: Signal,
    modulation: str,
    samples_per_symbol: float | None = None,
    timing_offset: float = 0.0,
    freq_0: float | None = None,
    freq_1: float | None = None,
    synchronized: bool = False,
) -> DemodulationResult:
    """
    Main V2 demodulation entry point.

    Parameters
    ----------
    signal:
        V2 Signal object.

    modulation:
        Detected modulation type.

    samples_per_symbol:
        Samples per symbol for symbol-based modulation.

    timing_offset:
        Recovered timing offset.

    freq_0, freq_1:
        BFSK frequencies.

    synchronized:
        If True, the input already contains recovered
        symbol-rate samples and the demodulator must
        not repeat carrier/timing recovery.
    """

    _validate_signal(signal)

    modulation = str(
        modulation
    ).strip()

    # ---------------------------------------------------------
    # BPSK
    # ---------------------------------------------------------

    if modulation == "BPSK":

        if samples_per_symbol is not None:

            _validate_sps(
                samples_per_symbol
            )

            from prototype.modulation.demodulator import (
                extract_symbols,
                normalize_symbols,
            )

            symbols = extract_symbols(
                signal.samples,
                samples_per_symbol,
                timing_offset,
            )

            symbols = normalize_symbols(
                symbols
            )

        else:

            symbols = signal.samples

        result = demodulate_bpsk(
            symbols
        )

        return DemodulationResult(
            modulation="BPSK",
            bits=result["bits"],
            num_bits=result["num_bits"],
            num_symbols=result["num_symbols"],
            decision_margin=result[
                "decision_margin"
            ],
            metadata={
                "decision_symbols": result[
                    "decision_symbols"
                ],
                "samples_per_symbol": (
                    None
                    if samples_per_symbol is None
                    else float(
                        samples_per_symbol
                    )
                ),
                "timing_offset": float(
                    timing_offset
                ),
            },
        )

    # ---------------------------------------------------------
    # QPSK
    # ---------------------------------------------------------

    if modulation == "QPSK":

        if synchronized:

            result = (
                demodulate_qpsk_synchronized(
                    signal.samples
                )
            )

            return DemodulationResult(
                modulation="QPSK",
                bits=result["bits"],
                num_bits=result["num_bits"],
                num_symbols=result["num_symbols"],
                decision_margin=result[
                    "decision_margin"
                ],
                metadata={
                    "decision_symbols": result[
                        "decision_symbols"
                    ],
                    "corrected_symbols": result[
                        "corrected_symbols"
                    ],
                    "phase_ambiguity_rotation_rad":
                        result[
                            "phase_ambiguity_rotation_rad"
                        ],
                    "phase_ambiguity_rotation_deg":
                        result[
                            "phase_ambiguity_rotation_deg"
                        ],
                    "synchronized": True,
                },
            )

        if samples_per_symbol is None:
            raise ValueError(
                "QPSK requires samples_per_symbol."
            )

        _validate_sps(
            samples_per_symbol
        )

        result = demodulate_qpsk(
            signal.samples,
            samples_per_symbol,
            timing_offset,
        )

        return DemodulationResult(
            modulation="QPSK",
            bits=result["bits"],
            num_bits=result["num_bits"],
            num_symbols=result["num_symbols"],
            decision_margin=result[
                "decision_margin"
            ],
            metadata={
                "symbols": result[
                    "symbols"
                ],
                "corrected_symbols": result[
                    "corrected_symbols"
                ],
                "decision_symbols": result[
                    "decision_symbols"
                ],
                "phase_estimate_rad": result[
                    "phase_estimate_rad"
                ],
                "quadrature_balance": result[
                    "quadrature_balance"
                ],
                "samples_per_symbol": float(
                    samples_per_symbol
                ),
                "timing_offset": float(
                    timing_offset
                ),
                "synchronized": False,
            },
        )

    # ---------------------------------------------------------
    # 16-QAM
    # ---------------------------------------------------------

    if modulation == "16-QAM":

        if samples_per_symbol is None:
            raise ValueError(
                "16-QAM requires samples_per_symbol."
            )

        _validate_sps(
            samples_per_symbol
        )

        # The synchronized stream may still carry an arbitrary blind
        # carrier-phase rotation: the M-th power estimator is tuned for
        # constant-envelope PSK and is unreliable on amplitude-modulated
        # 16-QAM (observed ~30-degree errors even on clean captures).
        # Refine the phase decision-directed before symbol decisions.
        refined_samples, phase_rotation_rad, converged = refine_qam16_phase(
            signal.samples
        )

        result = demodulate_qam16(
            refined_samples,
            samples_per_symbol,
            timing_offset,
        )

        return DemodulationResult(
            modulation="16-QAM",
            bits=result["bits"],
            num_bits=result["num_bits"],
            num_symbols=result["num_symbols"],
            decision_margin=result[
                "decision_margin"
            ],
            metadata={
                "symbols": result[
                    "symbols"
                ],
                "decision_symbols": result[
                    "decision_symbols"
                ],
                "quadrature_balance": result[
                    "quadrature_balance"
                ],
                "samples_per_symbol": float(
                    samples_per_symbol
                ),
                "timing_offset": float(
                    timing_offset
                ),
                "phase_refinement_rad": float(
                    phase_rotation_rad
                ),
                "phase_refinement_converged": bool(
                    converged
                ),
            },
        )

    # ---------------------------------------------------------
    # BFSK
    # ---------------------------------------------------------

    if modulation == "BFSK":

        if samples_per_symbol is None:
            raise ValueError(
                "BFSK requires samples_per_symbol."
            )

        if (
            freq_0 is None
            or freq_1 is None
        ):
            raise ValueError(
                "BFSK requires freq_0 and freq_1."
            )

        _validate_sps(
            samples_per_symbol
        )

        result = demodulate_bfsk(
            signal.samples,
            signal.sample_rate,
            int(round(samples_per_symbol)),
            float(freq_0),
            float(freq_1),
        )

        return DemodulationResult(
            modulation="BFSK",
            bits=result["bits"].astype(
                np.uint8
            ),
            num_bits=int(
                result["num_symbols"]
            ),
            num_symbols=int(
                result["num_symbols"]
            ),
            decision_margin=result[
                "decision_margin"
            ],
            metadata={
                "decision_symbols": result[
                    "decision_symbols"
                ],
                "freq_0": float(
                    freq_0
                ),
                "freq_1": float(
                    freq_1
                ),
                "samples_per_symbol": float(
                    samples_per_symbol
                ),
            },
        )

    # ---------------------------------------------------------
    # 8-PSK
    # ---------------------------------------------------------

    if modulation == "8-PSK":

        from prototype.modulation.digital import demodulate_psk8

        if samples_per_symbol is None:
            raise ValueError(
                "8-PSK requires samples_per_symbol."
            )

        _validate_sps(
            samples_per_symbol
        )

        result = demodulate_psk8(
            signal.samples,
            samples_per_symbol,
            timing_offset,
        )

        return DemodulationResult(
            modulation="8-PSK",
            bits=result["bits"],
            num_bits=result["num_bits"],
            num_symbols=result["num_symbols"],
            decision_margin=result[
                "decision_margin"
            ],
            metadata={
                "symbols": result["symbols"],
                "decision_symbols": result[
                    "decision_symbols"
                ],
                "phase_estimate_rad": result[
                    "phase_estimate_rad"
                ],
                "samples_per_symbol": float(
                    samples_per_symbol
                ),
                "timing_offset": float(
                    timing_offset
                ),
            },
        )

    # ---------------------------------------------------------
    # OOK / ASK
    # ---------------------------------------------------------

    if modulation in ("OOK", "ASK"):

        from prototype.modulation.digital import demodulate_ook

        if samples_per_symbol is None:
            raise ValueError(
                "OOK requires samples_per_symbol."
            )

        _validate_sps(
            samples_per_symbol
        )

        result = demodulate_ook(
            signal.samples,
            samples_per_symbol,
            timing_offset,
        )

        return DemodulationResult(
            modulation=modulation,
            bits=result["bits"],
            num_bits=result["num_bits"],
            num_symbols=result["num_symbols"],
            decision_margin=result[
                "decision_margin"
            ],
            metadata={
                "symbols": result["symbols"],
                "decision_symbols": result[
                    "decision_symbols"
                ],
                "samples_per_symbol": float(
                    samples_per_symbol
                ),
                "timing_offset": float(
                    timing_offset
                ),
            },
        )

    # ---------------------------------------------------------
    # Unknown
    # ---------------------------------------------------------

    if modulation == "Unknown":

        raise ValueError(
            "Cannot demodulate an Unknown modulation."
        )

    raise ValueError(
        f"Unsupported modulation: {modulation}"
    )