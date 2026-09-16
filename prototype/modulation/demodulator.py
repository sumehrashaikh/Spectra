import numpy as np


def extract_symbols(samples, samples_per_symbol, timing_offset):
    """
    Extract one complex sample per symbol using recovered timing.

    Parameters
    ----------
    samples : complex ndarray
        Isolated IQ samples.

    samples_per_symbol : float
        Estimated samples per symbol.

    timing_offset : float
        Estimated timing offset.

    Returns
    -------
    symbols : complex ndarray
        One complex sample per recovered symbol.
    """

    if samples is None or len(samples) == 0:
        raise ValueError("Signal is empty.")

    if samples_per_symbol <= 0:
        raise ValueError("Samples per symbol must be positive.")

    if timing_offset < 0:
        raise ValueError("Timing offset must be non-negative.")

    samples = np.asarray(samples)

    indices = []
    position = float(timing_offset)

    while position < len(samples):
        index = int(round(position))

        if index >= len(samples):
            break

        indices.append(index)
        position += float(samples_per_symbol)

    if len(indices) == 0:
        raise ValueError("No symbols could be extracted.")

    return samples[np.asarray(indices, dtype=int)]


def normalize_symbols(symbols):
    """
    Normalize recovered symbols by their RMS magnitude.
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if len(symbols) == 0:
        raise ValueError("No symbols supplied.")

    rms = np.sqrt(np.mean(np.abs(symbols) ** 2))

    if rms < 1e-12:
        raise ValueError("Symbol signal has insufficient energy.")

    return symbols / rms


def estimate_qpsk_phase(symbols):
    """
    Estimate QPSK carrier phase using the fourth-power method.

    QPSK removes modulation information approximately because
    raising a QPSK symbol to the fourth power produces nearly
    the same phase for every ideal symbol.

    Returns
    -------
    phase : float
        Estimated phase rotation in radians.
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if len(symbols) == 0:
        raise ValueError("No symbols supplied.")

    fourth_power = symbols ** 4

    mean_value = np.mean(fourth_power)

    if abs(mean_value) < 1e-12:
        return 0.0

    phase = np.angle(mean_value)

    # QPSK constellation is at odd multiples of pi/4.
    phase = (phase - np.pi) / 4.0

    return float(phase)


def rotate_symbols(symbols, phase):
    """
    Remove an estimated carrier phase rotation.
    """

    return np.asarray(symbols) * np.exp(-1j * phase)


def qpsk_decision(symbols):
    """
    Make Gray-coded QPSK hard decisions.

    Mapping:

        +I, +Q -> 00
        -I, +Q -> 01
        -I, -Q -> 11
        +I, -Q -> 10

    Returns
    -------
    bits : ndarray
        Recovered 0/1 bits.

    decided_symbols : ndarray
        Ideal QPSK constellation points.

    margin : float
        Average decision margin.
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if len(symbols) == 0:
        raise ValueError("No symbols supplied.")

    bits = []
    decided = []
    margins = []

    for symbol in symbols:

        i = symbol.real
        q = symbol.imag

        if i >= 0 and q >= 0:
            pair = (0, 0)
            ideal = 1 + 1j

        elif i < 0 and q >= 0:
            pair = (0, 1)
            ideal = -1 + 1j

        elif i < 0 and q < 0:
            pair = (1, 1)
            ideal = -1 - 1j

        else:
            pair = (1, 0)
            ideal = 1 - 1j

        bits.extend(pair)
        decided.append(ideal)

        distance_i = abs(i)
        distance_q = abs(q)

        margins.append(
            min(distance_i, distance_q)
        )

    bits = np.asarray(bits, dtype=np.uint8)
    decided = np.asarray(decided, dtype=np.complex128)

    margin = float(np.mean(margins))

    return bits, decided, margin


def demodulate_qpsk(
    samples,
    samples_per_symbol,
    timing_offset
):
    """
    Complete baseline QPSK symbol extraction and demodulation.

    Returns a dictionary containing:

    - recovered symbols
    - corrected symbols
    - recovered bits
    - decision symbols
    - phase estimate
    - decision margin
    - quadrature balance
    """

    symbols = extract_symbols(
        samples,
        samples_per_symbol,
        timing_offset
    )

    symbols = normalize_symbols(symbols)

    phase = estimate_qpsk_phase(symbols)

    corrected = rotate_symbols(
        symbols,
        phase
    )

    bits, decided, margin = qpsk_decision(
        corrected
    )

    mean_i = np.mean(np.abs(corrected.real))
    mean_q = np.mean(np.abs(corrected.imag))

    quadrature_ratio = (
        min(mean_i, mean_q)
        /
        (max(mean_i, mean_q) + 1e-12)
    )

    return {
        "symbols": symbols,
        "corrected_symbols": corrected,
        "decision_symbols": decided,
        "bits": bits,
        "phase_estimate_rad": phase,
        "decision_margin": margin,
        "quadrature_balance": float(quadrature_ratio),
        "num_symbols": int(len(corrected)),
        "num_bits": int(len(bits)),
    }


def calculate_ber(recovered_bits, reference_bits):
    """
    Calculate BER between recovered and reference bits.
    """

    recovered_bits = np.asarray(
        recovered_bits,
        dtype=np.uint8
    )

    reference_bits = np.asarray(
        reference_bits,
        dtype=np.uint8
    )

    count = min(
        len(recovered_bits),
        len(reference_bits)
    )

    if count == 0:
        raise ValueError("No bits available for BER calculation.")

    recovered_bits = recovered_bits[:count]
    reference_bits = reference_bits[:count]

    direct_errors = int(
        np.sum(
            recovered_bits != reference_bits
        )
    )

    inverted_errors = int(
        np.sum(
            (1 - recovered_bits) != reference_bits
        )
    )

    direct_ber = direct_errors / count
    inverted_ber = inverted_errors / count

    return {
        "compared_bits": count,
        "direct_errors": direct_errors,
        "inverted_errors": inverted_errors,
        "direct_ber": float(direct_ber),
        "inverted_ber": float(inverted_ber),
    }

# ============================================================
# 16-QAM DEMODULATION
# ============================================================

def qam16_decision(symbols):
    """
    Hard-decision Gray-coded 16-QAM demodulator.

    Mapping on each axis:

        00 -> -3
        01 -> -1
        11 -> +1
        10 -> +3

    Each complex symbol therefore carries 4 bits:
        I bits + Q bits
    """

    symbols = np.asarray(
        symbols,
        dtype=np.complex128
    )

    if len(symbols) == 0:
        raise ValueError(
            "No symbols supplied."
        )

    # --------------------------------------------------------
    # Normalized 16-QAM constellation
    #
    # Original levels:
    #     -3, -1, +1, +3
    #
    # Generator normalizes by sqrt(10).
    # Therefore we make the decisions using the
    # normalized levels.
    # --------------------------------------------------------

    levels = np.array(
        [-3, -1, 1, 3],
        dtype=float
    ) / np.sqrt(10.0)

    gray_bits = {
        -3: (0, 0),
        -1: (0, 1),
         1: (1, 1),
         3: (1, 0),
    }

    bits = []

    decided = []

    margins = []

    for symbol in symbols:

        i = float(
            np.real(symbol)
        )

        q = float(
            np.imag(symbol)
        )

        # ----------------------------------------------------
        # Find nearest I level
        # ----------------------------------------------------

        i_distances = np.abs(
            levels - i
        )

        i_index = int(
            np.argmin(
                i_distances
            )
        )

        i_level = int(
            [-3, -1, 1, 3][i_index]
        )

        # ----------------------------------------------------
        # Find nearest Q level
        # ----------------------------------------------------

        q_distances = np.abs(
            levels - q
        )

        q_index = int(
            np.argmin(
                q_distances
            )
        )

        q_level = int(
            [-3, -1, 1, 3][q_index]
        )

        # ----------------------------------------------------
        # Convert levels to Gray-coded bits
        # ----------------------------------------------------

        i_bits = gray_bits[
            i_level
        ]

        q_bits = gray_bits[
            q_level
        ]

        bits.extend(
            i_bits
        )

        bits.extend(
            q_bits
        )

        # ----------------------------------------------------
        # Reconstruct ideal constellation point
        # ----------------------------------------------------

        ideal = (
            i_level
            + 1j * q_level
        ) / np.sqrt(10.0)

        decided.append(
            ideal
        )

        # ----------------------------------------------------
        # Decision margin
        #
        # Distance from actual symbol to the
        # nearest axis level.
        # ----------------------------------------------------

        i_margin = np.min(
            i_distances
        )

        q_margin = np.min(
            q_distances
        )

        margins.append(
            min(
                i_margin,
                q_margin
            )
        )

    return (
        np.asarray(
            bits,
            dtype=np.uint8
        ),
        np.asarray(
            decided,
            dtype=np.complex128
        ),
        float(
            np.mean(margins)
        )
    )


def normalize_qam16_symbols(symbols):
    """
    Normalize received 16-QAM symbols using RMS energy.
    """

    symbols = np.asarray(
        symbols,
        dtype=np.complex128
    )

    if len(symbols) == 0:
        raise ValueError(
            "No symbols supplied."
        )

    rms = np.sqrt(
        np.mean(
            np.abs(symbols) ** 2
        )
    )

    if rms < 1e-12:
        raise ValueError(
            "Symbol signal has insufficient energy."
        )

    return symbols / rms


def demodulate_qam16(
    samples,
    samples_per_symbol,
    timing_offset
):
    """
    Complete baseline 16-QAM symbol extraction
    and demodulation.
    """

    symbols = extract_symbols(
        samples,
        samples_per_symbol,
        timing_offset
    )

    # --------------------------------------------------------
    # Normalize received symbols
    # --------------------------------------------------------

    symbols = normalize_qam16_symbols(
        symbols
    )

    # --------------------------------------------------------
    # Hard decision
    # --------------------------------------------------------

    bits, decided, margin = qam16_decision(
        symbols
    )

    # --------------------------------------------------------
    # Calculate I/Q balance
    # --------------------------------------------------------

    mean_i = np.mean(
        np.abs(
            symbols.real
        )
    )

    mean_q = np.mean(
        np.abs(
            symbols.imag
        )
    )

    quadrature_balance = (
        min(
            mean_i,
            mean_q
        )
        /
        (
            max(
                mean_i,
                mean_q
            )
            + 1e-12
        )
    )

    return {
        "symbols":
            symbols,

        "decision_symbols":
            decided,

        "bits":
            bits,

        "decision_margin":
            margin,

        "quadrature_balance":
            float(
                quadrature_balance
            ),

        "num_symbols":
            int(
                len(symbols)
            ),

        "num_bits":
            int(
                len(bits)
            ),
    }

def demodulate_bfsk(samples, sample_rate, samples_per_symbol, freq_0, freq_1):
    """
    Coherent BFSK demodulation.

    Each symbol is classified by comparing its correlation
    with the two BFSK frequencies.

    Returns:
        bits              - recovered 0/1 bits
        decision_symbols  - selected frequency for each symbol
        decision_margin   - average normalized separation
    """

    samples = np.asarray(samples, dtype=complex)

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")

    num_symbols = len(samples) // samples_per_symbol

    if num_symbols == 0:
        raise ValueError("Not enough samples for one symbol")

    bits = []
    decision_symbols = []
    margins = []

    for symbol_index in range(num_symbols):
        start = symbol_index * samples_per_symbol
        end = start + samples_per_symbol

        symbol = samples[start:end]

        t = np.arange(len(symbol)) / sample_rate

        ref_0 = np.exp(-1j * 2 * np.pi * freq_0 * t)
        ref_1 = np.exp(-1j * 2 * np.pi * freq_1 * t)

        correlation_0 = abs(np.sum(symbol * ref_0))
        correlation_1 = abs(np.sum(symbol * ref_1))

        if correlation_0 >= correlation_1:
            bit = 0
            selected_frequency = freq_0
            winning = correlation_0
            losing = correlation_1
        else:
            bit = 1
            selected_frequency = freq_1
            winning = correlation_1
            losing = correlation_0

        margin = (winning - losing) / max(winning + losing, 1e-12)

        bits.append(bit)
        decision_symbols.append(selected_frequency)
        margins.append(margin)

    bits = np.asarray(bits, dtype=int)
    decision_symbols = np.asarray(decision_symbols, dtype=float)

    return {
        "bits": bits,
        "decision_symbols": decision_symbols,
        "decision_margin": float(np.mean(margins)),
        "num_symbols": len(bits),
    }
