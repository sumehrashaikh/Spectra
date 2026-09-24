"""
SNR sweep benchmarking.

Generates synthetic test signals (clearly labeled synthetic), applies
the channel simulator (AWGN; optionally CFO/timing/phase), runs the
full analysis pipeline, and reports per-SNR detection, classification,
symbol-rate, and BER results against ground truth.

These benchmarks characterize the *processing chain* on simulated
signals. They are NOT real-world, hardware, or certification evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prototype.simulation.channel import ChannelConfig, apply_channel


def generate_test_signal(
    modulation: str,
    num_symbols: int = 256,
    samples_per_symbol: int = 8,
    sample_rate: float = 8000.0,
    carrier_hz: float = 500.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a pulse-shaped synthetic test signal.

    Returns (samples_at_rf, reference_bits).
    """
    rng = np.random.default_rng(seed)
    from prototype.parameters.symbol_rate import rrc_filter

    if modulation in ("BPSK", "QPSK", "16-QAM"):
        bits_per_symbol = {"BPSK": 1, "QPSK": 2, "16-QAM": 4}[modulation]
        bits = rng.integers(0, 2, num_symbols * bits_per_symbol)

        if modulation == "BPSK":
            symbols = (2 * bits - 1).astype(np.complex128)
        elif modulation == "QPSK":
            pairs = bits.reshape(-1, 2)
            constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
            gray = np.array([0, 1, 3, 2])  # matches demodulator mapping
            symbols = constellation[gray[pairs[:, 0] * 2 + pairs[:, 1]]]
        else:
            levels = np.array([-3, -1, 1, 3], dtype=float)
            i = levels[bits[0::2]]
            q = levels[bits[1::2]]
            symbols = (i + 1j * q) / np.sqrt(10.0)

        sps = samples_per_symbol
        upsampled = np.zeros(symbols.size * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        shaped = np.convolve(upsampled, rrc_filter(sps), mode="same")
        t = np.arange(shaped.size) / sample_rate
        samples = shaped * np.exp(1j * 2 * np.pi * carrier_hz * t)
        return samples, bits

    raise ValueError(f"Unsupported benchmark modulation '{modulation}'.")


def _run_one(
    modulation: str,
    snr_db: float,
    num_symbols: int,
    seed: int,
    with_cfo: bool,
) -> dict[str, Any]:
    """Run one benchmark point; returns honest metrics with None for N/A."""
    from prototype.pipeline import analyze_samples

    sample_rate = 8000.0
    carrier_hz = 500.0

    samples, reference_bits = generate_test_signal(
        modulation, num_symbols=num_symbols, seed=seed
    )

    config_kwargs: dict[str, Any] = {"snr_db": float(snr_db), "seed": seed}
    if with_cfo:
        config_kwargs.update(
            frequency_offset_hz=25.0,
            phase_offset_rad=0.5,
            timing_offset_samples=7,
        )

    channel_output = apply_channel(
        samples, sample_rate, ChannelConfig(**config_kwargs)
    )

    entry: dict[str, Any] = {
        "snr_db": float(snr_db),
        "impairments": "awgn" if not with_cfo else "awgn+cfo+phase+timing",
        "detected": False,
        "classification": None,
        "classification_correct": False,
        "symbol_rate_hz": None,
        "symbol_rate_error_hz": None,
        "ber": None,
        "bit_errors": None,
        "compared_bits": None,
        "warnings": [],
    }

    try:
        result = analyze_samples(
            channel_output.samples,
            sample_rate,
            reference_bits=reference_bits,
            input_info={"file": f"synthetic-{modulation}-sweep"},
        )
    except Exception as exc:  # benchmarks must never crash the sweep
        entry["warnings"].append(f"pipeline exception: {exc}")
        return entry

    entry["detected"] = bool(result.detections)
    entry["classification"] = (result.classification or {}).get("modulation")
    entry["classification_correct"] = entry["classification"] == modulation

    rate_summary = result.symbol_rate or {}
    if rate_summary.get("symbol_rate_hz"):
        true_rate = sample_rate / 8.0
        entry["symbol_rate_hz"] = float(rate_summary["symbol_rate_hz"])
        entry["symbol_rate_error_hz"] = float(
            abs(rate_summary["symbol_rate_hz"] - true_rate)
        )

    if result.ber:
        entry["ber"] = result.ber.get("ber")
        entry["bit_errors"] = result.ber.get("bit_errors")
        entry["compared_bits"] = result.ber.get("compared_bits")

    entry["warnings"] = list(result.warnings)
    return entry


def run_sweep(
    modulation: str = "QPSK",
    snr_db_list: tuple[float, ...] = (30.0, 20.0, 10.0, 0.0),
    num_symbols: int = 256,
    seed: int = 42,
    with_cfo: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the SNR sweep for one modulation. Results are returned per SNR
    point in ascending order with honest None values where a stage
    could not produce a measurement.
    """
    results = [
        _run_one(modulation, snr_db, num_symbols, seed + i, with_cfo)
        for i, snr_db in enumerate(sorted(snr_db_list, reverse=True))
    ]
    return results
