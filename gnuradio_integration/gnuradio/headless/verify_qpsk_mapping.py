"""Verify QPSK symbol-to-bit mapping directly from GNU Radio digital blocks.

This script executes against a working GNU Radio installation to determine the exact
mapping between constellation_decoder_cb symbol outputs and the NumPy bit slicing convention
(bit0: Re > 0, bit1: Im > 0). It writes the verified mapping to `gnuradio/qpsk_mapping.json`.
"""
import datetime
import json
import os
import sys
from pathlib import Path


def main():
    try:
        from gnuradio import blocks, digital, gr
    except ImportError as e:
        print(f"ERROR: GNU Radio is not installed or importable in this environment: {e}", file=sys.stderr)
        sys.exit(1)

    gr_version = gr.version()
    print(f"Detected GNU Radio version: {gr_version}")

    const = digital.constellation_qpsk().base()
    points = const.points()
    print(f"Constellation points from digital.constellation_qpsk(): {points}")
    print(f"Bits per symbol: {const.bits_per_symbol()}")

    # Test each point through a minimal flowgraph using constellation_decoder_cb
    # Vector Source -> Constellation Decoder -> Vector Sink
    mapping_table = {}
    verified_entries = []

    for pt in points:
        tb = gr.top_block("verify_mapping", catch_exceptions=True)
        # Send 10 identical points to ensure decoder settles
        src = blocks.vector_source_c([pt] * 10, False)
        dec = digital.constellation_decoder_cb(const)
        snk = blocks.vector_sink_b()

        tb.connect(src, dec)
        tb.connect(dec, snk)
        tb.run()

        out_bytes = snk.data()
        if not out_bytes:
            raise RuntimeError(f"No output received for constellation point {pt}")
        decoded_symbol = int(out_bytes[-1])

        # NumPy convention: bit0 = (Re > 0), bit1 = (Im > 0)
        bit0 = "1" if pt.real > 0 else "0"
        bit1 = "1" if pt.imag > 0 else "0"
        bits_str = f"{bit0}{bit1}"

        mapping_table[str(decoded_symbol)] = {
            "symbol": decoded_symbol,
            "point": {"real": float(pt.real), "imag": float(pt.imag)},
            "bits": bits_str,
            "bit0_re_gt_0": int(bit0),
            "bit1_im_gt_0": int(bit1)
        }
        verified_entries.append({
            "decoded_symbol": decoded_symbol,
            "point_real": float(pt.real),
            "point_imag": float(pt.imag),
            "bits": bits_str
        })

    mapping_payload = {
        "gnuradio_version": gr_version,
        "verification_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "convention": "NumPy convention: bit0 = (Re > 0), bit1 = (Im > 0)",
        "mapping": {str(k): v["bits"] for k, v in mapping_table.items()},
        "details": mapping_table
    }

    out_path = Path(__file__).resolve().parent.parent / "qpsk_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping_payload, f, indent=2)

    print(f"Successfully verified QPSK mapping and saved to: {out_path}")
    print("Mapping table:")
    for k, v in mapping_payload["mapping"].items():
        print(f"  Symbol {k} -> Bits {v}")


if __name__ == "__main__":
    main()
