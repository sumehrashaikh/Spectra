"""SPECTRA Headless Demodulation Flowgraph.

This script executes a headless GNU Radio flowgraph for BPSK and QPSK demodulation.
It reads a raw complex64 IQ file and writes decoded symbols to an output file.

OUTPUT FORMAT CONTRACT:
The output file contains raw unsigned bytes (uint8), exactly ONE byte per decoded symbol,
with NO headers or metadata. This file holds demodulated constellation SYMBOLS (e.g. 0..3 for QPSK,
0..1 for BPSK), NOT raw unpacked bitstreams. Downstream consumers map these symbols to bits
using a verified mapping table.
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Headless PSK Demodulator for SPECTRA")
    parser.add_argument("--input-file", type=str, help="Path to input complex64 IQ file")
    parser.add_argument("--output-symbols-file", type=str, help="Path to write raw uint8 symbol bytes")
    parser.add_argument("--sps", type=float, default=8.0, help="Samples per symbol")
    parser.add_argument("--loop-bw", type=float, default=0.0628, help="Costas loop bandwidth in radians")
    parser.add_argument("--samp-rate", type=float, default=100000.0, help="Sample rate in Hz")
    parser.add_argument("--modulation", type=str, choices=["BPSK", "QPSK"], default="QPSK", help="Modulation scheme")
    parser.add_argument("--output-complex-file", type=str, default="", help="Optional path to write post-Costas complex64 symbols")
    parser.add_argument("--check", action="store_true", help="Print GNU Radio version and exit")
    return parser.parse_args()


def run_demod(input_file, output_file, sps, loop_bw, samp_rate, modulation, output_complex_file=""):
    from gnuradio import blocks, digital, gr

    tb = gr.top_block("spectra_demod_headless", catch_exceptions=True)

    # 1. File Source (complex64)
    src = blocks.file_source(gr.sizeof_gr_complex, input_file, False, 0, 0)

    # 2. Symbol Sync (Gardner)
    sync = digital.symbol_sync_cc(
        digital.TED_GARDNER,
        float(sps),
        float(loop_bw),
        1.0,
        1.0,
        1.5,
        1,
        digital.constellation_bpsk().base(),
        digital.IR_MMSE_8TAP,
        128,
        []
    )

    # 3. Costas Loop (order 4 for QPSK, order 2 for BPSK)
    costas_order = 4 if modulation.upper() == "QPSK" else 2
    costas = digital.costas_loop_cc(float(loop_bw), costas_order, False)

    # 4. Constellation Decoder
    if modulation.upper() == "QPSK":
        const_obj = digital.constellation_qpsk().base()
    else:
        const_obj = digital.constellation_bpsk().base()
    const_obj.set_npwr(1.0)
    decoder = digital.constellation_decoder_cb(const_obj)

    # 5. File Sink (unsigned char symbols)
    sink = blocks.file_sink(gr.sizeof_char, output_file, False)
    sink.set_unbuffered(True)

    # Connect blocks
    tb.connect(src, sync)
    tb.connect(sync, costas)
    tb.connect(costas, decoder)
    tb.connect(decoder, sink)

    # Optional tap: post-Costas synchronized complex symbols
    if output_complex_file:
        complex_sink = blocks.file_sink(gr.sizeof_gr_complex, output_complex_file, False)
        complex_sink.set_unbuffered(True)
        tb.connect(costas, complex_sink)

    # Run to completion (terminates upon EOF of src)
    tb.run()


def main():
    args = parse_args()

    if args.check:
        try:
            from gnuradio import gr
            print(f"GNU Radio {gr.version()}")
            sys.exit(0)
        except Exception as e:
            print(f"GNU Radio unavailable: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.input_file or not args.output_symbols_file:
        print("Error: --input-file and --output-symbols-file are required.", file=sys.stderr)
        sys.exit(1)

    try:
        run_demod(
            input_file=args.input_file,
            output_file=args.output_symbols_file,
            sps=args.sps,
            loop_bw=args.loop_bw,
            samp_rate=args.samp_rate,
            modulation=args.modulation,
            output_complex_file=args.output_complex_file
        )
    except Exception as e:
        print(f"Demodulation failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
