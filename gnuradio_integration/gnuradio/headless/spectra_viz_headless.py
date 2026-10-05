"""SPECTRA Headless Visualization Flowgraph.

Computes FFT, PSD, and STFT waterfall matrix from complex64 IQ data using native GNU Radio blocks.
Writes binary output vectors to the specified output directory.
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Headless Visualization Flowgraph for SPECTRA")
    parser.add_argument("--input-file", type=str, help="Path to input complex64 IQ file")
    parser.add_argument("--output-dir", type=str, help="Directory to write output binary files")
    parser.add_argument("--samp-rate", type=float, default=100000.0, help="Sample rate in Hz")
    parser.add_argument("--fft-size", type=int, default=1024, help="FFT size")
    parser.add_argument("--check", action="store_true", help="Print GNU Radio version and exit")
    return parser.parse_args()


def run_viz(input_file, output_dir, samp_rate, fft_size):
    from gnuradio import blocks, fft, gr
    from gnuradio.fft import window

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_out = out_dir / "psd_frames.f32"

    tb = gr.top_block("spectra_viz_headless", catch_exceptions=True)

    # 1. Complex File Source
    src = blocks.file_source(gr.sizeof_gr_complex, input_file, False, 0, 0)

    # 2. Stream to Vector
    s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)

    # 3. FFT (Forward, Blackman-Harris window, Shifted)
    win = window.blackmanharris(fft_size)
    fft_block = fft.fft_vcc(fft_size, True, win, True, 1)

    # 4. Complex to Mag Squared (Power)
    c2mag = blocks.complex_to_mag_squared(fft_size)

    # 5. File Sink (float32 vector)
    sink = blocks.file_sink(gr.sizeof_float * fft_size, str(psd_out), False)
    sink.set_unbuffered(True)

    # Connect
    tb.connect(src, s2v)
    tb.connect(s2v, fft_block)
    tb.connect(fft_block, c2mag)
    tb.connect(c2mag, sink)

    # Run to completion (EOF of input file)
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

    if not args.input_file or not args.output_dir:
        print("Error: --input-file and --output-dir are required.", file=sys.stderr)
        sys.exit(1)

    try:
        run_viz(
            input_file=args.input_file,
            output_dir=args.output_dir,
            samp_rate=args.samp_rate,
            fft_size=args.fft_size
        )
    except Exception as e:
        print(f"Visualization flowgraph failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
