import numpy as np
from dsp_gnuradio.bridge import GNURadioBridge

def main():
    print("=== GNU Radio Integration Quick Test ===")
    bridge = GNURadioBridge()
    available = bridge.is_available()
    print(f"GNU Radio Environment Detected: {available}")
    
    # Generate a simple test QPSK signal
    symbols = np.random.choice([1+1j, -1+1j, -1-1j, 1-1j], size=500)
    sps = 4
    samples = np.repeat(symbols, sps).astype(np.complex64)
    noise = (np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))) * 0.05
    rx_signal = (samples + noise).astype(np.complex64)
    
    print(f"Executing demodulation on {len(rx_signal)} samples...")
    result = bridge.demodulate(
        samples=rx_signal,
        sample_rate=2_000_000.0,
        mod_type="QPSK",
        sps=4.0,
        excess_bw=0.35
    )
    
    if result.success:
        print("\n--- SUCCESS ---")
        print(f"EVM: {result.evm_percent:.2f}%")
        print(f"Recovered Symbols: {len(result.symbols)}")
        print(f"Bitstream length: {len(result.bits)} bits")
    else:
        print(f"Demodulation Error: {result.error_message}")

if __name__ == '__main__':
    main()
