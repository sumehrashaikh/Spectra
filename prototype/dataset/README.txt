This dataset contains real-world IQ samples for automatic radio modulation recognition.

Each HDF5 file contains:
- X: IQ samples, shape (N, 1024, 2)
- y_mod: modulation labels
- y_chan: channel type (0 = clean, 1 = multipath)
- y_snr: signal-to-noise ratio (dB)
