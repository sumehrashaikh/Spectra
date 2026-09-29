"""GNU Radio integration (strategic, honest).

Uses real GNU Radio blocks via subprocess when GNU Radio is available; otherwise falls
back to the equivalent NumPy DSP in backend.dsp (documented in the result
as backend:'numpy-fallback', backend_used:'numpy').
"""
from backend.gnuradio.bridge import check_gnuradio_available as bridge_check_gr
from backend.gnuradio.bridge import run_gnuradio_demod


def check_gnuradio_available():
    is_avail, _ = bridge_check_gr()
    return is_avail


def flowgraph_info(backend_used="numpy"):
    has_gr = check_gnuradio_available()
    return {
        "name": "psk_demod_flowgraph",
        "stages": ["IQ Input", "Matched (RRC) Filter", "Costas/Carrier Recovery",
                   "Symbol Synchronization", "Demodulator", "Bit Output"],
        "backend": "gnuradio" if backend_used == "gnuradio" else "numpy-fallback",
        "backend_used": backend_used,
        "gnuradio_available": has_gr,
    }


def run_psk_flowgraph(iq, modulation="QPSK", sps=None, samp_rate=100000.0, snr_db=10.0):
    """Run PSK demodulation. Tries GNU Radio via bridge when SPS and GNU Radio are available;
    falls back cleanly to NumPy DSP when unavailable or undetermined."""
    has_gr = check_gnuradio_available()

    # If SPS is known and valid, attempt GNU Radio execution
    if has_gr and sps is not None and sps >= 2.0:
        gr_res = run_gnuradio_demod(
            iq,
            modulation=modulation,
            sps=float(sps),
            samp_rate=float(samp_rate)
        )
        if gr_res.get("status") == "ok":
            gr_res["flowgraph"] = flowgraph_info(backend_used="gnuradio")
            return gr_res

    # Fallback to NumPy
    from backend.dsp.constellation_demod import demodulate
    res = demodulate(iq, modulation, sps=sps, samp_rate=samp_rate, snr_db=snr_db)
    res["backend_used"] = "numpy"
    res["gnuradio_available"] = has_gr
    res["flowgraph"] = flowgraph_info(backend_used="numpy")
    if sps is None:
        res["flowgraph"]["skip_reason"] = "Symbol rate NOT_DETERMINED; cannot compute SPS for GNU Radio."
    return res
