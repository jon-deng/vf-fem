"""
Don't know what to do with this yet.
"""

import numpy as np

def get_dynamic_fluid_props(fluid_props, time):
    """
    Returns fluid properties at specified time given a time varying set of properties.

    Parameters
    ----------
    fluid_props : dict
        Time varying set of fluid properties. If a property called 'key' is time varying, there
        should be a key called 'key_time' that stores time points at which the property changes.
        fluid_props[]
    """
    out = {}

    dynamic_properties = ['p_sub', 'p_sup']
    for key in dynamic_properties:
        if f'{key}_time' in fluid_props:
            out[key] = np.interp(time, fluid_props[f'{key}_time'], fluid_props[key])
        else:
            out[key] = fluid_props[key]

    for key, value in fluid_props.items():
        if key not in dynamic_properties:
            out[key] = value

    return out

def get_fundamental_freq_phase_amp(f, sample_freq):
    """
    Return an estimate of fundamental frequency and phase using DFT

    Parameters
    ----------
    f : array_like of float
        Discrete (real) signals sampled at certain frequency.
    sample_freq : float
        Sampling frequency in [Hz]

    Returns
    -------
    freq0, phi0, a0 : float
        Fundamental frequency [Hz], phase shift [time], and amplitude [signal] estimates
    """
    N = f.size

    dft_f = np.fft.rfft(f)
    freq = np.arange(N//2+1) / (N/sample_freq)
    phase = np.angle(dft_f)
    mag = np.abs(dft_f)

    idx_f0 = np.argmax(mag)

    f0 = freq[idx_f0]
    tphase0 = phase[idx_f0]/(2*np.pi) * 1/f0
    amp0 = 2*mag[idx_f0]/N

    return f0, tphase0, amp0
